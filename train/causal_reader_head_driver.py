"""Fit fixed CPU heads on TRAIN caches, then evaluate frozen terminal heads.

No backbone loading or note generation. Fitting cannot open DEV caches; the
evaluation entrypoint requires all four terminal heads and fitted controls.
"""
from __future__ import annotations

import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from train.causal_reader_fit import (constant_logps,fit_constant,
                                    fit_position_constants,fit_head,fixed_order)
from train.causal_reader_gate import CausalFeatures,SelectiveHead,gate_inputs
from train.causal_reader_gate_driver import sources,require_frozen,require_science_release
from weekend_thinking.artifacts import atomic_json,canonical_hash,sha256_file,utc_now

ORDERS=(2026102131,2026102132)
ROLES=('B0_raw','T_raw','T_note')


def read(path):return json.loads(Path(path).read_text())


def require_release(path):
    p=read(path)
    if p['status']!='FROZEN_CAUSAL_READER_PROTOCOL' or p['sources']!=sources():
        raise ValueError('Exact frozen head protocol required')
    require_frozen(p)
    require_science_release(p,path)
    return p


def cache(directory,p,protocol_sha,cohort,role):
    """All ordered native rows, including errors/caps; no outcome filtering."""
    spec=p['data'][cohort];n=spec['rows'];h=p['hidden_size']
    if sha256_file(spec['labels_file'])!=spec['labels_sha256']:
        raise ValueError('Frozen native source labels changed')
    labels=[json.loads(line) for line in Path(spec['labels_file']).open()]
    hidden=np.empty((n,16,h),np.float32);entropy=np.empty((n,16),np.float32)
    logps=np.empty((n,16),np.float32);index=[None]*n;signatures=set()
    for part in range(p['partitions']['science']['features']):
        root=directory/cohort/role/f'part-{part}';meta=read(root/'complete.json')
        if (meta['status']!='COMPLETE_FROZEN_READER_FEATURES' or meta['command']!='features'
                or meta['phase']!='science' or meta['protocol_sha256']!=protocol_sha
                or meta['cohort']!=cohort or meta['role']!=role
                or not meta['execution_device'].startswith('xla')):
            raise ValueError('Complete actual native science cache required')
        signatures.add(json.dumps(meta['signature'],sort_keys=True))
        for name,digest in meta['files'].items():
            if sha256_file(root/name)!=digest:raise ValueError('Original native cache changed')
        records=[json.loads(line) for line in (root/'index.jsonl').open()]
        fields={name:np.load(root/(name+'.npy'),allow_pickle=False,mmap_mode='r')
                for name in ('hidden','entropy','target_logps')}
        count=n//p['partitions']['science']['features']
        expected={'hidden':(count,16,h),'entropy':(count,16),'target_logps':(count,16)}
        if len(records)!=count or meta['rows']!=count:
            raise ValueError('Incomplete frozen cache partition')
        for name,value in fields.items():
            if value.shape!=expected[name] or value.dtype!=np.float32 or not np.isfinite(value).all():
                raise ValueError('Invalid native cache shape/values')
        for local,row in enumerate(records):
            i=row['global_index']
            if i%p['partitions']['science']['features']!=part or not 0<=i<n or index[i] is not None:
                raise ValueError('Duplicate/incorrect native partition index')
            label=labels[i]
            if (row['item_id']!=label['item_id'] or row['document_id']!=label['document_id']
                    or row['target_sha256']!=canonical_hash(label['continuation16_ids'])):
                raise ValueError('Native source/target identity mismatch')
            index[i]=row;hidden[i]=fields['hidden'][local];entropy[i]=fields['entropy'][local]
            logps[i]=fields['target_logps'][local]
    if any(row is None for row in index) or len(signatures)!=1:
        raise ValueError('Incomplete cache or inconsistent actual model')
    return dict(causal=CausalFeatures(torch.from_numpy(hidden),torch.from_numpy(entropy)),
                lp=torch.from_numpy(logps).double(),index=index,labels=labels)


def inputs(raw,other,p,role):
    for first,second in zip(raw['index'],other['index']):
        if any(first[k]!=second[k] for k in ('global_index','item_id','document_id','target_sha256')):
            raise ValueError('Unpaired raw/assisted causal caches')
    return gate_inputs(raw['causal'],other['causal'],
        note_actions=[r['note_actions'] for r in other['index']] if role=='T_note' else [0]*len(other['index']),
        capped=[r['capped'] for r in other['index']] if role=='T_note' else [False]*len(other['index']),
        vocabulary_size=p['vocabulary_size'])


def fit(a,p):
    started=time.monotonic();digest=sha256_file(a.protocol)
    # This entrypoint has no DEV cohort argument or DEV cache read.
    raw=cache(a.features,p,digest,'TRAIN','B0_raw');artifacts={};controls={}
    probe_indices=sorted(range(5120),key=lambda i:hashlib.sha256(
        ('causal-reader-fixed-probe:'+raw['labels'][i]['document_id']).encode()).hexdigest())[:256]
    deadline=min(started+20*60,time.monotonic()+(
        datetime.fromisoformat(a.deadline_utc)-datetime.now(timezone.utc)).total_seconds())
    for role in ('T_note','T_raw'):
        other=cache(a.features,p,digest,'TRAIN',role);x=inputs(raw,other,p,role)
        controls[role]=dict(global_mixture=fit_constant(raw['lp'],other['lp']),
                            position_schedule=fit_position_constants(raw['lp'],other['lp']))
        for seed in ORDERS:
            root=a.output/f'{role}-order{seed}';root.mkdir()
            with (root/'training.jsonl').open('x') as logs,(root/'probe.jsonl').open('x') as probes:
                def emit(value):logs.write(json.dumps(value,allow_nan=False)+'\n');logs.flush()
                def probe(head,step):
                    with torch.no_grad():
                        ids=torch.tensor(probe_indices);g=head.gate(head(x[ids]));closed=g==0
                        helpful=other['lp'][ids]>raw['lp'][ids]
                        groups={kind:[j for j,i in enumerate(probe_indices) if raw['labels'][i]['kind']==kind]
                                for kind in ('calculation','ordinary')}
                        value=dict(step=step,closed_fraction=closed.double().mean().item(),
                            closed_helpful_tokens=(closed&helpful).sum().item(),
                            closed_other_tokens=(closed&~helpful).sum().item(),
                            closed_fraction_by_position=closed.double().mean(0).tolist(),
                            closed_fraction_by_reporting_stratum={k:closed[v].double().mean().item()
                                                                  for k,v in groups.items() if v})
                    probes.write(json.dumps(value)+'\n');probes.flush()
                def checkpoint(value):torch.save(value,root/f'step-{value["step"]:04}.pt')
                head=fit_head(x,raw['lp'],other['lp'],order=fixed_order(5120,seed),
                    deadline_monotonic=deadline,emit=emit,probe=probe,checkpoint=checkpoint)
            torch.save(head.state_dict(),root/'terminal800.pt')
            artifacts[f'{role}:{seed}']=dict(file=str((root/'terminal800.pt').resolve()),
                sha256=sha256_file(root/'terminal800.pt'))
        del other,x
    atomic_json(a.output/'complete.json',dict(status='COMPLETE_FOUR_TRAIN_ONLY_TERMINAL800_HEADS',
        protocol_sha256=digest,sources=sources(),heads=artifacts,TRAIN_controls=controls,
        fixed_TRAIN_probe_item_ids=[raw['index'][i]['item_id'] for i in probe_indices],
        no_DEV_cache_opened=True,seconds=time.monotonic()-started,completed_utc=utc_now()))


def bootstrap(values,rng):
    """Unique-source paired percentile intervals; no token pseudo-replication."""
    values=np.asarray(values,dtype=np.float64)
    if values.ndim!=2 or len(values)!=4096 or not np.isfinite(values).all():
        raise ValueError('Complete finite4096-source contrasts required')
    means=[]
    for first in range(0,5000,64):
        indices=rng.integers(0,len(values),size=(min(64,5000-first),len(values)))
        means.append(values[indices].mean(1))
    draws=np.concatenate(means)
    summary=[dict(mean=float(values[:,i].mean()),CI95=np.quantile(draws[:,i],[.025,.975]).tolist(),
                  document_SD=float(values[:,i].std(ddof=1)),
                  minimum=float(values[:,i].min()),median=float(np.median(values[:,i])))
             for i in range(values.shape[1])]
    return summary,draws


def secondary(a,p,fit_meta,raw,cohort_digest,primary_totals):
    """Altered contexts on a fixed subset, with original note metadata held."""
    chosen=p['secondary_interventions']['item_ids'];positions={r['item_id']:i for i,r in enumerate(raw['index'])}
    ids=torch.tensor([positions[k] for k in chosen]);h=p['hidden_size'];reports={};contrasts={}
    base=dict(causal=CausalFeatures(raw['causal'].hidden[ids],raw['causal'].entropy[ids]),
              lp=raw['lp'][ids],index=[raw['index'][i] for i in ids.tolist()])
    for mode in ('empty','other','oracle','corrupt'):
        hidden=np.empty((256,16,h),np.float32);entropy=np.empty((256,16),np.float32)
        lp=np.empty((256,16),np.float32);index=[None]*256;lookup={k:i for i,k in enumerate(chosen)}
        for part in range(4):
            directory=a.interventions/f'part-{part}';meta=read(directory/'complete.json')
            if (meta['status']!='COMPLETE_FIXED_SECONDARY_NOTE_INTERVENTIONS'
                    or meta['protocol_sha256']!=cohort_digest or meta['rows']!=64
                    or not meta['execution_device'].startswith('xla')
                    or not meta['metadata_held_at_original_generated_values']):
                raise ValueError('Complete fixed actual note interventions required')
            root=directory/mode
            for name in ('hidden.npy','entropy.npy','target_logps.npy','index.jsonl'):
                if sha256_file(root/name)!=meta['files'][mode+'/'+name]:
                    raise ValueError('Native intervention artifact changed')
            records=[json.loads(line) for line in (root/'index.jsonl').open()]
            fields=[np.load(root/(name+'.npy'),allow_pickle=False) for name in ('hidden','entropy','target_logps')]
            if (len(records)!=64 or [v.shape for v in fields]!=[(64,16,h),(64,16),(64,16)]
                    or any(v.dtype!=np.float32 or not np.isfinite(v).all() for v in fields)):
                raise ValueError('Complete finite secondary source fields required')
            for local,row in enumerate(records):
                i=lookup[row['item_id']]
                if index[i] is not None:raise ValueError('Repeated intervention source')
                index[i]=row;hidden[i]=fields[0][local];entropy[i]=fields[1][local];lp[i]=fields[2][local]
        if any(row is None for row in index):raise ValueError('Incomplete secondary sources')
        other=dict(causal=CausalFeatures(torch.from_numpy(hidden),torch.from_numpy(entropy)),
                   lp=torch.from_numpy(lp).double(),index=index)
        x=inputs(base,other,p,'T_note');values={}
        for seed in ORDERS:
            binding=fit_meta['heads'][f'T_note:{seed}'];head=SelectiveHead(h*2+5)
            if sha256_file(binding['file'])!=binding['sha256']:raise ValueError('Frozen terminal head changed')
            head.load_state_dict(torch.load(binding['file'],weights_only=True))
            with torch.no_grad():mixed=head.target_logps(head(x),base['lp'],other['lp'])
            values[str(seed)]=mixed.sum(-1).numpy()
        np.savez(a.output/f'secondary-{mode}-document_scores.npz',**values)
        reports[mode]={k:float(v.mean()) for k,v in values.items()}
        for seed in ORDERS:
            own=primary_totals[f'T_note:{seed}'][ids.numpy()]
            difference=own-values[str(seed)];rng=np.random.default_rng(2026102134)
            draws=np.concatenate([difference[rng.integers(0,256,size=(min(64,5000-start),256))].mean(1)
                                  for start in range(0,5000,64)])
            contrasts[f'{seed}:own-minus-{mode}']=dict(mean=float(difference.mean()),
                                                       CI95=np.quantile(draws,[.025,.975]).tolist())
    atomic_json(a.output/'secondary-note-interventions.json',dict(
        status='COMPLETE_FIXED_SECONDARY_NOTE_CONTENT_DIAGNOSIS',documents=256,
        mean_full16_log_probabilities=reports,item_ids=chosen,
        own_minus_intervention=contrasts,
        original_generated_length_stop_metadata_held=True,
        limits='Fixed hash-selected sources before model calls. These contexts are interventions, not deployed outputs, primary contrasts or evidence of faithful internal reasoning.'))


def evaluate(a,p):
    fit_meta=read(a.fitted/'complete.json');digest=sha256_file(a.protocol)
    if (fit_meta['status']!='COMPLETE_FOUR_TRAIN_ONLY_TERMINAL800_HEADS'
            or fit_meta['protocol_sha256']!=digest or fit_meta['sources']!=sources()
            or len(fit_meta['heads'])!=4 or not fit_meta['no_DEV_cache_opened']):
        raise ValueError('Freeze all four TRAIN-only terminal heads before reading DEV')
    rng=np.random.default_rng(2026102133);reports={};main_draws={};absolute={};means={}
    for cohort in ('DEV_calculation','DEV_ordinary'):
        raw=cache(a.features,p,digest,cohort,'B0_raw');totals={'B0':raw['lp'].sum(-1).numpy()}
        columns={};gates={}
        for role in ('T_note','T_raw'):
            other=cache(a.features,p,digest,cohort,role);x=inputs(raw,other,p,role)
            totals[role]=other['lp'].sum(-1).numpy()
            global_weight=fit_meta['TRAIN_controls'][role]['global_mixture']['weight']
            totals[role+':global']=constant_logps(raw['lp'],other['lp'],global_weight).sum(-1).numpy()
            schedule=fit_meta['TRAIN_controls'][role]['position_schedule']['weights']
            position=torch.stack([constant_logps(raw['lp'][:,i],other['lp'][:,i],w)
                                  for i,w in enumerate(schedule)],1)
            totals[role+':position']=position.sum(-1).numpy()
            for seed in ORDERS:
                binding=fit_meta['heads'][f'{role}:{seed}']
                if sha256_file(binding['file'])!=binding['sha256']:raise ValueError('Terminal head changed')
                head=SelectiveHead(p['hidden_size']*2+5)
                head.load_state_dict(torch.load(binding['file'],weights_only=True))
                with torch.no_grad():
                    logits=head(x);lp=head.target_logps(logits,raw['lp'],other['lp'])
                    g=head.gate(logits)
                key=f'{role}:{seed}';totals[key]=lp.sum(-1).numpy();gates[key]=g.numpy()
                np.save(a.output/f'{cohort}-{role}-{seed}-target_logps.npy',lp.numpy())
                np.save(a.output/f'{cohort}-{role}-{seed}-gates.npy',g.numpy())
            del other,x
        for seed in ORDERS:
            own=totals[f'T_note:{seed}']
            columns[f'{seed}:own-minus-B0']=own-totals['B0']
            columns[f'{seed}:own-minus-global']=own-totals['T_note:global']
            columns[f'{seed}:own-minus-raw-ensemble']=own-totals[f'T_raw:{seed}']
            columns[f'{seed}:own-minus-position-secondary']=own-totals['T_note:position']
        keys=list(columns);summaries,draws=bootstrap(np.column_stack(list(columns.values())),rng)
        reports[cohort]=dict(zip(keys,summaries));absolute[cohort]={k:float(v.mean()) for k,v in totals.items()}
        means[cohort]=[float(columns[f'{seed}:own-minus-B0'].mean()) for seed in ORDERS]
        main_draws[cohort]=draws[:,[keys.index(f'{seed}:own-minus-B0') for seed in ORDERS]]
        atomic_json(a.output/(cohort+'-routing.json'),{k:dict(closed_fraction=float((g==0).mean()),
            mean_weight_by_position=g.mean(0).tolist()) for k,g in gates.items()})
        np.savez(a.output/(cohort+'-document_scores.npz'),**totals)
        if cohort=='DEV_calculation':secondary(a,p,fit_meta,raw,digest,totals)
    decisions={}
    for seed in ORDERS:
        c=reports['DEV_calculation'];o=reports['DEV_ordinary'][f'{seed}:own-minus-B0']
        base=c[f'{seed}:own-minus-B0']
        decisions[str(seed)]=dict(base=base['mean']>=.25 and base['CI95'][0]>0,
            global_control=c[f'{seed}:own-minus-global']['CI95'][0]>0,
            raw_ensemble=c[f'{seed}:own-minus-raw-ensemble']['CI95'][0]>0,
            ordinary_guard=-o['CI95'][0]/16<=.03)
    prevalence=[]
    for q in (0.,.001,.01,.05,.1,.25,.5,.75,1.):
        mixed=q*main_draws['DEV_calculation']+(1-q)*main_draws['DEV_ordinary']
        prevalence.append(dict(hypothetical_calculation_prevalence=q,
            mean=[q*c+(1-q)*o for c,o in zip(means['DEV_calculation'],means['DEV_ordinary'])],
            CI95_by_order=np.quantile(mixed,[.025,.975],axis=0).T.tolist()))
    atomic_json(a.output/'readout.json',dict(status='COMPLETE_FROZEN_HEAD_PAIRED_READOUT',
        protocol_sha256=digest,fit_complete_sha256=sha256_file(a.fitted/'complete.json'),
        absolute_mean_log_probability_per16=absolute,contrasts=reports,
        eight_conjunctive_requirements=decisions,
        overall_pass=all(all(v.values()) for v in decisions.values()),
        prevalence_tradeoff=prevalence,natural_prevalence_measured=False,
        limits='Checkpoint-conditional4096-document paired5000-draw percentile intervals. Position control secondary. No trained backbone, autonomous discovery, serving-efficiency or natural-corpus gain claim.',
        completed_utc=utc_now()))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('fit','evaluate'))
    for name in ('protocol','features','output'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--fitted',type=Path)
    parser.add_argument('--interventions',type=Path)
    parser.add_argument('--deadline-utc',required=True)
    a=parser.parse_args();torch.set_num_threads(4);p=require_release(a.protocol)
    if datetime.now(timezone.utc)>=datetime.fromisoformat(a.deadline_utc):raise TimeoutError('Finite CPU stage deadline')
    a.output.mkdir(parents=True,exist_ok=False)
    if a.command=='fit':fit(a,p)
    elif a.fitted is None or a.interventions is None:raise ValueError('Four frozen heads and complete fixed interventions required')
    else:evaluate(a,p)


if __name__=='__main__':main()

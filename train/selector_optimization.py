"""Bounded CPU-only hard/smooth selector ablation on existing TRAIN caches.

No model loading, generation, feature extraction, DEV endpoint or backbone update.
The completed selective-reader experiment and its sources remain unchanged.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time

import numpy as np
import torch
from torch.nn import functional as F

from train.causal_reader_gate import SelectiveHead
from train.causal_reader_fit import constant_logps, fit_constant, fit_position_constants
from train.causal_reader_head_driver import cache, inputs
from weekend_thinking.artifacts import atomic_json, canonical_hash, sha256_file, utc_now
from weekend_thinking.stats import cluster_bootstrap

ROOT=Path(__file__).resolve().parents[1]
IDENTITY='selector-optimization-train-20261021-v1'
ORDERS=tuple(range(2026102141,2026102147))
GATES=('hard','smooth')
ROLES=('T_note','T_raw')
INITIAL_LOGIT=math.log(.0199/.9801)
SMOOTH_SLOPE=1.99
SMOOTH_OFFSET=math.log(.01/.99)


class SmoothSelectiveHead(SelectiveHead):
    """Smooth sigmoid calibrated to match the original value AND derivative.

    At original initialization, both gates have g=.01 and dg/du=.019701.
    There is no intentional finite-logit dead region. Floating-point saturation
    and different off-initialization geometry remain possible limitations.
    """
    def log_weights(self,logits):
        active=torch.ones_like(logits,dtype=torch.bool)
        z=SMOOTH_SLOPE*(logits[active]-INITIAL_LOGIT)+SMOOTH_OFFSET
        return active,F.logsigmoid(z),F.logsigmoid(-z)


def make_head(gate,dimensions):
    if gate not in GATES:raise ValueError('Only the two prospectively fixed gates')
    return (SelectiveHead if gate=='hard' else SmoothSelectiveHead)(dimensions)


def split_sources(labels):
    """Source-ID hash only; labels are used for balanced reporting strata."""
    if len(labels)!=5120 or len({r['document_id'] for r in labels})!=5120:
        raise ValueError('Original complete unique TRAIN sources required')
    fit=[];hold=[];probe=[]
    for kind in ('calculation','ordinary'):
        ids=[i for i,r in enumerate(labels) if r['kind']==kind]
        if len(ids)!=2560:raise ValueError('Original balanced TRAIN cache required')
        ids.sort(key=lambda i:hashlib.sha256(
            (IDENTITY+':internal-split:'+labels[i]['document_id']).encode()).hexdigest())
        hold.extend(ids[:512]);fit.extend(ids[512:])
        probe.extend(sorted(ids[512:],key=lambda i:hashlib.sha256(
            (IDENTITY+':fit-probe:'+labels[i]['document_id']).encode()).hexdigest())[:128])
    return dict(fit=sorted(fit),hold=sorted(hold),probe=sorted(probe))


def fixed_orders(documents,seed):
    if documents!=4096 or seed not in ORDERS:raise ValueError('Frozen size/order required')
    rng=np.random.default_rng(seed)
    return np.concatenate([rng.permutation(documents) for _ in range(10)])


def numerical_fit(head,features,raw,assisted,order,check,emit,probe):
    """Exactly ten matched passes/640 updates. Inputs contain FIT sources only."""
    if (features.shape[:2]!=(4096,16) or raw.shape!=(4096,16)
            or assisted.shape!=raw.shape or len(order)!=40960
            or features.dtype!=torch.float64 or features.device.type!='cpu'):
        raise ValueError('Only complete CPU internal-FIT tensors are accepted')
    if not all(torch.isfinite(v).all() for v in (features,raw,assisted)):
        raise ValueError('Finite frozen tensors required')
    for epoch in range(10):
        if not np.array_equal(np.sort(order[epoch*4096:(epoch+1)*4096]),np.arange(4096)):
            raise ValueError('Each source appears once per pass')
    optimizer=torch.optim.AdamW([{'params':[head.weight],'weight_decay':.0001},
        {'params':[head.bias],'weight_decay':0}],lr=.01,betas=(.9,.999),eps=1e-8)
    raw=raw.detach().double();assisted=assisted.detach().double();probe(head,0)
    for step in range(640):
        check();ids=torch.as_tensor(order[step*64:(step+1)*64],dtype=torch.long)
        x=features[ids];optimizer.zero_grad(set_to_none=True)
        before=torch.cat((head.weight.detach(),head.bias.detach()[None])).clone()
        loss=head.loss(x,raw[ids],assisted[ids])
        if not torch.isfinite(loss):raise RuntimeError('Nonfinite selector loss')
        loss.backward();grad=torch.nn.utils.clip_grad_norm_(head.parameters(),1.)
        if not torch.isfinite(grad):raise RuntimeError('Nonfinite selector gradient')
        optimizer.step()
        with torch.no_grad():
            u=head(x);g=head.gate(u);after=torch.cat((head.weight,head.bias[None]))
            if not torch.isfinite(after).all():raise RuntimeError('Nonfinite parameters')
            emit(dict(step=step+1,CE=loss.item(),data_gradient_norm=grad.item(),
                optimizer_step_norm=torch.linalg.vector_norm(after-before).item(),
                closed_fraction=(g==0).double().mean().item(),
                near_closed_fraction=(g<=.001).double().mean().item(),
                mean_gate=g.mean().item(),minimum_logit=u.min().item(),maximum_logit=u.max().item()))
        if step==0 or (step+1)%64==0:probe(head,step+1)
    return head


def source_hashes():
    names=('train/selector_optimization.py','train/causal_reader_gate.py',
           'train/causal_reader_fit.py','train/causal_reader_head_driver.py',
           'scripts/check_selector_optimization.py','weekend_thinking/stats.py')
    return {name:sha256_file(ROOT/name) for name in names}


def reject_completed_DEV_reads(event,args):
    """Runtime guard against accidentally consulting the completed panels."""
    if event=='open' and args and isinstance(args[0],(str,bytes,os.PathLike)):
        path=Path(os.fsdecode(args[0]))
        if any(part in ('DEV_calculation','DEV_ordinary') or part.startswith('DEV_') for part in path.parts):
            raise PermissionError('This development study may open TRAIN data only')


def load_train(p):
    """The original protocol hash binds paths; this API has no cohort argument."""
    parent=json.loads(Path(p['parent_protocol_file']).read_text())
    if sha256_file(p['parent_protocol_file'])!=p['parent_protocol_sha256']:
        raise ValueError('Parent cache provenance changed')
    # Exclude other original cohorts from the loader's visible protocol as well.
    parent={**parent,'data':{'TRAIN':parent['data']['TRAIN']}}
    raw=cache(Path(p['features_root']),parent,p['parent_protocol_sha256'],'TRAIN','B0_raw')
    split=split_sources(raw['labels'])
    if canonical_hash(split)!=p['split_sha256']:raise ValueError('Frozen source split changed')
    return parent,raw,split


def bootstrap(values,seed,*,document_ids):
    values=np.asarray(values,dtype=np.float64)
    if (values.ndim!=2 or len(values)!=512 or len(document_ids)!=512
            or len(set(document_ids))!=512 or not np.isfinite(values).all()):
        raise ValueError('512 unique internal-validation documents required per stratum')
    result=cluster_bootstrap(values,document_ids,unit='source document',seed=seed)
    return [dict(mean=float(result['mean'][i]),CI95=result['CI95'][i].tolist(),
                 resampling=result['metadata'])
            for i in range(values.shape[1])]


def run(a,p):
    started=time.monotonic();deadline=datetime.fromisoformat(a.deadline_utc)
    def check():
        if datetime.now(timezone.utc)>=deadline or time.monotonic()-started>3600:
            raise TimeoutError('Finite CPU optimization study; no automatic retry')
        if shutil.disk_usage(ROOT).free<35*2**30:raise RuntimeError('Filesystem reserve')
        if sum(v.stat().st_size for v in a.output.rglob('*') if v.is_file())>256*2**20:
            raise RuntimeError('Fixed256MiB output ceiling')
        mem=int(next(l.split()[1] for l in Path('/proc/meminfo').read_text().splitlines()
                     if l.startswith('MemAvailable:')))*1024
        if mem<512*2**30:raise RuntimeError('Host-memory reserve')
    a.output.mkdir(parents=True,exist_ok=False);check()
    parent,raw,split=load_train(p);fit_ids=torch.tensor(split['fit']);probe_ids=torch.tensor(split['probe'])
    controls={};heads={};original_manifests={}
    for role in ('B0_raw',*ROLES):
        for part in range(8):
            path=Path(p['features_root'])/'TRAIN'/role/f'part-{part}'/'complete.json'
            original_manifests[str(path)]=sha256_file(path)
    # Never score HOLD while any head remains unfitted. Probe indices are FIT only.
    for role in ROLES:
        other=cache(Path(p['features_root']),parent,p['parent_protocol_sha256'],'TRAIN',role)
        x=inputs(raw,other,parent,role);xf=x[fit_ids]
        controls[role]=dict(global_mixture=fit_constant(raw['lp'][fit_ids],other['lp'][fit_ids]),
                            position_schedule=fit_position_constants(raw['lp'][fit_ids],other['lp'][fit_ids]))
        for gate in GATES:
            for seed in ORDERS:
                check();name=f'{role}-{gate}-order{seed}';root=a.output/name;root.mkdir()
                with (root/'training.jsonl').open('x') as logs,(root/'probe.jsonl').open('x') as probes:
                    def emit(v):logs.write(json.dumps(v,allow_nan=False)+'\n');logs.flush()
                    def probe(head,step):
                        with torch.no_grad():
                            g=head.gate(head(x[probe_ids]));lp=head.target_logps(head(x[probe_ids]),raw['lp'][probe_ids],other['lp'][probe_ids])
                            values={}
                            for kind in ('calculation','ordinary'):
                                ix=[j for j,i in enumerate(split['probe']) if raw['labels'][i]['kind']==kind]
                                values[kind]=dict(mean_gate=g[ix].mean().item(),closed_fraction=(g[ix]==0).double().mean().item(),
                                    near_closed_fraction=(g[ix]<=.001).double().mean().item(),
                                    gain_per16=(lp[ix]-raw['lp'][probe_ids][ix]).sum(-1).mean().item())
                        probes.write(json.dumps(dict(step=step,fit_only=True,strata=values),allow_nan=False)+'\n');probes.flush()
                    head=numerical_fit(make_head(gate,x.shape[-1]),xf,raw['lp'][fit_ids],other['lp'][fit_ids],
                        fixed_orders(4096,seed),check,emit,probe)
                torch.save(head.state_dict(),root/'terminal640.pt')
                heads[name]=dict(file=str((root/'terminal640.pt').resolve()),sha256=sha256_file(root/'terminal640.pt'))
                print('HEAD_FROZEN',name,flush=True)
        del other,x,xf
    atomic_json(a.output/'fitted.json',dict(status='COMPLETE24_INTERNAL_FIT_TERMINAL640_HEADS',
        protocol_sha256=sha256_file(a.protocol),heads=heads,controls=controls,
        split=split,all_frozen_before_internal_validation=True,completed_utc=utc_now()))
    # Only now apply every frozen head to the fixed internal-validation split.
    validation_started_utc=utc_now()
    hold=split['hold'];ids=torch.tensor(hold);totals={'B0':raw['lp'][ids].sum(-1).numpy()}
    routing={};fit_means={}
    for role in ROLES:
        check();other=cache(Path(p['features_root']),parent,p['parent_protocol_sha256'],'TRAIN',role)
        x=inputs(raw,other,parent,role)
        totals[role]=other['lp'][ids].sum(-1).numpy()
        totals[role+':global']=constant_logps(raw['lp'][ids],other['lp'][ids],controls[role]['global_mixture']['weight']).sum(-1).numpy()
        weights=controls[role]['position_schedule']['weights']
        totals[role+':position']=torch.stack([constant_logps(raw['lp'][ids,i],other['lp'][ids,i],w)
                                            for i,w in enumerate(weights)],1).sum(-1).numpy()
        for gate in GATES:
            for seed in ORDERS:
                name=f'{role}-{gate}-order{seed}';head=make_head(gate,x.shape[-1]);binding=heads[name]
                if sha256_file(binding['file'])!=binding['sha256']:raise ValueError('Frozen head changed')
                head.load_state_dict(torch.load(binding['file'],weights_only=True))
                with torch.no_grad():
                    lp=head.target_logps(head(x[ids]),raw['lp'][ids],other['lp'][ids]);g=head.gate(head(x[ids]))
                    fit_lp=head.target_logps(head(x[fit_ids]),raw['lp'][fit_ids],other['lp'][fit_ids])
                totals[name]=lp.sum(-1).numpy();fit_means[name]={}
                np.savez_compressed(a.output/(name+'-validation.npz'),target_logps=lp.numpy(),gates=g.numpy())
                routing[name]={}
                for kind in ('calculation','ordinary'):
                    ix=[j for j,i in enumerate(hold) if raw['labels'][i]['kind']==kind]
                    fix=[j for j,i in enumerate(split['fit']) if raw['labels'][i]['kind']==kind]
                    routing[name][kind]=dict(mean_gate=g[ix].mean().item(),closed_fraction=(g[ix]==0).double().mean().item(),
                        near_closed_fraction=(g[ix]<=.001).double().mean().item())
                    fit_means[name][kind]=(fit_lp[fix]-raw['lp'][fit_ids][fix]).sum(-1).mean().item()
        del other,x
    reports={};decisions={}
    for kind in ('calculation','ordinary'):
        ix=[j for j,i in enumerate(hold) if raw['labels'][i]['kind']==kind];columns={}
        for gate in GATES:
            for seed in ORDERS:
                name=f'T_note-{gate}-order{seed}';own=totals[name]
                for label,baseline in [('B0','B0'),('global','T_note:global'),
                    ('position','T_note:position'),('raw-ensemble',f'T_raw-{gate}-order{seed}')]:
                    columns[name+':minus-'+label]=(own-totals[baseline])[ix]
        for role in ROLES:
            for seed in ORDERS:
                columns[f'{role}-order{seed}:smooth-minus-hard']=(totals[f'{role}-smooth-order{seed}']-totals[f'{role}-hard-order{seed}'])[ix]
        summaries=bootstrap(np.column_stack(list(columns.values())),2026102147+(kind=='ordinary'),
            document_ids=[raw['labels'][hold[j]]['document_id'] for j in ix])
        reports[kind]=dict(zip(columns,summaries))
    for gate in GATES:
        decisions[gate]={}
        for seed in ORDERS:
            name=f'T_note-{gate}-order{seed}';c=reports['calculation'];o=reports['ordinary'][name+':minus-B0']
            base=c[name+':minus-B0'];route=routing[name]['calculation']
            decisions[gate][str(seed)]=dict(base=base['mean']>=.25 and base['CI95'][0]>0,
                global_control=c[name+':minus-global']['CI95'][0]>0,
                raw_ensemble=c[name+':minus-raw-ensemble']['CI95'][0]>0,
                ordinary_guard=-o['CI95'][0]/16<=.03,
                position_secondary=c[name+':minus-position']['CI95'][0]>0,
                effectively_stalled=route['near_closed_fraction']>=.99 and base['mean']<=.01)
    for path,digest in original_manifests.items():
        if sha256_file(path)!=digest:raise ValueError('Original cache manifest changed')
    np.savez_compressed(a.output/'internal-validation-document-scores.npz',**totals)
    atomic_json(a.output/'readout.json',dict(status='COMPLETE_TRAIN_ONLY_SELECTOR_OPTIMIZATION_STUDY',
        identity=IDENTITY,protocol_sha256=sha256_file(a.protocol),fitted_sha256=sha256_file(a.output/'fitted.json'),
        reports=reports,decisions=decisions,routing=routing,fit_gain_per16=fit_means,
        all_six_primary_pass={g:all(all(v[k] for k in ('base','global_control','raw_ensemble','ordinary_guard'))
                                       for v in decisions[g].values()) for g in GATES},
        absolute_mean_log_probability_per16={kind:{k:float(v[[j for j,i in enumerate(hold) if raw['labels'][i]['kind']==kind]].mean())
            for k,v in totals.items()} for kind in ('calculation','ordinary')},
        no_DEV_cache_or_new_model_calls=True,accelerator_chip_hours=0,
        original_cache_manifests=original_manifests,seconds=time.monotonic()-started,
        internal_validation_started_utc=validation_started_utc,
        limits='Internal TRAIN-only development split, not fresh confirmation. Six paired head orders over one frozen predictor pair. No order/checkpoint/recipe selection. Smoothness does not rule out saturation or identify a unique failure cause.',
        completed_utc=utc_now()))
    print('COMPLETE_TRAIN_ONLY_SELECTOR_OPTIMIZATION_STUDY',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for field in ('protocol','output'):parser.add_argument('--'+field,type=Path,required=True)
    parser.add_argument('--deadline-utc',required=True);a=parser.parse_args()
    if os.environ.get('USE_TORCH_XLA')!='0':raise ValueError('Explicit CPU-only environment required')
    torch.set_num_threads(4);torch.set_num_interop_threads(1)
    p=json.loads(a.protocol.read_text())
    if (p['identity']!=IDENTITY or p['status']!='FROZEN_TRAIN_ONLY_OPTIMIZATION_PROTOCOL'
            or p['sources']!=source_hashes() or tuple(p['orders'])!=ORDERS
            or p['gates']!=list(GATES) or p['roles']!=list(ROLES)):
        raise ValueError('Exact prospectively frozen optimization protocol required')
    if sha256_file(p['design_file'])!=p['design_sha256']:raise ValueError('Prospective design changed')
    sys.addaudithook(reject_completed_DEV_reads)
    run(a,p)


if __name__=='__main__':main()

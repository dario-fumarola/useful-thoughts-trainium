"""I6: constrained one-scalar full-logit mixing, frozen B0, no model training.

Fit the predeclared finite grid on separate TRAIN256 correct/empty contexts.
Evaluate ONE fitted weight on the unchanged Phase1 GENERAL256+ARITHMETIC256.
All nonzero mixtures use the entire released FP32 vocabulary, never a weighted
average of sampled-token log probabilities. w=0 returns the raw branch exactly.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import time
from types import SimpleNamespace

import torch

from train import interface_oracle_probe as parent
from weekend_thinking.artifacts import atomic_json, canonical_hash, require_utc_deadline, sha256_file, utc_now

ROOT=Path(__file__).resolve().parents[1]
SCHEMA='phase1-I6-scalar-logit-mixer-v1'
GRID=(0.,1/32,1/16,1/8,1/4,3/8,1/2,3/4,1.)
TRAIN_TAX=-.4
ENDPOINT_ATOL=1e-4
MODES=('raw','empty','correct','corrupt','shuffled','other')


def read(path):return json.loads(Path(path).read_text())
def rows(path):return [json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()]
def sources():return {**parent.sources(),'train/interface_mixer_probe.py':sha256_file(__file__)}
def deadline(until):
    if datetime.now(timezone.utc)>=require_utc_deadline(until):raise TimeoutError('Finite I6 deadline reached; no automatic retry')


def first_training_rows(path):
    """Literal source order; first record per unique TRAIN document, no outcomes."""
    from train.evaluate_local import RECORD_HEADER
    result=[];seen=set()
    with Path(path).open() as stream:
        for line_number,line in enumerate(stream,1):
            header=RECORD_HEADER.match(line)
            if header is None:raise ValueError('Canonical source record header required')
            if header[3]!='train' or header[2] in seen:continue
            r=json.loads(line);past=r['prefix'];target=r['future']
            if len(past)!=512 or len(target)!=16:raise ValueError('TRAIN needs complete native512+16')
            result.append(dict(cohort='train',split='train',item_id=r['id'],document_id=r['document_id'],
                prefix_ids=past,target_ids=target,copy_note_ids=target,reanchor_ids=past[-24:],source_line=line_number))
            seen.add(r['document_id'])
            if len(result)==256:return result
    raise ValueError('Require full256 unique TRAIN documents; never shrink')


def context(row,mode,inputs,*,training=False):
    if mode=='raw':return list(row['prefix_ids'])
    if training:
        if mode not in ('empty','correct'):raise ValueError('Fit has no wrong-note or DEV objective')
        note=[] if mode=='empty' else row['target_ids']
    else:note=inputs['controls'][f'{row["cohort"]}:{row["item_id"]}']['values'][mode]
    return parent.reader_context(row,note,'I3',inputs['parts'])


def prepare(output,phase1,phase1_control,records,until):
    from transformers import AutoTokenizer
    deadline(until);old,x=parent.load(phase1);ref,_=parent.reference(phase1,old,x)
    closure=read(phase1_control/'score.closure.json')
    if not (closure['all4_exit0'] and closure['all12_owned_PID_start_identities_absent']
            and closure['assigned_devices_empty'] and closure['owned_group_survivors']==[]
            and closure['protocol_sha256']==sha256_file(phase1/'protocol.json')):
        raise ValueError('Complete physically closed original Phase1 required')
    # The original general preparation manifest pins the entire source file.
    general_manifest=next(Path(name) for name in old['input_hashes'] if Path(name).parent.name=='phase1-interface-oracle-general-dev256-v1' and Path(name).name=='manifest.json')
    g=read(general_manifest)
    if g['input_hashes'].get(str(records.resolve()))!=sha256_file(records):raise ValueError('Use original Phase1 production source bytes')
    training=first_training_rows(records);dev=x['rows']
    if {r['document_id'] for r in training}&{r['document_id'] for r in dev}:raise ValueError('TRAIN fit/DEV document overlap')
    tok=AutoTokenizer.from_pretrained(old['model_dir'],local_files_only=True,trust_remote_code=False);defined=set(tok.get_vocab().values())
    if any(type(t)is not int or t not in defined for r in training for t in r['prefix_ids']+r['target_ids']):raise ValueError('Native defined TRAIN IDs required')
    inputs=dict(training_rows=training,dev_rows=dev,controls=x['controls'],parts=x['parts'],qualification_refs=x['qualification_refs'])
    hashes={}
    for training_flag,items,modes in ((True,training,MODES[:3]),(False,dev,MODES)):
        for r in items:
            hashes[f'{r["cohort"]}:{r["item_id"]}']={m:canonical_hash(context(r,m,inputs,training=training_flag)) for m in modes}
    inputs['context_hashes']=hashes
    deadline(until);output.mkdir(parents=True,exist_ok=False);atomic_json(output/'inputs.json',inputs)
    pin_paths=[phase1/'protocol.json',phase1/'inputs.json',phase1/'cpu-reference/reference.json',
        phase1/'cpu-reference/next-logps.pt',phase1_control/'score.closure.json',general_manifest,records]
    p=dict(schema=SCHEMA,created_utc=utc_now(),source_files=sources(),inputs_sha256=sha256_file(output/'inputs.json'),
        input_hashes={str(path.resolve()):sha256_file(path) for path in pin_paths},phase1=str(phase1.resolve()),
        model_dir=old['model_dir'],assets=old['assets'],base_signature=ref['signature'],vocab_size=old['vocab_size'],
        precision='fp32',width=1024,future_width=16,shards=4,training_documents=256,dev_documents=512,
        expected_training_rows=768,expected_dev_rows=3072,grid=list(GRID),train_tax_min=TRAIN_TAX,
        fit='Highest mean TRAIN correct full16 LL among grid weights with mean TRAIN empty-minus-raw>=-.4; exact ties smaller weight; w0 exact identity is always available.',
        interface='I6: raw logits + w*(I3 note logits-raw logits)',note_interface='I3',deadline_utc=until,
        endpoint_atol=ENDPOINT_ATOL,native_atol=.002,thresholds=parent.THRESHOLDS,bootstrap=old['bootstrap'],
        train_ordered_item_ids_sha256=canonical_hash([r['item_id'] for r in training]),
        dev_ordered_ids_sha256=canonical_hash([(r['cohort'],r['item_id']) for r in dev]),
        original_coverage=old['arithmetic_publication_coverage'],cohort_document_overlap=old['cohort_document_overlap'],
        support='Full released-model head; no tokenizer mask or temperature transform',
        endpoint='w0 uses an explicit raw identity branch, checked bytewise for every TRAIN mode token vector; w1 full-logit result is compared with the actual unmixed note branch on every TRAIN row.',
        costs='Each document computes one raw forward and each note-mode forward plus mixing softmax; extra clean pass is not free unless already available.',
        limitations=['Constrained one-dimensional grid fit, not a learned MLP or exact Quiet-STaR.',
            'Future-visible full-target-copy oracle; not a useful generated-note policy.','Same DEV outcomes reused after six-interface exploration.'])
    atomic_json(output/'protocol.json',p);return p


def load(output):
    p=read(output/'protocol.json');x=read(output/'inputs.json')
    if (p.get('schema')!=SCHEMA or p['source_files']!=sources() or p['grid']!=list(GRID)
            or p['train_tax_min']!=TRAIN_TAX or p['endpoint_atol']!=ENDPOINT_ATOL
            or p['precision']!='fp32' or p['thresholds']!=parent.THRESHOLDS
            or (p['width'],p['future_width'],p['shards'],p['expected_training_rows'],p['expected_dev_rows'])!=(1024,16,4,768,3072)
            or p['inputs_sha256']!=sha256_file(output/'inputs.json')):raise ValueError('Frozen mixer recipe/source changed')
    for name,digest in p['input_hashes'].items():
        if sha256_file(name)!=digest:raise ValueError('Bound parent/source bytes changed')
    if (len(x['training_rows'])!=256 or len(x['dev_rows'])!=512
            or any(r['split']!='train' for r in x['training_rows'])
            or canonical_hash([r['item_id'] for r in x['training_rows']])!=p['train_ordered_item_ids_sha256']
            or canonical_hash([(r['cohort'],r['item_id']) for r in x['dev_rows']])!=p['dev_ordered_ids_sha256']):raise ValueError('Complete fixed TRAIN/DEV inventories required')
    return p,x


def token_logps(logits,targets):
    if logits.ndim==2:logits=logits[None]
    return logits.log_softmax(-1).gather(-1,targets[...,None])[...,0]


def mixed_token_logps(raw,note,targets,weights,raw_lp=None):
    """Actual full-V convex logit interpolation; only exact zero bypasses it."""
    w=torch.as_tensor(weights,dtype=raw.dtype,device=raw.device)[:,None,None]
    mixed=raw+(note-raw)*w
    lp=token_logps(mixed,targets.expand(len(weights),-1))
    raw_lp=token_logps(raw,targets) if raw_lp is None else raw_lp
    note_lp=token_logps(note,targets)
    # Exact identity is part of the defined endpoint, avoiding a pointless
    # alternate batched softmax rounding path when no note contributes.
    return torch.where(w[:,:,0]==0,raw_lp,lp),raw_lp,note_lp


@torch.no_grad()
def full_logits(backend,past,target):
    if not past or not 1<=len(target)<=16 or len(past)+16>1024:raise ValueError('Bounded causal context+target required')
    ids=torch.full((1,1024),151643,dtype=torch.long);ids[0,:len(past)+len(target)]=torch.tensor(past+target)
    positions=(torch.arange(16)+len(past)-1)[None].to(backend.device)
    hidden=backend.hidden(ids.to(backend.device))
    if hidden.dtype!=torch.float32:raise ValueError('Native FP32 trunk required')
    selected=hidden.gather(1,positions[:,:,None].expand(-1,-1,hidden.shape[-1]))
    logits=backend.model.lm_head(selected)
    if logits.dtype!=torch.float32:raise ValueError('Native FP32 head before float required')
    backend.sync();return logits[0]


def setup(output,p,x,shard,directory):
    from train.evaluate import asset_provenance
    from train.evaluate_local import setup_model
    if asset_provenance(Path(p['model_dir']))!=p['assets']:raise ValueError('Actual B0 assets changed')
    started=time.monotonic();backend,_,checkpoint,signature,_=setup_model(SimpleNamespace(model_dir=Path(p['model_dir']),checkpoint=None,device='xla'))
    if checkpoint is not None or signature!=p['base_signature'] or any(v.dtype!=torch.float32 for v in backend.model.parameters()):raise ValueError('Actual unchanged FP32 B0 required')
    if backend.model.get_input_embeddings().weight is not backend.model.get_output_embeddings().weight:raise ValueError('Actual tied weights required')
    model_seconds=time.monotonic()-started
    old=Path(p['phase1']);op,ox=parent.load(old);_,native=parent.reference(old,op,ox);q=x['qualification_refs'][shard]
    # Identical fixed1024 native gate as the parent, before any row publication.
    ids=torch.full((1,1024),151643,dtype=torch.long);ids[0,:len(q['context_ids'])]=torch.tensor(q['context_ids'])
    with torch.no_grad():
        h=backend.hidden(ids.to(backend.device))
        if h.dtype!=torch.float32:raise ValueError('Native qualification trunk must be FP32')
        actual=backend.model.lm_head(h[:,len(q['context_ids'])-1]).float().log_softmax(-1)
        backend.sync();actual=actual.cpu()
    error=float((actual-native[shard:shard+1]).abs().max());torch.save(actual,directory/'forward-logps.pt')
    if not math.isfinite(error) or error>.002:raise RuntimeError('Native full-head gate failed before mixer rows')
    return backend,dict(signature=signature,model_setup_seconds=model_seconds,native_full_head_max_error=error,
        forward_logps_sha256=sha256_file(directory/'forward-logps.pt'))


def verify_token_values(values,length):
    if len(values)!=length or not all(math.isfinite(v) and v<=0 for v in values):raise RuntimeError('Complete finite target LP vector required')


@torch.no_grad()
def run_shard(output,stage,shard,until):
    if shard not in range(4) or stage not in ('train','dev'):raise ValueError('Fixed four shards and stage required')
    deadline(until);p,x=load(output)
    if until!=p['deadline_utc']:raise ValueError('Frozen deadline required')
    fit=None if stage=='train' else read(output/'fit.json')
    if stage=='dev':
        if fit.get('status')!='complete_TRAIN_only_constrained_fit' or fit['protocol_sha256']!=sha256_file(output/'protocol.json'):
            raise ValueError('Completed frozen TRAIN fit required before DEV')
        weights=[fit['selected_weight']]
    else:weights=list(GRID)
    d=output/stage/f'shard-{shard}';d.mkdir(parents=True,exist_ok=False)
    report=dict(status='running',rows=0,stage=stage,shard=shard,started_utc=utc_now(),protocol_sha256=sha256_file(output/'protocol.json'))
    atomic_json(d/'run.json',report)
    try:
        backend,info=setup(output,p,x,shard,d);report.update(info)
        items=x['training_rows'] if stage=='train' else x['dev_rows'];modes=MODES[:3] if stage=='train' else MODES
        with (d/'scores.jsonl').open('x') as stream:
            for r in items[shard::4]:
                target=r['target_ids'];n=len(target);key=f'{r["cohort"]}:{r["item_id"]}'
                start=time.monotonic();raw=full_logits(backend,r['prefix_ids'],target)
                y=torch.tensor([target+[151643]*(16-n)],device=backend.device)
                raw_lp=token_logps(raw,y).reshape(-1);backend.sync();raw_values=raw_lp.cpu()[:n]
                base=dict(cohort=r['cohort'],item_id=r['item_id'],document_id=r['document_id'],target_ids=target,target_sha256=canonical_hash(target))
                values=raw_values.tolist();verify_token_values(values,n)
                stream.write(json.dumps(dict(**base,mode='raw',status='ok',token_logps=values,
                    context_sha256=canonical_hash(r['prefix_ids']),wall_seconds=time.monotonic()-start),allow_nan=False)+'\n');stream.flush();report['rows']+=1
                for mode in modes[1:]:
                    deadline(until);start=time.monotonic();past=context(r,mode,x,training=stage=='train')
                    if canonical_hash(past)!=x['context_hashes'][key][mode]:raise ValueError('Frozen I3 context changed')
                    note=full_logits(backend,past,target);mixed,raw_again,note_lp=mixed_token_logps(raw,note,y,weights,raw_lp.reshape(1,16))
                    backend.sync();mixed=mixed.cpu()[:,:n];raw_again=raw_again.cpu().reshape(-1)[:n];note_lp=note_lp.cpu().reshape(-1)[:n]
                    for v in mixed.tolist():verify_token_values(v,n)
                    verify_token_values(note_lp.tolist(),n)
                    zero_exact=None;one_error=None
                    if stage=='train':
                        zero_exact=(mixed[0].contiguous().numpy().tobytes()==raw_values.contiguous().numpy().tobytes()==raw_again.contiguous().numpy().tobytes())
                        one_error=float((mixed[-1]-note_lp).abs().max())
                        if not zero_exact or not math.isfinite(one_error) or one_error>ENDPOINT_ATOL:raise RuntimeError('All-TRAIN w0/w1 endpoint gate failed')
                    row=dict(**base,mode=mode,status='ok',context_sha256=canonical_hash(past),weights=weights,
                        token_logps=mixed.tolist(),note_endpoint_token_logps=note_lp.tolist(),
                        zero_raw_bitwise_equal=zero_exact,one_note_max_error=one_error,wall_seconds=time.monotonic()-start)
                    stream.write(json.dumps(row,allow_nan=False)+'\n');stream.flush();report['rows']+=1
                    del note,mixed
                del raw
                print(json.dumps(dict(utc=utc_now(),stage=stage,shard=shard,rows=report['rows'])),flush=True)
        report.update(status='complete',scores_sha256=sha256_file(d/'scores.jsonl'))
    except BaseException as error:report.update(status='failed',error=f'{type(error).__name__}: {error}');raise
    finally:report['finished_utc']=utc_now();atomic_json(d/'run.json',report,overwrite=True)


def complete_rows(output,stage,p):
    actual=[];pins={};expected=192 if stage=='train' else 768
    for i in range(4):
        d=output/stage/f'shard-{i}';r=read(d/'run.json')
        if (r['status']!='complete' or r['rows']!=expected or r['protocol_sha256']!=sha256_file(output/'protocol.json')
                or r['signature']!=p['base_signature'] or not math.isfinite(r['native_full_head_max_error']) or r['native_full_head_max_error']>.002
                or sha256_file(d/'scores.jsonl')!=r['scores_sha256']
                or sha256_file(d/'forward-logps.pt')!=r['forward_logps_sha256']):raise ValueError('All four immutable completed shards required')
        actual.extend(rows(d/'scores.jsonl'));pins[str(d/'run.json')]=sha256_file(d/'run.json');pins[str(d/'scores.jsonl')]=r['scores_sha256']
    return actual,pins


def select_weight(actual,items,context_hashes):
    index={};expected={(r['item_id'],m) for r in items for m in MODES[:3]}
    records={r['item_id']:r for r in items}
    for v in actual:
        key=(v['item_id'],v['mode'])
        if key not in expected or key in index or v['status']!='ok' or v['cohort']!='train':raise ValueError('Complete unique TRAIN-only rows required')
        r=records[v['item_id']]
        if (v['document_id']!=r['document_id'] or v['target_ids']!=r['target_ids']
                or v['target_sha256']!=canonical_hash(r['target_ids'])
                or v['context_sha256']!=context_hashes[f'train:{r["item_id"]}'][v['mode']]):raise ValueError('TRAIN identity/target/context changed')
        if v['mode']!='raw' and (v['weights']!=list(GRID) or v['zero_raw_bitwise_equal'] is not True
                or not math.isfinite(v['one_note_max_error']) or v['one_note_max_error']>ENDPOINT_ATOL):raise ValueError('Every TRAIN endpoint must pass')
        index[key]=v
    if set(index)!=expected:raise ValueError('Never shrink fit denominator')
    grid=[]
    for j,w in enumerate(GRID):
        raw=correct=empty=0.
        for r in items:
            a=index[(r['item_id'],'raw')];b=index[(r['item_id'],'correct')];e=index[(r['item_id'],'empty')]
            for v in (a,b,e):
                if v['target_ids']!=r['target_ids']:raise ValueError('TRAIN target changed')
            av=a['token_logps'];bv=b['token_logps'][j];ev=e['token_logps'][j]
            for v in (av,bv,ev):verify_token_values(v,16)
            raw+=sum(av);correct+=sum(bv);empty+=sum(ev)
        z=dict(weight=w,raw_mean=raw/len(items),correct_mean=correct/len(items),empty_mean=empty/len(items))
        z['empty_minus_raw']=z['empty_mean']-z['raw_mean'];z['eligible']=z['empty_minus_raw']>=TRAIN_TAX;grid.append(z)
    if not grid[0]['eligible'] or grid[0]['empty_minus_raw']!=0:raise ValueError('w0 exact fallback failed')
    chosen=min((v for v in grid if v['eligible']),key=lambda v:(-v['correct_mean'],v['weight']))
    return dict(grid=grid,selected_weight=chosen['weight'],all512_note_endpoint_rows_pass=True,
        max_one_endpoint_error=max(v['one_note_max_error'] for v in actual if v['mode']!='raw'))


def fit(output):
    p,x=load(output);actual,pins=complete_rows(output,'train',p);result=select_weight(actual,x['training_rows'],x['context_hashes'])
    result.update(status='complete_TRAIN_only_constrained_fit',protocol_sha256=sha256_file(output/'protocol.json'),
        inputs_sha256=sha256_file(output/'inputs.json'),input_hashes=pins,training_documents=256,
        fit_rule=p['fit'],finished_utc=utc_now())
    atomic_json(output/'fit.json',result);return result


def summarize(output):
    p,x=load(output);f=read(output/'fit.json');train,pins=complete_rows(output,'train',p)
    recomputed=select_weight(train,x['training_rows'],x['context_hashes'])
    if any(f[k]!=v for k,v in recomputed.items()):raise ValueError('Frozen TRAIN selected weight changed')
    actual,dev_pins=complete_rows(output,'dev',p);index={};expected={(r['cohort'],r['item_id'],m) for r in x['dev_rows'] for m in MODES}
    records={(r['cohort'],r['item_id']):r for r in x['dev_rows']}
    for v in actual:
        key=(v['cohort'],v['item_id'],v['mode']);r=records.get(key[:2])
        if key not in expected or key in index or v['status']!='ok' or v['target_ids']!=r['target_ids'] or v['document_id']!=r['document_id']:raise ValueError('Complete unchanged DEV cells required')
        if v['mode']=='raw':lp=v['token_logps']
        else:
            if v['weights']!=[f['selected_weight']]:raise ValueError('DEV must use only TRAIN selected weight')
            lp=v['token_logps'][0]
        verify_token_values(lp,len(r['target_ids']))
        if v['context_sha256']!=x['context_hashes'][f'{key[0]}:{key[1]}'][key[2]]:raise ValueError('DEV context changed')
        index[key]=dict(full_span_ll=sum(lp),k1_ll=lp[0],numeric_only_ll=sum(a for a,on in zip(lp,r.get('numeric_token_mask',[])) if on))
    if set(index)!=expected:raise ValueError('All3072 DEV cells required')
    from weekend_thinking.stats import paired_cluster_bootstrap
    results={}
    for cohort in parent.COHORTS:
        items=[r for r in x['dev_rows'] if r['cohort']==cohort];absolute={};contrasts={}
        for metric in (('full_span_ll','k1_ll','numeric_only_ll') if cohort=='arithmetic' else ('full_span_ll','k1_ll')):
            def values(mode):return [dict(item_id=r['item_id'],document_id=r['document_id'],score=index[(cohort,r['item_id'],mode)][metric]) for r in items]
            absolute[metric]={m:sum(v['score'] for v in values(m))/256 for m in MODES};cc={}
            for a,b in (('raw','empty'),('raw','correct'),('empty','correct'),('corrupt','correct'),('shuffled','correct'),('other','correct')):
                cc[f'{b}_minus_{a}']=paired_cluster_bootstrap(values(a),values(b),
                    repeats=p['bootstrap']['repeats'],seed=p['bootstrap']['seed'])
            contrasts[metric]=cc
        results[cohort]=dict(documents=256,absolute=absolute,contrasts=contrasts,
            passes=all(contrasts['full_span_ll'][k]['difference_b_minus_a']>=bound for k,bound in parent.THRESHOLDS.items()),
            raw_fullspan_headroom=-absolute['full_span_ll']['raw'])
    result=dict(status='complete',rows=len(actual),failures=0,cohorts=results,selected_weight=f['selected_weight'],
        protocol_sha256=sha256_file(output/'protocol.json'),fit_sha256=sha256_file(output/'fit.json'),
        input_hashes={**pins,**dev_pins},finished_utc=utc_now(),limitations=p['limitations'])
    atomic_json(output/'summary.json',result);return result


def main():
    a=argparse.ArgumentParser(description=__doc__);a.add_argument('command',choices=('prepare','train','fit','dev','summarize'))
    a.add_argument('--output',type=Path,required=True);a.add_argument('--phase1',type=Path);a.add_argument('--phase1-control',type=Path)
    a.add_argument('--records',type=Path);a.add_argument('--shard-index',type=int);a.add_argument('--deadline-utc')
    v=a.parse_args()
    if v.command in ('prepare','train','dev') and v.deadline_utc is None:a.error('Finite --deadline-utc required')
    if v.command=='prepare':prepare(v.output,v.phase1,v.phase1_control,v.records,v.deadline_utc)
    elif v.command in ('train','dev'):run_shard(v.output,v.command,v.shard_index,v.deadline_utc)
    elif v.command=='fit':fit(v.output)
    else:summarize(v.output)


if __name__=='__main__':main()

"""Independent NumPy reconstruction of the TRAIN-only hard/smooth ablation.

Shared original cache loader/source-split definition; separate feature, gate,
mixture and source-paired statistical arithmetic. No fitting or model calls.
"""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import json
import math
import sys
from pathlib import Path
import numpy as np
import torch

from train.causal_reader_head_driver import cache
from train.selector_optimization import split_sources,reject_completed_DEV_reads
from weekend_thinking.artifacts import atomic_json,sha256_file,canonical_hash,utc_now

ORDERS=tuple(range(2026102141,2026102147))
def read(path):return json.loads(Path(path).read_text())
def close(a,b,name):
    if not np.allclose(a,b,rtol=0,atol=1e-9):raise ValueError('Independent mismatch: '+name)
def blend(raw,other,logg,lograw):return np.logaddexp(raw+lograw,other+logg)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for field in ('protocol','result','output'):parser.add_argument('--'+field,type=Path,required=True)
    parser.add_argument('--deadline-utc',required=True);a=parser.parse_args();torch.set_num_threads(4)
    sys.addaudithook(reject_completed_DEV_reads)
    deadline=datetime.fromisoformat(a.deadline_utc)
    def check():
        if datetime.now(timezone.utc)>=deadline:raise TimeoutError('Finite verification deadline')
    p=read(a.protocol);r=read(a.result/'readout.json');f=read(a.result/'fitted.json')
    if (r['status']!='COMPLETE_TRAIN_ONLY_SELECTOR_OPTIMIZATION_STUDY'
            or r['protocol_sha256']!=sha256_file(a.protocol)
            or r['fitted_sha256']!=sha256_file(a.result/'fitted.json')
            or f['completed_utc']>r['internal_validation_started_utc']
            or not f['all_frozen_before_internal_validation'] or len(f['heads'])!=24):
        raise ValueError('Complete fixed24-head development experiment required')
    if sha256_file(p['parent_protocol_file'])!=p['parent_protocol_sha256']:
        raise ValueError('Original protocol changed')
    parent=read(p['parent_protocol_file']);parent={**parent,'data':{'TRAIN':parent['data']['TRAIN']}}
    raw=cache(Path(p['features_root']),parent,p['parent_protocol_sha256'],'TRAIN','B0_raw')
    split=split_sources(raw['labels'])
    if split!=f['split'] or canonical_hash(split)!=p['split_sha256']:raise ValueError('Source split changed')
    hold=split['hold'];lp0=raw['lp'].numpy()[hold];totals={'B0':lp0.sum(1)};routing={}
    for role in ('T_note','T_raw'):
        check();other=cache(Path(p['features_root']),parent,p['parent_protocol_sha256'],'TRAIN',role)
        lp1=other['lp'].numpy()[hold];totals[role]=lp1.sum(1)
        controls=f['controls'][role]
        for name,w in [('global',controls['global_mixture']['weight']),
                       ('position',controls['position_schedule']['weights'])]:
            with np.errstate(divide='ignore'):
                totals[role+':'+name]=blend(lp0,lp1,np.log(w),np.log1p(-np.asarray(w))).sum(1)
        for gate in ('hard','smooth'):
            for seed in ORDERS:
                check();name=f'{role}-{gate}-order{seed}';binding=f['heads'][name]
                if sha256_file(binding['file'])!=binding['sha256']:raise ValueError('Terminal changed')
                logs=[json.loads(l) for l in (Path(binding['file']).parent/'training.jsonl').open()]
                if [x['step'] for x in logs]!=list(range(1,641)):raise ValueError('Exactly640updates required')
                state=torch.load(binding['file'],weights_only=True);w=state['weight'].numpy();b=state['bias'].item()
                out=np.empty_like(lp0);gates=np.empty_like(lp0)
                for start in range(0,1024,128):
                    selected=hold[start:start+128];sl=slice(start,start+len(selected))
                    def norm(v):
                        v=v.numpy()[selected].astype(np.float64)
                        return v/np.sqrt(np.mean(v*v,axis=-1,keepdims=True)+1e-6)
                    u=norm(raw['causal'].hidden)@w[:parent['hidden_size']]+norm(other['causal'].hidden)@w[parent['hidden_size']:2*parent['hidden_size']]+b
                    u+=raw['causal'].entropy.numpy()[selected].astype(np.float64)/math.log(parent['vocabulary_size'])*w[-5]
                    u+=other['causal'].entropy.numpy()[selected].astype(np.float64)/math.log(parent['vocabulary_size'])*w[-4]
                    if role=='T_note':
                        u+=np.array([other['index'][i]['note_actions']/128 for i in selected])[:,None]*w[-3]
                        u+=np.array([float(other['index'][i]['capped']) for i in selected])[:,None]*w[-2]
                    u+=np.arange(16)[None]/16*w[-1]
                    if gate=='hard':
                        active=u>math.log(.01/.99);g=np.zeros_like(u);mixed=lp0[sl].copy()
                        distance=u[active]-math.log(.01/.99)
                        denominator=np.logaddexp(math.log(.99),math.log(.01)+distance)
                        lg=math.log(.01)+distance+np.log(-np.expm1(-distance))-denominator
                        g[active]=np.exp(lg)
                        mixed[active]=blend(lp0[sl][active],lp1[sl][active],lg,-denominator)
                    else:
                        z=1.99*(u-math.log(.0199/.9801))+math.log(.01/.99)
                        logg=-np.logaddexp(0.,-z);lograw=-np.logaddexp(0.,z)
                        g=np.exp(logg);mixed=blend(lp0[sl],lp1[sl],logg,lograw)
                    out[sl]=mixed;gates[sl]=g
                saved=np.load(a.result/(name+'-validation.npz'),allow_pickle=False)
                close(out,saved['target_logps'],name+' target logps');close(gates,saved['gates'],name+' gates')
                totals[name]=out.sum(1);routing[name]={}
                for kind in ('calculation','ordinary'):
                    ix=[j for j,i in enumerate(hold) if raw['labels'][i]['kind']==kind]
                    routing[name][kind]=dict(near_closed_fraction=float((gates[ix]<=.001).mean()))
        del other
    saved=np.load(a.result/'internal-validation-document-scores.npz',allow_pickle=False)
    if set(saved.files)!=set(totals):raise ValueError('All absolute arms required')
    reports={}
    for kind in ('calculation','ordinary'):
        check();ix=[j for j,i in enumerate(hold) if raw['labels'][i]['kind']==kind];columns={}
        for key,value in totals.items():
            close(value,saved[key],key+' document scores')
            close(value[ix].mean(),r['absolute_mean_log_probability_per16'][kind][key],key+' absolute mean')
        for gate in ('hard','smooth'):
            for seed in ORDERS:
                name=f'T_note-{gate}-order{seed}'
                for label,baseline in [('B0','B0'),('global','T_note:global'),('position','T_note:position'),
                    ('raw-ensemble',f'T_raw-{gate}-order{seed}')]:
                    columns[name+':minus-'+label]=(totals[name]-totals[baseline])[ix]
        for role in ('T_note','T_raw'):
            for seed in ORDERS:
                columns[f'{role}-order{seed}:smooth-minus-hard']=(totals[f'{role}-smooth-order{seed}']-totals[f'{role}-hard-order{seed}'])[ix]
        values=np.column_stack(list(columns.values()));rng=np.random.default_rng(2026102147+(kind=='ordinary'));draws=[]
        for start in range(0,5000,32):
            ids=rng.integers(0,512,size=(min(32,5000-start),512));draws.append(values[ids].mean(1))
        draws=np.concatenate(draws);reports[kind]={}
        for i,(key,v) in enumerate(columns.items()):
            ci=np.quantile(draws[:,i],[.025,.975]);old=r['reports'][kind][key]
            close([v.mean(),*ci],[old['mean'],*old['CI95']],key+' paired statistics')
            reports[kind][key]=dict(mean=float(v.mean()),CI95=ci.tolist())
    decisions={};all_pass={}
    for gate in ('hard','smooth'):
        decisions[gate]={}
        for seed in ORDERS:
            name=f'T_note-{gate}-order{seed}';c=reports['calculation'];o=reports['ordinary'][name+':minus-B0'];base=c[name+':minus-B0']
            decisions[gate][str(seed)]=dict(base=base['mean']>=.25 and base['CI95'][0]>0,
                global_control=c[name+':minus-global']['CI95'][0]>0,
                raw_ensemble=c[name+':minus-raw-ensemble']['CI95'][0]>0,
                ordinary_guard=-o['CI95'][0]/16<=.03,
                position_secondary=c[name+':minus-position']['CI95'][0]>0,
                effectively_stalled=routing[name]['calculation']['near_closed_fraction']>=.99 and base['mean']<=.01)
        all_pass[gate]=all(all(v[k] for k in ('base','global_control','raw_ensemble','ordinary_guard')) for v in decisions[gate].values())
    if decisions!=r['decisions'] or all_pass!=r['all_six_primary_pass']:raise ValueError('Independent decisions disagree')
    atomic_json(a.output,dict(status='PASS_INDEPENDENT_TRAIN_ONLY_OPTIMIZATION_RECONSTRUCTION',
        protocol_sha256=sha256_file(a.protocol),readout_sha256=sha256_file(a.result/'readout.json'),
        heads=24,internal_validation_documents=1024,paired_draws=5000,all_six_primary_pass=all_pass,
        no_model_or_DEV_calls=True,completed_utc=utc_now(),
        limits='Shared original TRAIN cache loader/source-split definition; independent NumPy causal-feature, gate, mixture and statistical reconstruction. Does not externally replicate or verify every optimizer trajectory.'))
    print('PASS_INDEPENDENT_TRAIN_ONLY_OPTIMIZATION_RECONSTRUCTION',flush=True)

if __name__=='__main__':main()

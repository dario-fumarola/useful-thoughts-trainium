"""Reconstruct fresh frozen-head predictions with NumPy; no fitting/models."""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import json
import math
from pathlib import Path
import numpy as np
import torch

from train.causal_reader_head_driver import cache
from weekend_thinking.artifacts import sha256_file


def close(actual,expected,name):
    if not np.allclose(actual,expected,rtol=0,atol=1e-9):
        raise ValueError('Independent confirmation mismatch: '+name)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('protocol','features','readout','output'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--deadline-utc',required=True);a=parser.parse_args();torch.set_num_threads(4)
    deadline=datetime.fromisoformat(a.deadline_utc)
    def check():
        if datetime.now(timezone.utc)>=deadline: raise TimeoutError('Finite independent reconstruction')
    p=json.loads(a.protocol.read_text());r=json.loads((a.readout/'readout.json').read_text())
    if r['protocol_sha256']!=sha256_file(a.protocol) or r['status']!='COMPLETE_FROZEN_READER_FRESH_CONFIRMATION':
        raise ValueError('Complete fixed confirmation required')
    fitted=json.loads(Path(p['fitted_heads']['file']).read_text())
    if sha256_file(p['fitted_heads']['file'])!=p['fitted_heads']['sha256']:
        raise ValueError('Original fixed heads/controls changed')
    orders=range(2026102141,2026102147);reports={}
    for cohort in ('CONFIRM_calculation','CONFIRM_natural'):
        check();raw=cache(a.features,p,sha256_file(a.protocol),cohort,'B0_raw')
        lp0=raw['lp'].numpy();totals={'B0':lp0.sum(1)}
        gates=np.load(a.readout/(cohort+'-gates.npz'),allow_pickle=False)
        target_logps=np.load(a.readout/(cohort+'-targets.npz'),allow_pickle=False)
        saved=np.load(a.readout/(cohort+'-document-scores.npz'),allow_pickle=False)
        for role in ('T_note','T_raw'):
            other=cache(a.features,p,sha256_file(a.protocol),cohort,role);lp1=other['lp'].numpy()
            totals[role]=lp1.sum(1);controls=fitted['controls'][role]
            for mode,w in [('global',controls['global_mixture']['weight']),
                           ('position',controls['position_schedule']['weights'])]:
                with np.errstate(divide='ignore'):
                    totals[role+':'+mode]=np.logaddexp(lp0+np.log1p(-np.asarray(w)),lp1+np.log(w)).sum(1)
            for order in orders:
                check();name=f'{role}-hard-order{order}';binding=p['heads'][name]
                if binding!=fitted['heads'][name] or sha256_file(binding['file'])!=binding['sha256']:
                    raise ValueError('Frozen terminal head changed')
                state=torch.load(binding['file'],weights_only=True);w=state['weight'].numpy();b=state['bias'].item()
                output=np.empty_like(lp0);route=np.empty_like(lp0);h=p['hidden_size']
                for first in range(0,len(lp0),128):
                    sl=slice(first,first+128)
                    def norm(tensor):
                        x=tensor.numpy()[sl].astype(np.float64)
                        return x/np.sqrt((x*x).mean(-1,keepdims=True)+1e-6)
                    u=norm(raw['causal'].hidden)@w[:h]+norm(other['causal'].hidden)@w[h:2*h]+b
                    u+=raw['causal'].entropy.numpy()[sl].astype(np.float64)/math.log(p['vocabulary_size'])*w[-5]
                    u+=other['causal'].entropy.numpy()[sl].astype(np.float64)/math.log(p['vocabulary_size'])*w[-4]
                    if role=='T_note':
                        u+=np.array([x['note_actions']/128 for x in other['index'][sl]])[:,None]*w[-3]
                        u+=np.array([float(x['capped']) for x in other['index'][sl]])[:,None]*w[-2]
                    u+=np.arange(16)[None]/16*w[-1]
                    active=u>math.log(.01/.99);g=np.zeros_like(u);mixed=lp0[sl].copy()
                    distance=u[active]-math.log(.01/.99)
                    denominator=np.logaddexp(math.log(.99),math.log(.01)+distance)
                    lg=math.log(.01)+distance+np.log(-np.expm1(-distance))-denominator
                    g[active]=np.exp(lg)
                    mixed[active]=np.logaddexp(lp0[sl][active]-denominator,lp1[sl][active]+lg)
                    output[sl]=mixed;route[sl]=g
                close(output,target_logps[name],name+' token probabilities')
                close(route,gates[name],name+' mixture weights')
                totals[name]=output.sum(1)
            del other
        if set(totals)!=set(saved.files):raise ValueError('Every absolute arm required')
        for name,value in totals.items():
            close(value,saved[name],name+' document scores')
            close(value.mean(),r['absolute'][cohort][name],name+' absolute mean')
        columns={}
        for order in orders:
            own=totals[f'T_note-hard-order{order}']
            for baseline in ('B0','T_note:global','T_note:position','T_raw:global','T_raw:position','matched_raw'):
                other=f'T_raw-hard-order{order}' if baseline=='matched_raw' else baseline
                columns[f'{order}:minus-{baseline}']=own-totals[other]
        for role in ('T_note','T_raw'):
            for mode in ('global','position'):
                columns[f'{role}:{mode}:minus-B0']=totals[role+':'+mode]-totals['B0']
        values=np.column_stack(list(columns.values()));n=len(values)
        rng=np.random.default_rng(2026102151+(cohort=='CONFIRM_natural'));draws=[]
        for first in range(0,5000,32):
            indices=rng.integers(0,n,size=(min(32,5000-first),n));draws.append(values[indices].mean(1))
        draws=np.concatenate(draws);reports[cohort]={}
        for i,name in enumerate(columns):
            v=dict(mean=float(values[:,i].mean()),CI95=np.quantile(draws[:,i],[.025,.975]).tolist())
            expected=r['reports'][cohort][name]
            close([v['mean'],*v['CI95']],[expected['mean'],*expected['CI95']],name+' source-paired interval')
            reports[cohort][name]=v
    decisions={}
    for order in orders:
        key=f'{order}:minus-';c=reports['CONFIRM_calculation'];n=reports['CONFIRM_natural']
        controls=('T_note:global','T_note:position','T_raw:global','T_raw:position','matched_raw')
        decisions[str(order)]=dict(calculation_base=c[key+'B0']['mean']>=.25 and c[key+'B0']['CI95'][0]>0,
            calculation_all_controls=all(c[key+b]['CI95'][0]>0 for b in controls),
            natural_gain=n[key+'B0']['mean']>=.01 and n[key+'B0']['CI95'][0]>0,
            natural_all_controls=all(n[key+b]['CI95'][0]>0 for b in controls),
            natural_old_retention_guard=-n[key+'B0']['CI95'][0]/16<=.03)
    if decisions!=r['decisions']:raise ValueError('Independent per-order decisions disagree')
    calc=all(v['calculation_base'] and v['calculation_all_controls'] for v in decisions.values())
    natural=all(v['natural_gain'] and v['natural_all_controls'] for v in decisions.values())
    if calc!=r['calculation_confirmation_all_six'] or natural!=r['natural_prediction_all_six']:
        raise ValueError('Independent joint verdicts disagree')
    result=dict(status='PASS_INDEPENDENT_FRESH_READER_CONFIRMATION_RECONSTRUCTION',
        protocol_sha256=sha256_file(a.protocol),readout_sha256=sha256_file(a.readout/'readout.json'),
        calculation_confirmation_all_six=calc,natural_prediction_all_six=natural,
        heads=12,documents=5120,contrasts=80,
        limits='Shares original cache loader; independent NumPy features/gates/mixtures/statistics. No optimizer or model execution; not external replication.')
    with a.output.open('x') as stream:stream.write(json.dumps(result,indent=2)+'\n')
    print(result['status'],flush=True)


if __name__=='__main__':main()

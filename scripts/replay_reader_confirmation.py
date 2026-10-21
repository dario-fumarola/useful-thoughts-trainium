"""Replay source-free fresh-confirmation statistics using NumPy, no models."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def close(a,b,name):
    if not np.allclose(a,b,rtol=0,atol=1e-9):
        raise ValueError('Public confirmation mismatch: '+name)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);a=parser.parse_args();root=a.evidence
    manifest=json.loads((root/'manifest.json').read_text())
    for name,expected in manifest['files'].items():
        if hashlib.sha256((root/name).read_bytes()).hexdigest()!=expected:
            raise ValueError('Evidence digest: '+name)
    r=json.loads((root/'readout.json').read_text())
    orders=tuple(range(2026102141,2026102147))
    baselines=('B0','T_note:global','T_note:position','T_raw:global','T_raw:position','matched_raw')
    reports={};all_ids=set()
    for cohort,n in [('CONFIRM_calculation',1024),('CONFIRM_natural',4096)]:
        identities=json.loads((root/(cohort+'-identities.json')).read_text())
        ids={row['document_id'] for row in identities}
        if len(identities)!=n or len(ids)!=n or ids & all_ids:
            raise ValueError('Complete disjoint source identities required')
        all_ids.update(ids)
        totals=np.load(root/(cohort+'-document-scores.npz'),allow_pickle=False)
        gates=np.load(root/(cohort+'-gates.npz'),allow_pickle=False)
        targets=np.load(root/(cohort+'-targets.npz'),allow_pickle=False)
        heads={f'{role}-hard-order{order}' for role in ('T_note','T_raw') for order in orders}
        if set(gates.files)!=heads or set(targets.files)!=heads:
            raise ValueError('Every frozen treatment/control head must be present')
        absolute_arms=heads|{'B0','T_note','T_raw','T_note:global','T_note:position',
                            'T_raw:global','T_raw:position'}
        if set(totals.files)!=absolute_arms:
            raise ValueError('All 19 absolute prediction arms are required')
        for name in totals.files:
            if totals[name].shape!=(n,) or not np.isfinite(totals[name]).all():
                raise ValueError('Complete finite document scores required')
            close(totals[name].mean(),r['absolute'][cohort][name],name+' absolute mean')
        for name in targets.files:
            if targets[name].shape!=(n,16) or gates[name].shape!=(n,16):raise ValueError('All target positions')
            if not np.isfinite(gates[name]).all() or not ((gates[name]>=0)&(gates[name]<=1)).all():
                raise ValueError('Valid gate contributions required')
            close(targets[name].sum(1),totals[name],name+' complete target sum')
            for key,value in dict(mean_gate=gates[name].mean(),closed_fraction=(gates[name]==0).mean(),
                                  near_closed_fraction=(gates[name]<=.001).mean()).items():
                close(value,r['routing'][cohort][name][key],name+' routing')
        columns={}
        for order in orders:
            own=totals[f'T_note-hard-order{order}']
            for baseline in baselines:
                other=f'T_raw-hard-order{order}' if baseline=='matched_raw' else baseline
                columns[f'{order}:minus-{baseline}']=own-totals[other]
        for role in ('T_note','T_raw'):
            for mode in ('global','position'):
                columns[f'{role}:{mode}:minus-B0']=totals[role+':'+mode]-totals['B0']
        values=np.column_stack(list(columns.values()));draws=[]
        rng=np.random.default_rng(2026102151+(cohort=='CONFIRM_natural'))
        for first in range(0,5000,32):
            ix=rng.integers(0,n,size=(min(32,5000-first),n));draws.append(values[ix].mean(1))
        draws=np.concatenate(draws);reports[cohort]={}
        for i,name in enumerate(columns):
            v=dict(mean=float(values[:,i].mean()),CI95=np.quantile(draws[:,i],[.025,.975]).tolist())
            expected=r['reports'][cohort][name]
            metadata=expected['resampling']
            if (metadata['resampling_unit']!='source document'
                    or metadata['independent_clusters']!=n or metadata['items']!=n
                    or metadata['bootstrap_draws']!=5000
                    or metadata['bootstrap_seed']!=2026102151+(cohort=='CONFIRM_natural')):
                raise ValueError('Declared source-paired bootstrap contract')
            close([v['mean'],*v['CI95']],[expected['mean'],*expected['CI95']],name+' paired interval')
            reports[cohort][name]=v
    decisions={}
    for order in orders:
        key=f'{order}:minus-';c=reports['CONFIRM_calculation'];n=reports['CONFIRM_natural']
        decisions[str(order)]=dict(calculation_base=c[key+'B0']['mean']>=.25 and c[key+'B0']['CI95'][0]>0,
            calculation_all_controls=all(c[key+b]['CI95'][0]>0 for b in baselines[1:]),
            natural_gain=n[key+'B0']['mean']>=.01 and n[key+'B0']['CI95'][0]>0,
            natural_all_controls=all(n[key+b]['CI95'][0]>0 for b in baselines[1:]),
            natural_old_retention_guard=-n[key+'B0']['CI95'][0]/16<=.03)
    if decisions!=r['decisions']:raise ValueError('Per-order decisions')
    calc=all(v['calculation_base'] and v['calculation_all_controls'] for v in decisions.values())
    natural=all(v['natural_gain'] and v['natural_all_controls'] for v in decisions.values())
    if calc!=r['calculation_confirmation_all_six'] or natural!=r['natural_prediction_all_six']:
        raise ValueError('Joint endpoint decisions')
    result=dict(status='PASS_PUBLIC_FRESH_READER_NUMERICAL_REPLAY',files=len(manifest['files']),documents=5120,
        calculation_confirmation_all_six=calc,natural_prediction_all_six=natural,contrasts=80,
        limits='Projected scores, routing and statistics, not omitted source/model execution or external replication.')
    with a.output.open('x') as stream:stream.write(json.dumps(result,indent=2)+'\n')
    print(result['status'],flush=True)


if __name__=='__main__':main()

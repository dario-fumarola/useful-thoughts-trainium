"""Evaluate prospectively frozen selectors on fresh source panels; never fit."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import torch

from train.causal_reader_gate import SelectiveHead
from train.causal_reader_head_driver import cache, inputs
from train.causal_reader_fit import constant_logps
from train.causal_reader_gate_driver import require_frozen, science_budget
from weekend_thinking.artifacts import atomic_json, sha256_file, utc_now
from weekend_thinking.stats import cluster_bootstrap

ORDERS=tuple(range(2026102141,2026102147))
BASELINES=('B0','T_note:global','T_note:position','T_raw:global','T_raw:position','matched_raw')


def read(path): return json.loads(Path(path).read_text())


def frozen_heads(p):
    states={}
    binding=p['fitted_heads']
    if sha256_file(binding['file'])!=binding['sha256']: raise ValueError('Original head freeze changed')
    fitted=read(binding['file'])
    if not fitted['all_frozen_before_internal_validation']: raise ValueError('Frozen terminal heads required')
    for role in ('T_note','T_raw'):
        for order in ORDERS:
            name=f'{role}-hard-order{order}';spec=fitted['heads'][name]
            if spec!=p['heads'][name] or sha256_file(spec['file'])!=spec['sha256']:
                raise ValueError('A frozen head changed')
            states[name]=torch.load(spec['file'],weights_only=True)
    return states,fitted['controls']


def contrast_columns(totals):
    columns={}
    for order in ORDERS:
        name=f'T_note-hard-order{order}'
        for baseline in BASELINES:
            other=f'T_raw-hard-order{order}' if baseline=='matched_raw' else baseline
            columns[f'{order}:minus-{baseline}']=totals[name]-totals[other]
    for role in ('T_note','T_raw'):
        for mode in ('global','position'):
            columns[f'{role}:{mode}:minus-B0']=totals[role+':'+mode]-totals['B0']
    return columns


def verdicts(reports):
    output={}
    c=reports['CONFIRM_calculation'];n=reports['CONFIRM_natural']
    for order in ORDERS:
        key=f'{order}:minus-';base=c[key+'B0'];natural=n[key+'B0']
        output[str(order)]=dict(
            calculation_base=base['mean']>=.25 and base['CI95'][0]>0,
            calculation_all_controls=all(c[key+b]['CI95'][0]>0 for b in BASELINES[1:]),
            natural_gain=natural['mean']>=.01 and natural['CI95'][0]>0,
            natural_all_controls=all(n[key+b]['CI95'][0]>0 for b in BASELINES[1:]),
            natural_old_retention_guard=-natural['CI95'][0]/16<=.03)
    return output


def run(a):
    torch.set_num_threads(4);p=read(a.protocol);require_frozen(p)
    if p.get('study_schema')!='frozen-reader-confirmation-v1': raise ValueError('Fresh confirmation contract')
    runtime=read(a.runtime)
    if (runtime['status']!='ALL_FROZEN_HEAD_CONFIRMATION_FEATURE_PROCESSES_CLOSED'
            or runtime['protocol_sha256']!=sha256_file(a.protocol)
            or runtime['total_reserved_chip_hours_including_prior_qualification']>science_budget(p)):
        raise ValueError('All original native work must close within its fixed budget')
    deadline=datetime.fromisoformat(a.deadline_utc)
    def check():
        if datetime.now(timezone.utc)>=deadline: raise TimeoutError('Finite CPU readout')
    states,controls=frozen_heads(p);a.output.mkdir(exist_ok=False)
    reports={};absolute={};routing={};accounting={}
    for cohort in p['science_cohorts']:
        check();raw=cache(a.features,p,sha256_file(a.protocol),cohort,'B0_raw')
        totals={'B0':raw['lp'].sum(-1).numpy()};routes={};targets={}
        for role in ('T_note','T_raw'):
            other=cache(a.features,p,sha256_file(a.protocol),cohort,role);x=inputs(raw,other,p,role)
            totals[role]=other['lp'].sum(-1).numpy()
            c=controls[role]
            totals[role+':global']=constant_logps(raw['lp'],other['lp'],c['global_mixture']['weight']).sum(-1).numpy()
            totals[role+':position']=torch.stack([constant_logps(raw['lp'][:,i],other['lp'][:,i],w)
                for i,w in enumerate(c['position_schedule']['weights'])],1).sum(-1).numpy()
            for order in ORDERS:
                check();name=f'{role}-hard-order{order}';head=SelectiveHead(x.shape[-1]);head.load_state_dict(states[name])
                with torch.no_grad():
                    u=head(x);lp=head.target_logps(u,raw['lp'],other['lp']);g=head.gate(u)
                totals[name]=lp.sum(-1).numpy();routes[name]=g.numpy();targets[name]=lp.numpy()
                routing.setdefault(cohort,{})[name]=dict(mean_gate=g.mean().item(),
                    closed_fraction=(g==0).double().mean().item(),
                    near_closed_fraction=(g<=.001).double().mean().item())
            if role=='T_note':
                accounting[cohort]=dict(documents=len(other['index']),
                    generated_actions=sum(r['note_actions'] for r in other['index']),
                    capped_notes=sum(r['capped'] for r in other['index']))
            del other,x
        columns=contrast_columns(totals)
        ids=[r['document_id'] for r in raw['index']]
        if len(set(ids))!=p['data'][cohort]['rows']: raise ValueError('Complete unique source documents')
        stats=cluster_bootstrap(np.column_stack(list(columns.values())),ids,unit='source document',
            seed=2026102151+(cohort=='CONFIRM_natural'))
        reports[cohort]={name:dict(mean=float(stats['mean'][i]),CI95=stats['CI95'][i].tolist(),
            resampling=stats['metadata']) for i,name in enumerate(columns)}
        absolute[cohort]={name:float(v.mean()) for name,v in totals.items()}
        np.savez_compressed(a.output/(cohort+'-document-scores.npz'),**totals)
        np.savez_compressed(a.output/(cohort+'-gates.npz'),**routes)
        np.savez_compressed(a.output/(cohort+'-targets.npz'),**targets)
    decisions=verdicts(reports)
    atomic_json(a.output/'readout.json',dict(status='COMPLETE_FROZEN_READER_FRESH_CONFIRMATION',
        protocol_sha256=sha256_file(a.protocol),fitted_heads_sha256=p['fitted_heads']['sha256'],
        reports=reports,absolute=absolute,routing=routing,decisions=decisions,
        calculation_confirmation_all_six=all(v['calculation_base'] and v['calculation_all_controls'] for v in decisions.values()),
        natural_prediction_all_six=all(v['natural_gain'] and v['natural_all_controls'] for v in decisions.values()),
        generation_accounting=accounting,runtime_sha256=sha256_file(a.runtime),
        reserved_chip_hours=runtime['total_reserved_chip_hours_including_prior_qualification'],
        completed_utc=utc_now(),
        limits='Six frozen heads over one model pair, no fitting or order selection. Fresh source/position sampling, not proven novelty relative to pretraining. Teacher-forced full16 likelihood, unadjusted source-paired intervals. Calculation and unenriched natural endpoints are separate; tolerable regression is not a net predictive gain.'))
    print('COMPLETE_FROZEN_READER_FRESH_CONFIRMATION',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('protocol','features','runtime','output'): parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--deadline-utc',required=True);run(parser.parse_args())


if __name__=='__main__': main()

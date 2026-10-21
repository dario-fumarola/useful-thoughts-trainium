"""Check actual native frozen-reader features against independent CPU evidence.

Consumes completed qualification artifacts, never loads backbone weights or
fits a head. CPU fixtures cannot satisfy the native execution-device checks.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from train.causal_reader_gate import SelectiveHead
from weekend_thinking.artifacts import atomic_json,sha256_file


def read(path):return json.loads(Path(path).read_text())


def artifact(path,*,protocol_sha,role,command,device,hidden_size,vocabulary_size):
    complete=read(path/'complete.json')
    if (complete['status']!='COMPLETE_FROZEN_READER_FEATURES'
            or complete['protocol_sha256']!=protocol_sha or complete['role']!=role
            or complete['command']!=command or not complete['execution_device'].startswith(device)):
        raise ValueError('Actual native/reference identity mismatch')
    for name,digest in complete['files'].items():
        if sha256_file(path/name)!=digest:raise ValueError('Qualification artifact changed')
    index=[json.loads(line) for line in (path/'index.jsonl').open()]
    values={name:np.load(path/(name+'.npy'),allow_pickle=False)
            for name in ('hidden','entropy','target_logps','first_logps')}
    if len(index)!=complete['rows']:
        raise ValueError('Incomplete native/reference source index')
    expected={'hidden':(len(index),16,hidden_size),'entropy':(len(index),16),
              'target_logps':(len(index),16),'first_logps':(len(index),vocabulary_size)}
    for name,array in values.items():
        if (array.dtype!=np.float32 or not np.isfinite(array).all() or array.shape!=expected[name]
                or list(array.shape)!=complete['shapes'][name]):
            raise ValueError('Incomplete/nonfinite qualified arrays')
    return complete,index,values


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('protocol','references','native','causality','output'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--approved-exceptions',type=Path)
    a=p.parse_args();protocol=read(a.protocol);digest=sha256_file(a.protocol)
    if protocol['status']!='FROZEN_CAUSAL_READER_PROTOCOL':raise ValueError('Exact protocol required')
    limits=protocol['tolerances'];reports={};collected={};deviations=[];approved={}
    if a.approved_exceptions:
        approval=read(a.approved_exceptions)
        if (approval['status']!='OWNER_APPROVED_QUALIFICATION_AND_COST_DEVIATIONS'
                or approval['qualification_protocol_sha256']!=digest
                or not approval['before_any_selector_fit_or_scientific_DEV_outcome']):
            raise ValueError('Explicit owner approval of original qualification required')
        approved={(v['role'],v['field']):v for v in approval['accepted_numerical_deviations']}
    def compare(role,field,error,limit):
        if error<=limit:return
        exception=approved.get((role,field))
        if (exception is None or exception['limit']!=limit
                or abs(exception['observed']-error)>1e-10):
            raise ValueError(f'Native {role} {field} error exceeds frozen tolerance: {error}')
        deviations.append(dict(role=role,field=field,observed=error,limit=limit))
    for role in ('B0_raw','T_raw','T_note'):
        identity=dict(protocol_sha=digest,role=role,hidden_size=protocol['hidden_size'],
                      vocabulary_size=protocol['vocabulary_size'])
        ref_meta,reference,expected=artifact(a.references/role,command='reference',device='cpu',**identity)
        if len(reference)!=32:raise ValueError('All32 qualification sources required')
        reference_by_index={r['global_index']:i for i,r in enumerate(reference)}
        if set(reference_by_index)!=set(range(32)):raise ValueError('Unique complete qualification IDs required')
        fields={name:np.empty_like(value) for name,value in expected.items()}
        seen=set();errors={name:0. for name in expected}
        for part in range(protocol['partitions']['qualification']['features']):
            meta,index,values=artifact(a.native/role/f'part-{part}',command='features',device='xla',**identity)
            if meta['signature']!=ref_meta['signature']:raise ValueError('Loaded model differs from CPU reference')
            for local,row in enumerate(index):
                target=reference_by_index[row['global_index']]
                if row!=reference[target] or target in seen:
                    raise ValueError('Native source/context/target identity mismatch')
                seen.add(target)
                for name in fields:
                    fields[name][target]=values[name][local]
                    errors[name]=max(errors[name],float(np.abs(values[name][local].astype(np.float64)
                                                             -expected[name][target]).max()))
        if len(seen)!=32:raise ValueError('Missing native qualification source')
        for name,error in errors.items():
            compare(role,name,error,limits[name+'_max_absolute'])
        relative=np.linalg.norm(fields['hidden'].astype(np.float64)-expected['hidden'],axis=-1)/np.maximum(
            np.linalg.norm(expected['hidden'].astype(np.float64),axis=-1),1e-12)
        compare(role,'hidden_relative_L2',float(relative.max()),limits['hidden_max_relative_L2'])
        mass=np.abs(np.log(np.exp(fields['first_logps'].astype(np.float64)).sum(-1)))
        if mass.max()>limits['normalization_log_error']:
            raise ValueError('Native full-vocabulary distribution is not normalized')
        cause=read(a.causality/role/'complete.json')
        if (cause['status']!='COMPLETE_ACTUAL_NATIVE_CAUSALITY_CHECK' or cause['protocol_sha256']!=digest
                or cause['role']!=role or not cause['execution_device'].startswith('xla')
                or cause['signature']!=ref_meta['signature']
                or len(cause['checks'])!=6):
            raise ValueError('Missing actual-model causal perturbation evidence')
        if any(max(r['hidden_error'],r['entropy_error'])>limits['causal_perturbation_max_absolute']
               for r in cause['checks']):raise ValueError('Future/current information reaches gate inputs')
        checks=ref_meta['greedy_CPU_checks']
        if role=='T_note' and (len(checks)!=32 or any(r['maximum_behavior_error']>.001
                or r['maximum_argmax_logit_regret']>.001 for r in checks)):
            raise ValueError('All native greedy sequences require CPU conditional verification')
        reports[role]=dict(documents=32,maximum_absolute_errors=errors,
            hidden_max_relative_L2=float(relative.max()),normalization_log_error=float(mass.max()),
            causal_checks=6,greedy_CPU_checks=len(checks),
            CPU_numerical_near_tie_actions=sum(r['numerical_near_tie_actions'] for r in checks))
        collected[role]=fields
    raw=torch.from_numpy(collected['B0_raw']['first_logps'])
    assisted=torch.from_numpy(collected['T_note']['first_logps'])
    head=SelectiveHead(protocol['hidden_size']*2+5)
    if head.predict_logps(None,raw,enabled=False) is not raw:
        raise ValueError('Disabled path changes original B0 prediction object')
    normalization=[]
    for u in (-1000.,math.log(.0199/.9801),0.,1000.):
        scores=torch.full((32,),u,dtype=torch.float64)
        mixed=head.predict_logps(scores,raw,assisted)
        if u==-1000 and mixed is not raw:raise ValueError('Closed path is not exact B0')
        error=torch.logsumexp(mixed.double(),-1).abs().max().item()
        if error>limits['normalization_log_error']:raise ValueError('Actual-vector mixture is not normalized')
        normalization.append(dict(gate_logit=u,maximum_log_mass_error=error))
    if approved and {(v['role'],v['field']) for v in deviations}!=set(approved):
        raise ValueError('Owner exception does not exactly match original saved deviations')
    status='OWNER_ACCEPTED_QUALIFICATION_WITH_DECLARED_DEVIATIONS' if approved else 'PASS_ACTUAL_READER_FEATURE_QUALIFICATION'
    atomic_json(a.output,dict(status=status,
        original_requirements_pass=not bool(deviations),accepted_deviations=deviations,
        approval_sha256=sha256_file(a.approved_exceptions) if a.approved_exceptions else None,
        protocol_sha256=digest,roles=reports,actual_native_sources_per_role=32,
        full_vector_mixture_checks=normalization,disabled_and_closed_exact_raw_identity=True,
        source_sha256=sha256_file(__file__),limits='Actual-model numerical/causality mechanics only; no trained selector, scientific efficacy, broader robustness or fresh-DEV outcome.'))
    print(status,json.dumps(reports),flush=True)


if __name__=='__main__':main()

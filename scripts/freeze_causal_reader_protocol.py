"""Freeze one conditional qualification protocol after complete CPU feasibility.

This never constructs a model or releases scientific execution. It requires a
separate recorded whole-question decision and independently checked cohorts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from train.causal_reader_gate_driver import sources
from train.evaluate import asset_provenance
from weekend_thinking.artifacts import atomic_json,sha256_file,utc_now

ROOT=Path(__file__).resolve().parents[1]


def read(path):return json.loads(Path(path).read_text())


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('data','audit','decision','output'):parser.add_argument('--'+name,type=Path,required=True)
    a=parser.parse_args();audit=read(a.audit);decision=read(a.decision);complete=read(a.data/'complete.json')
    if (audit['status']!='PASS_ALL_SELECTED_SOURCE_NATIVE_EXCLUSION_AND_EXACT_EQUATION_CHECKS'
            or audit['documents']!=13344 or audit['exact_expressions']!=6672
            or audit['preparation_sha256']!=sha256_file(a.data/'complete.json')
            or audit['source_sha256']!=sha256_file(ROOT/'scripts/check_causal_reader_data.py')
            or complete['status']!='CPU_SOURCE_FEASIBILITY_COMPLETE_NOT_MODEL_RELEASE'):
        raise ValueError('Independent complete unchanged source/native audit required')
    if (decision['status']!='PASS_CPU_FEASIBILITY_RELEASE_ONLY_BOUNDED_NATIVE_QUALIFICATION'
            or decision['data_audit_sha256']!=sha256_file(a.audit)
            or decision['qualification_reserved_chip_hour_ceiling']!=.5
            or decision['whole_experiment_reserved_chip_hour_ceiling']!=8
            or decision['scientific_execution_released']):
        raise ValueError('Finite reviewed qualification-only decision required')
    binding_file=ROOT/'local/diagnostics/real-cpt-scaffold-v29-evaluation-v1/bindings/computation-seed2026102029.json'
    binding=read(binding_file)
    if (binding['status']!='AUDITED_V29_TERMINAL1280' or binding['step']!=1280
            or binding['checkpoint_sha256']!='bacd0f2522d5998c4b355c0f15209fcb36b5b0ddd361606e8fef89cefea714d4'
            or sha256_file(binding['checkpoint'])!=binding['checkpoint_sha256']
            or sha256_file(binding['audit_file'])!=binding['audit_sha256']):
        raise ValueError('Original first V29 frozen computation checkpoint required')
    model=ROOT/'local/assets/qwen3-1.7b-base';assets=asset_provenance(model)
    config=read(model/'config.json');data={}
    for name,count in [('TRAIN',5120),('DEV_calculation',4096),('DEV_ordinary',4096),('qualification',32)]:
        path=a.data/(name+'.jsonl');questions=a.data/(name+'-questions.jsonl')
        entry=complete['cohorts'][name]
        if (entry['rows']!=count or sha256_file(path)!=entry['sha256']
                or sha256_file(questions)!=entry['questions_sha256']):
            raise ValueError('Original frozen cohort counts/bytes required')
        data[name]=dict(rows=count,labels_file=str(path.resolve()),labels_sha256=entry['sha256'],
                       questions_file=str(questions.resolve()),questions_sha256=entry['questions_sha256'])
    questions=[json.loads(line) for line in Path(data['DEV_calculation']['questions_file']).open()]
    chosen=[]
    for part in range(4):
        stripe=[q['item_id'] for i,q in enumerate(questions) if i%4==part]
        chosen+=sorted(stripe,key=lambda item:hashlib.sha256(('causal-reader-secondary:'+item).encode()).hexdigest())[:64]
    a.output.parent.mkdir(parents=True,exist_ok=True)
    if a.output.exists():raise FileExistsError('No overwrite of frozen protocol')
    structured=ROOT/'validation/causal_reader_gate_design_20261021.json'
    cost=ROOT/'validation/causal_reader_cost_plan_20261021.json'
    atomic_json(a.output,dict(status='FROZEN_CAUSAL_READER_PROTOCOL',identity='causal-reader-gate-20261021-v1',
        created_utc=utc_now(),sources=sources(),design_file=str(structured),design_sha256=sha256_file(structured),
        cost_plan_file=str(cost),cost_plan_sha256=sha256_file(cost),data=data,
        data_audit_file=str(a.audit.resolve()),data_audit_sha256=sha256_file(a.audit),
        decision_file=str(a.decision.resolve()),decision_sha256=sha256_file(a.decision),
        model_dir=str(model),assets=assets,models=dict(B0=dict(id='B0',checkpoint=None),T=binding),
        binding_file=str(binding_file),binding_sha256=sha256_file(binding_file),
        hidden_size=config['hidden_size'],vocabulary_size=config['vocab_size'],
        qualification_release=dict(status='RELEASED_BOUNDED_ACTUAL_MODEL_QUALIFICATION_ONLY',
            source_documents=32,maximum_reserved_chip_hours=.5,physical_device=8,logical_cores=[32,33,34,35]),
        science_release_file=str((a.output.parent/'science-release.json').resolve()),
        partitions=dict(qualification=dict(generate=4,features=4,reference=1,causality=1),
                        science=dict(generate=8,features=8,interventions=4)),
        secondary_interventions=dict(item_ids=chosen,selection='First64 fixed SHA-priority IDs per source-index stripe, before model calls; four contexts/source; original generated length/cap metadata held.'),
        tolerances=dict(hidden_max_absolute=.001,hidden_max_relative_L2=.0001,
            entropy_max_absolute=.001,target_logps_max_absolute=.001,first_logps_max_absolute=.001,
            greedy_CPU_max_argmax_logit_regret=.001,greedy_CPU_behavior_logp_error=.001,
            normalization_log_error=.00002,causal_perturbation_max_absolute=.000001),
        limits='Qualification only. No science/experiment-head run until actual native parity/causality/normalization, finite measured-cost profile and a separate whole-question release. No writer RL, new CPT, model-size change or official TEST.'))
    print('FROZEN_QUALIFICATION_ONLY_NO_MODEL_CALLS',sha256_file(a.output),flush=True)


if __name__=='__main__':main()

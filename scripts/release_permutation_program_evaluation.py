"""Bind all six independently audited V28 terminal models before held inference.

This command records a release; it does not launch or score any model call.
"""
import argparse
import json
from pathlib import Path

from train.permutation_program_science_evaluate import MODEL_IDS,MANIFEST_SHA,DESIGN_SHA,sources
from weekend_thinking.artifacts import atomic_json,sha256_file,utc_now


def release(control,output):
    import torch
    control,output=Path(control).resolve(),Path(output).resolve()
    plan=json.loads((control/'plan.json').read_text())
    if (plan['prospective_design_sha256']!=DESIGN_SHA
            or set(plan['jobs'])!=MODEL_IDS
            or sha256_file(plan['prospective_design_file'])!=DESIGN_SHA):
        raise ValueError('The unchanged six-model prospective design is required')
    models={'B0':dict(id='B0',checkpoint=None)}
    closure_files={}
    for key in sorted(MODEL_IDS):
        job=plan['jobs'][key];root=Path(job['output']);phase=control/key/'terminal'
        audit_file=root/'terminal-full-audit.json'
        audit=json.loads(audit_file.read_text());audit_digest=sha256_file(audit_file)
        if (audit.get('status')!='PASS_V28_TERMINAL_ALL_STATE' or audit.get('step')!=4096
                or audit.get('model')!=key or audit.get('source_files')!=plan['source_files']
                or audit.get('protocol_sha256')!=job['protocol_sha256']
                or sha256_file(job['protocol'])!=job['protocol_sha256']):
            raise ValueError('Each actual full terminal state must pass the independent audit')
        for name in ('exit.json','closure.json','independent-all-state-audit.json'):
            path=phase/name;value=json.loads(path.read_text());closure_files[str(path)]=sha256_file(path)
            if name=='exit.json' and value['exit_code']!=0:raise ValueError('Original worker group must close successfully')
            if name=='closure.json' and not(value['all_original_session_processes_absent'] and value['assigned_physical_chip_empty']):
                raise ValueError('Physical process closure is required before held evaluation')
            if name=='independent-all-state-audit.json' and sha256_file(path)!=audit_digest:
                raise ValueError('Published terminal audit and run audit disagree')
        checkpoint=root/'step-0004096/model.pt'
        digest=sha256_file(checkpoint)
        if digest!=audit['model_sha256']:raise ValueError('Audited terminal model bytes changed')
        # Read lineage without allocating another complete model. An older seed's
        # optimizer may have rotated only after this independent audit; all six
        # terminal model files remain required.
        weights=torch.load(checkpoint,weights_only=True,map_location='cpu',mmap=True)
        if weights['step']!=4096 or weights['run_id']!=key:raise ValueError('Terminal lineage mismatch')
        models[key]=dict(id=key,run_id=key,step=4096,spec_hash=weights['spec_hash'],
            status='AUDITED_V28_TERMINAL4096',arm=job['arm'],seed=job['seed'],
            checkpoint=str(checkpoint),checkpoint_sha256=digest,
            audit_file=str(audit_file),audit_sha256=audit_digest)
        del weights
    record=dict(status='RELEASED_V28_FIXED_ALL_TERMINAL_EVALUATION',created_utc=utc_now(),
        experiment_id='controlled-computation-v28',model_specs=models,
        data_manifest_sha256=MANIFEST_SHA,prospective_design_sha256=DESIGN_SHA,
        evaluation_source_files=sources(),terminal_closure_files=closure_files,
        release_source_sha256=sha256_file(__file__),maximum_evaluation_reserved_chip_hours=18,
        decoding=dict(compute='FP32',shards=4,prefill_width=256,max_actions=128,greedy=True),
        scope='All fixed terminal modes/cohorts only; no prompt, model or outcome selection.',
        limits=['Release construction is not execution or learned-capability evidence.',
            'All caps, errors and unsupported outputs remain in fixed paired denominators.',
            'Intermediate state targets and answer-training contexts differ across complete recipes.',
            'This supervised mechanism study does not establish autonomous discovery or general pretraining improvement.'])
    atomic_json(output,record)
    return record


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study-control',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();record=release(args.study_control,args.output)
    print(json.dumps(dict(status=record['status'],models=len(record['model_specs']))))


if __name__=='__main__':main()

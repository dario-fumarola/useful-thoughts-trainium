"""Account actual owned reserved chip-time, including startup and closure."""
from pathlib import Path
from datetime import datetime
import json
R=Path('/home/ubuntu/dario');C=R/'local/diagnostics/document-evidence-weight-v25-ablation-v1'
read=lambda p:json.loads(p.read_text())
def interval(directory,chips):
    launch=read(directory/'launch.json');closed=read(directory/'closure.json')
    seconds=(datetime.fromisoformat(closed['utc'])-datetime.fromisoformat(launch['utc'])).total_seconds()
    assert seconds>0
    return dict(physical_chips=chips,launch_utc=launch['utc'],confirmed_closure_utc=closed['utc'],reserved_chip_hours=chips*seconds/3600,
        interpretation='Conservative ownership interval through confirmed physical closure; includes initialization and post-process checks.')
training=interval(C/'evidence_select-weight1-seed2026101923',2)
assert training['reserved_chip_hours']<=3
CQ=R/'local/diagnostics/document-evidence-weight-v25-world8-qualification-v1'
qualification={key:interval(CQ/key,2) for key in ('select-fresh4','select-resume8')}
assert sum(v['reserved_chip_hours'] for v in qualification.values())<=3
allocation=read(R/'validation/document_evidence_weight_v25_readout_execution_20261019.json')['stage_maximum_reserved_chip_hours']
phases={'TRAIN256_generation':'TRAIN-fit/GEN','held255_native_generation':'HELD-native/GEN','held255_closed_field_answer_replay':'HELD-replay/GEN','ordinary256_likelihood':'ORDINARY/GEN'}
evaluation={}
for name,path in phases.items():
    directory=C/path;plan=read(directory/'plan.json');closure=read(directory/'closure.json')
    seconds=(datetime.fromisoformat(closure['utc'])-datetime.fromisoformat(plan['utc'])).total_seconds()
    assert 0<seconds<=allocation[name]*3600
    evaluation[name]=dict(physical_chips=1,start_utc=plan['utc'],confirmed_closure_utc=closure['utc'],reserved_chip_hours=seconds/3600,
        maximum_allocated_chip_hours=allocation[name])
q=sum(v['reserved_chip_hours'] for v in qualification.values());e=sum(v['reserved_chip_hours'] for v in evaluation.values())
assert e<=.6
result=dict(status='COMPLETE_ACTUAL_RESERVED_CHIP_TIME_ALL_STAGES',training=training,qualification_separate=qualification,evaluation=evaluation,
    qualification_reserved_chip_hours=q,science_training_and_evaluation_reserved_chip_hours=training['reserved_chip_hours']+e,
    total_new_work_reserved_chip_hours=q+training['reserved_chip_hours']+e,
    controls='Completed V23 control reused; its original historical cost remains in its report and is not charged as new work.',
    execution_error_accounting='Check readout reports separately; completed ownership intervals do not establish scientific success.',
    accounting='Reserved physical-chip time, not dollar cost or theoretical FLOPs. CPU-only preparation/audits and backup IO off-device are not accelerator execution; conservative closure intervals may include some such overhead.')
with (C/'actual-cost.json').open('x') as f:json.dump(result,f,indent=2)
print(json.dumps(result))

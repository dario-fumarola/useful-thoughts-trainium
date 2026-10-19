"""Descriptive resource accounting after the frozen result is complete.

No gate changes,matched-serving claim,FLOP estimate or historical-project total.
"""
from pathlib import Path
from datetime import datetime,timezone
import csv,gzip,hashlib,json
import numpy as np
R=Path('/home/ubuntu/dario');C=R/'local/diagnostics/document-QA-v21-study-v1';F=R/'validation/document_QA_v21_complete_20261019.json';P=F.with_suffix('')
read=lambda p:json.loads(Path(p).read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();epoch=lambda s:datetime.fromisoformat(s).timestamp()
r=read(F);assert read(C/'final-readout-completion.json')['status']=='ALL_SEVEN_MODELS_COMPLETE_ANALYZED_INDEPENDENTLY_CHECKED_RENDERED_AND_PROTECTED';assert read(P/'independent-readout-check.json')['report_sha256']==sha(F)
training=[]
for j in read(C/'plan.json')['jobs']:
 D=C/j['key'];O=Path(j['output']);p=read(j['protocol']);completion=read(O/'completion.json');start=read(D/'launch.json');live=read(D/'first-live.json');close=read(D/'closure.json')
 assert completion['actual_steps']==256 and close['assigned_devices_empty']
 reserved=epoch(close['utc'])-epoch(start['utc']);observed=epoch(close['utc'])-epoch(live['utc']);assert reserved>=observed>0
 training.append(dict(model=j['key'],unique_questions=2048,question_occurrences=8192,updates=256,source_target_labels=2097152,task_target_labels={'qa_direct':42832,'qa_prior':251440,'qa_post':259694}[j['arm']],physical_chips=2,logical_cores=8,reserved_seconds=reserved,reserved_physical_chip_hours=reserved*2/3600,first_confirmed_all8_to_physical_closure_seconds=observed,loop_including_saves_seconds=completion['loop_including_saves_seconds'],checkpoint_walls=completion['checkpoint_walls'],closure_sha256=sha(D/'closure.json')))
groups=[]
generation=[dict(model='B0',control=str(C/'B0-generation')), *read(C/'trained-generation-registry.json')['groups']]
for kind,registry in [('generation',generation),('likelihood',read(C/'likelihood-registry.json')['groups'])]:
 for g in registry:
  D=Path(g['control']);start=min(epoch(read(D/f'shard-{s}/launch.json')['utc']) for s in range(4));live=epoch(read(D/'first-live.json')['utc']);close=read(D/'closure.json');end=epoch(close['utc']);assert end>=live>=start
  groups.append(dict(kind=kind,model=g['model'],group=g.get('label','B0-all'),control=str(D),physical_chips=1,logical_cores=4,reserved_seconds=end-start,reserved_physical_chip_hours=(end-start)/3600,first_confirmed_all4_to_physical_closure_seconds=end-live,closure_sha256=sha(D/'closure.json')))
cells=[]
for model,modes in r['tables'].items():
 values=[json.loads(l) for l in gzip.decompress((P/(model+'-all-generation-attempts.jsonl.gz')).read_bytes()).splitlines()]
 for mode,table in modes.items():
  vs=[v for v in values if v['mode']==mode];assert len(vs)==512
  actions=sum(v['n_actions'] for v in vs);walls=[v['wall_seconds'] for v in vs]
  cells.append(dict(model=model,mode=mode,questions=512,canonical_correct=table['correct'],total_generated_actions=actions,mean_actions_per_question=actions/512,actions_per_canonical_correct=actions/table['correct'] if table['correct'] else None,mean_prompt_tokens=sum(v['prompt_tokens'] for v in vs)/512,attempt_wall_sum_seconds=sum(walls),attempt_wall_median_seconds=float(np.median(walls)),attempt_wall_p90_seconds=float(np.quantile(walls,.9)),capped=table['capped'],execution_errors=table['execution_errors']))
out=dict(status='COMPLETE_DESCRIPTIVE_TRAINING_GENERATION_AND_MEASUREMENT_COST_ACCOUNTING',utc=datetime.now(timezone.utc).isoformat(),primary_report_sha256=sha(F),source_sha256=sha(__file__),training=training,execution_groups=groups,generation_cells=cells,totals=dict(science_training_reserved_physical_chip_hours=sum(v['reserved_physical_chip_hours'] for v in training),generation_reserved_physical_chip_hours=sum(v['reserved_physical_chip_hours'] for v in groups if v['kind']=='generation'),likelihood_measurement_reserved_physical_chip_hours=sum(v['reserved_physical_chip_hours'] for v in groups if v['kind']=='likelihood'),generated_actions=sum(v['total_generated_actions'] for v in cells)),limits=['Reservedtime=ownedlaunch to confirmedphysicalclosure;includes loading,nativechecks,coldcompilation,and shutdown. Sampling timestamps,not perfect activehardwareprofiling.','Perattemptwall includes actualgreedyprefill/decode under concurrentjobs;notcontrolledservinglatency or equalFLOPs.','Prior/empty/replay/oracle differ in suppliedcontextandgeneratedactions;oracle privileged.','Training uses sameupdates/shapes/sourceorder butdifferent tasklabelinventories.','Likelihood is evaluationoverhead,not deployedpolicywork. Thisaccount excludes preceding qualifications,data preparation andallhistorical experiments;not a totalprojectcost.'])
(P/'resource-accounting.json').write_text(json.dumps(out,indent=2))
with (P/'all-generation-token-latency-costs.csv').open('x',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(cells[0]));w.writeheader();w.writerows(cells)
print(json.dumps(dict(status=out['status'],totals=out['totals']),indent=2),flush=True)

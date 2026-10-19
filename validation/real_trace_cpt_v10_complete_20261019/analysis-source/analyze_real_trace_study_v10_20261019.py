"""Analyze the closed six-model real study, keeping both seeds/panels separate."""
from pathlib import Path
from datetime import datetime,timezone
from collections import Counter
import json,gzip,math
import numpy as np
from train.real_trace_results import contrast,decisions
from weekend_thinking.artifacts import sha256_file
R=Path('/home/ubuntu/dario');C=R/'local/diagnostics/real-trace-cpt-v10-study-v1'
registry=json.loads((C/'evaluation-registry.json').read_text())
assert registry['status']=='ALL_SIX_FROZEN_TERMINAL_EVALUATIONS_LAUNCHED'
def read(p):return json.loads(Path(p).read_text())
def rows(p):return [json.loads(l) for l in Path(p).open()]
controls={'B0':Path(registry['B0_control'])}
controls.update({key:Path(v['control']) for key,v in registry['models'].items()})
for key,d in controls.items():assert read(d/'CPU-scoring-completion.json')['status']=='COMPLETE_THREE_FIXED_PANELS_ALL_SCORES'
PUBLIC=R/'validation/real_trace_cpt_v10_complete_20261019';PUBLIC.mkdir(exist_ok=False)
cells={};absolute={};source_receipts={};payloads={}
for key,d in controls.items():
 completion=read(d/'CPU-scoring-completion.json');config=read(d/'evaluation-binding.json')
 root=Path(config['CPU_score_output_root']);cells[key]={};absolute[key]={}
 for panel in ('DEV_AST256','DEV_ordinary256','DEV_legacy256'):
  source=Path(completion['ordinary_source']) if panel=='DEV_ordinary256' else root/panel
  run=read(source/'run.json');assert run['status']=='COMPLETE_ALL_FIXED_DOCUMENT_LIKELIHOODS'
  assert run['documents']==256 and run['panel']==panel and run['model_id']==config['model_id']
  assert sha256_file(source/'logps.jsonl')==run['logps_sha256']
  original=rows(source/'logps.jsonl');modes=run['modes'];assert len(original)==256*len(modes)
  expected={'raw'} if panel=='DEV_ordinary256' else {'raw','empty','oracle','own','other'}
  assert set(modes)==expected
  bymode={m:[v for v in original if v['mode']==m] for m in modes}
  assert all(len(vs)==256 and len({v['item_id'] for v in vs})==256 and len({v['document_id'] for v in vs})==256 for vs in bymode.values())
  cells[key][panel]=bymode;absolute[key][panel]={}
  for mode,vs in bymode.items():
   values=np.array([v['logps16'] for v in vs],dtype=np.float64);assert values.shape==(256,16) and np.isfinite(values).all()
   components=[v.get('component_logprobs') for v in vs if v.get('component_status')=='exact_native_partition']
   absolute[key][panel][mode]=dict(documents=256,mean_span16_logprob=float(values.sum(1).mean()),mean_token_NLL=float(-values.mean()),
    generation_error_rows=sum(v['note_execution_status']!='ok' for v in vs),component_supported=len(components),
    component_descriptive_means=({name:sum(v[name] for v in components)/len(components) for name in components[0]} if components else None),
    component_uncertainty='No componentCI:primaryfull16usesall256;unsupportedcomponentboundariesareexplicit.')
  file=PUBLIC/f'{key}-{panel}-all{len(original)}-logps.jsonl.gz'
  file.write_bytes(gzip.compress((source/'logps.jsonl').read_bytes(),mtime=0))
  payloads[str(file.relative_to(R))]=dict(rows=len(original),sha256=sha256_file(file))
  source_receipts[f'{key}/{panel}']=dict(run_sha256=sha256_file(source/'run.json'),original_logps_sha256=run['logps_sha256'],model_signature=run['model_signature'],CPU_panel_wall_seconds=run['wall_seconds'],CPU_panel_scored_rows=len(original),reused_B0_ordinary_reference=(key=='B0' and panel=='DEV_ordinary256'))
def cmp(a,amode,b,bmode,panel='DEV_AST256',nll=False):return contrast(cells[a][panel][amode],cells[b][panel][bmode],token_nll=nll)
within_model={key:{panel:{name:cmp(key,a,key,b,panel=panel) for name,a,b in [('empty_minus_raw','raw','empty'),('oracle_minus_raw','raw','oracle'),('own_minus_raw','raw','own'),('own_minus_empty','empty','own'),('own_minus_other','other','own')]} for panel in ('DEV_AST256','DEV_legacy256')} for key in cells}
seeds={}
for seed in (2026101913,2026101914):
 raw=f'real_raw-seed{seed}';prior=f'real_prior-seed{seed}';post=f'real_post-seed{seed}'
 primary=dict(post_raw_minus_raw_raw=cmp(raw,'raw',post,'raw'),prior_own_minus_raw_raw=cmp(raw,'raw',prior,'own'),
  prior_own_minus_prior_raw=cmp(prior,'raw',prior,'own'),prior_own_minus_prior_empty=cmp(prior,'empty',prior,'own'),
  prior_own_minus_prior_other=cmp(prior,'other',prior,'own'),prior_own_minus_raw_own=cmp(raw,'own',prior,'own'),
  post_raw_minus_B0raw=cmp('B0','raw',post,'raw'),raw_raw_minus_B0raw=cmp('B0','raw',raw,'raw'),
  prior_own_minus_B0own=cmp('B0','own',prior,'own'),prior_own_minus_B0raw=cmp('B0','raw',prior,'own'))
 ordinary={arm:cmp(raw,'raw',model,'raw',panel='DEV_ordinary256',nll=True) for arm,model in [('prior',prior),('post',post)]}
 eligible=all(v['note_execution_status']=='ok' for m in ('own','other') for v in cells[prior]['DEV_AST256'][m])
 gate=decisions(primary['post_raw_minus_raw_raw'],ordinary['post'],primary['prior_own_minus_raw_raw'],
   primary['prior_own_minus_prior_raw'],primary['prior_own_minus_prior_empty'],inference_execution_complete=eligible)
 legacy=dict(post_raw_minus_raw_raw=cmp(raw,'raw',post,'raw',panel='DEV_legacy256'),
   prior_own_minus_raw_raw=cmp(raw,'raw',prior,'own',panel='DEV_legacy256'))
 seeds[str(seed)]=dict(primary_AST=primary,ordinary_token_NLL_regression=ordinary,legacy_secondary=legacy,declared_decisions=gate)
notes={}
for key,d in controls.items():
 config=read(d/'evaluation-binding.json');notes[key]={}
 for panel,path in config['notes'].items():
  bundle=read(Path(path)/'notes-bundle.json');assert bundle['model_id']==config['model_id'] and len(bundle['controls'])==256
  own=[c['own'] for c in bundle['controls']]
  for name,digest in bundle['source_pins'].items():assert sha256_file(name)==digest
  native=[read(name) for name in bundle['source_pins'] if Path(name).name=='native.json'];cached=[read(name) for name in bundle['source_pins'] if Path(name).name=='cache-prefill.json'];workers=[read(name) for name in bundle['source_pins'] if Path(name).name=='run.json'];assert len(native)==len(cached)==len(workers)==4
  notes[key][panel]=dict(documents=256,generated_actions=sum(v['n_actions'] for v in own),mean_actions=sum(v['n_actions'] for v in own)/256,
    capped=sum(v['stop_type']=='length_cap' for v in own),empty=sum(not v['content_ids'] for v in own),errors=sum(v['status']!='ok' for v in own),
    stop_types=dict(Counter(v['stop_type'] for v in own)),generation_wall_seconds_sum=sum(v['wall_seconds'] for v in own),
    fixed_reader_closing_boundary=dict(text='Answer:\n',native_tokens_per_document=2,total_for_panel=512),worker_wall_seconds_sum=sum(v['wall_seconds'] for v in workers),native_max_bidirectional_KL=max(v['max_bidirectional_kl_nats'] for v in native),native_max_logp_error=max(v['max_logp_error'] for v in native),native_all_argmax_equal=all(v['argmax_equal'] for v in native),cache_max_logp_error=max(v['max_logp_error'] for v in cached),limits='GenerationtimingisTrainiumFP32;likelihoodtimingisCPUFP32,notacontrolledend-to-endservicelatency.')
  # Public note outputs exclude source prompt IDs; complete originals stay archived.
  visible=[{k:v[k] for k in ('item_id','document_id','status','actions','content_ids','text','stop_type','n_actions','wall_seconds')} for v in own]
  file=PUBLIC/f'{key}-{panel}-all256-generated-notes.jsonl.gz'
  file.write_bytes(gzip.compress((''.join(json.dumps(v,separators=(',',':'))+'\n' for v in visible)).encode(),mtime=0))
  payloads[str(file.relative_to(R))]=dict(rows=256,sha256=sha256_file(file))
training_audit=read(C/'terminal-independent-audit.json')
assert training_audit['status']=='PASS_ALL_SIX_TERMINAL320_TRAINING_STATES_NATIVE48_ORDER_AND_COUNTERS'
training_costs=[{k:m[k] for k in ('arm','seed','actual_updates','terminal_counters','warm_last60_median_update_seconds')} for m in training_audit['models']]
report=dict(status='COMPLETE_TWO_SEED_REAL_DOCUMENT_STUDY_ALL_FIXED_PANELS',utc=datetime.now(timezone.utc).isoformat(),
 decisions=dict(training_benefit_pass_both_seeds=all(v['declared_decisions']['training_benefit_pass'] for v in seeds.values()),
    inference_benefit_pass_both_seeds=all(v['declared_decisions']['inference_benefit_pass'] for v in seeds.values())),
 seeds=seeds,within_model_contrasts=within_model,absolute_likelihood=absolute,training_costs=training_costs,loss_allocation=read(R/'validation/real_trace_cpt_v10_loss_allocation_20261019.json'),note_costs_and_failures=notes,payloads=payloads,source_receipts=source_receipts,
 limits=['SelectedFineMathDEVheldfromourfinetuning,notguaranteedpretrainingnoveltyorofficialTEST.','Each256documentpanelandsource/checkpointisseparate;20kpairedbootstrapwithseed2026101915conditiononcheckpoints,nottrainingseeduncertainty.','Ordinaryregressionreportedforall;predeclaredtraininggateappliespostguard. Equal-modepriorown/rawowncomparisonseparatesdataaugmentationfromaddinginferencecompute.','Likelihood primary; no free-answer accuracy or end-to-end service latency claim. All arms use 320 updates and 10,240 document occurrences, with different supervised-token totals and loss allocation; active token totals are not a FLOP measurement; padded training shapes match.','All128actioncapsandgenerationerrorsretained;anexecutionerrorprecludesinferencegatebutnotindependentrawtrainingbenefit.','Oracleincludesrootanswerandisprefix-computable,notdeployedgenerationevidence.','Neitherbeatingadegradedcontrolnorpositiveown-emptyaloneestablishesimprovementoverB0orcontent-sensitivecomputation.'])
with (R/'validation/real_trace_cpt_v10_complete_20261019.json').open('x') as f:json.dump(report,f,indent=2)
print(json.dumps(dict(decisions=report['decisions'],seeds=seeds),indent=2),flush=True)

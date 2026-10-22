"""Measured, bounded CPU replication of paired gains on a fixed inspected subset."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timezone
import json,math,os,subprocess,sys,time
import numpy as np
from weekend_thinking.artifacts import atomic_json,canonical_hash,sha256_file,utc_now
from weekend_thinking.stats import cluster_bootstrap
R=Path('/home/ubuntu/dario');C=Path(__file__).parent;S=R/'local/diagnostics/scaffold-learning-overnight-v1';E=R/'local/diagnostics/general-web-retention-20261022';plan=json.loads((C/'private-plan.json').read_text());D=Path(plan['output_directory']);stop=datetime.fromisoformat(plan['CPU_stop_UTC'])
while not ((S/'study-complete.json').exists() and (S/'standard-CPU-check/readout.json').exists() and (E/'complete-readout.json').exists()):
 if datetime.now(timezone.utc)>=stop:raise TimeoutError('No completed small CPU timing and parent readouts before large-check boundary')
 time.sleep(10)
small=json.loads((S/'standard-CPU-check/readout.json').read_text());per_forward=small['actual_wall_seconds']/small['actual_CPU_model_forwards'];seconds_left=(stop-datetime.now(timezone.utc)).total_seconds();count=next((n for n in plan['candidate_document_counts'] if per_forward*(4*n)*1.20+180<=seconds_left),0);projection=per_forward*(4*count)*1.20+180 if count else seconds_left+1;available=next(int(v.split()[1])*1024 for v in Path('/proc/meminfo').read_text().splitlines() if v.startswith('MemAvailable:'));go={'utc':utc_now(),'measured_small_check_seconds_per_forward_including_model_loads':small['actual_wall_seconds']/small['actual_CPU_model_forwards'],'projected_two_worker_wall_seconds':projection,'seconds_until_stop':(stop-datetime.now(timezone.utc)).total_seconds(),'available_host_GiB':available/2**30,'CPU_affinities':plan['CPU_affinities'],'selected_documents':count,'selection_uses_only_runtime_and_deadline':True};atomic_json(C/'measured-go-no-go.json',go)
if not count or projection>go['seconds_until_stop'] or available<576*2**30:atomic_json(C/'feasibility-stop.json',dict(go,reason='Do not reduce the fixed subset or model set, interfere with primary fits or risk the owner deadline.'));raise SystemExit('Large CPU check prepared but not executed')
if sha256_file(C/'worker.py')!=plan['worker_sha256'] or sha256_file(C/'run_check.py')!=plan['coordinator_sha256']:raise ValueError('Preserve the pre-result CPU implementation')
r=json.loads(Path(plan['native_readout']).read_text());models=plan['models']
if set(r['models'])|{'B0'}!=set(models) or set(sum(plan['worker_models'],[]))!=set(models) or len(sum(plan['worker_models'],[]))!=8:raise ValueError('Every fixed model must be independently scored once')
plan={**plan,'item_ids':plan['item_ids'][:count],'selected_documents':count,'selected_item_ids_sha256':canonical_hash(plan['item_ids'][:count])};atomic_json(C/'execution-plan.json',plan);D.mkdir(exist_ok=False);started=time.monotonic()
def worker(slot):
 command=['taskset','-c',plan['CPU_affinities'][slot],sys.executable,'-u',str(C/'worker.py'),'--plan',str(C/'execution-plan.json'),'--slot',str(slot)];env={**os.environ,'USE_TORCH_XLA':'0','OMP_NUM_THREADS':'24','MKL_NUM_THREADS':'24','OPENBLAS_NUM_THREADS':'1'}
 with (C/f'slot-{slot}.log').open('x') as log:subprocess.run(command,cwd=R,env=env,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=max(1,(stop-datetime.now(timezone.utc)).total_seconds()+10))
with ThreadPoolExecutor(max_workers=2) as pool:list(pool.map(worker,range(2)))
source_rows={v['item_id']:v for v in map(json.loads,Path(plan['source_file']).open())};ids=plan['item_ids'];docs=[source_rows[i]['document_id'] for i in ids];native_path=Path(plan['native_scores'])
if sha256_file(native_path)!=r['public_scores_sha256']:raise ValueError('Original native scores changed')
with np.load(native_path,allow_pickle=False) as source:
 index={v:i for i,v in enumerate(source['CONFIRM__item_ids'].tolist())};indices=[index[i] for i in ids]
 if [source['CONFIRM__document_ids'][i] for i in indices]!=docs:raise ValueError('Exact source/native identities required')
 native={name:source['CONFIRM__'+name][indices].astype(np.float64) for name in models}
cpu={name:np.load(D/(name+'.npy'),allow_pickle=False) for name in models};comparisons=[(v['treatment'],v['control']) for v in r['metrics']['CONFIRM']['contrasts'].values()];columns=[]
for treatment,control in comparisons:
 a=cpu[treatment]-cpu[control];b=native[treatment]-native[control];columns.extend([a.mean(1),a[:,-16:].sum(1),b.mean(1),b[:,-16:].sum(1),(a-b).mean(1),(a-b)[:,-16:].sum(1)])
b=cluster_bootstrap(np.column_stack(columns),docs,unit='unique source document',repeats=5000,seed=17);contrasts={}
for i,(treatment,control) in enumerate(comparisons):
 values={}
 for j,name in enumerate(['CPU_full_gain','CPU_suffix16_gain','native_subset_full_gain','native_subset_suffix16_gain','CPU_minus_native_paired_full_gain','CPU_minus_native_paired_suffix16_gain']):values[name]={'mean':float(b['mean'][i*6+j]),'CI95':b['CI95'][i*6+j].tolist()}
 contrasts[treatment+'__minus__'+control]={'treatment':treatment,'control':control,'values':values,'statistics':b['metadata']}
arrays={prefix+'__'+name:value for prefix,scores in [('CPU',cpu),('NATIVE',native)] for name,value in scores.items()};arrays['item_ids']=np.asarray(ids);arrays['document_ids']=np.asarray(docs);public=D/'public-evidence';public.mkdir();np.savez_compressed(public/'original-token-scores.npz',**arrays);absolute={prefix:{name:{'full_window_NLL':float(-value.mean()),'suffix16_log_probability':float(value[:,-16:].sum(1).mean())} for name,value in scores.items()} for prefix,scores in [('CPU',cpu),('NATIVE',native)]};maximum=max(abs(v['values']['CPU_minus_native_paired_full_gain']['mean']) for v in contrasts.values());result={'status':'COMPLETE_STANDARD_CPU_FIXED_SUBSET_COMPARISON','utc':utc_now(),'models':models,'documents':len(ids),'actual_CPU_model_forwards':8*len(ids),'new_accelerator_calls':0,'absolute':absolute,'contrasts':contrasts,'maximum_absolute_mean_paired_gain_backend_difference':maximum,'maximum_individual_token_log_probability_backend_difference':max(float(np.abs(cpu[m]-native[m]).max()) for m in models),'public_scores_sha256':sha256_file(public/'original-token-scores.npz'),'native_full_confirmation_readout_sha256':sha256_file(plan['native_readout']),'public_plan':json.loads((C/'prospective-design.json').read_text()),'runtime_source_allocation':go,'selected_item_ids_sha256':canonical_hash(ids),'worker_completions':[json.loads((C/f'slot-{s}-complete.json').read_text()) for s in [0,1]],'actual_wall_seconds':time.monotonic()-started,'scope':'Independent standard Hugging Face FP32 forward implementation on an identity-fixed inspected subset. Native primary outcomes remain unchanged; this is neither a new independent dataset nor an external training replication.'};atomic_json(C/'complete-readout.json',result);atomic_json(public/'readout.json',result);atomic_json(public/'prospective-design.json',result['public_plan']);print(json.dumps({'status':result['status'],'maximum_mean_paired_backend_difference':maximum,'actual_CPU_forwards':8*len(ids),'wall_seconds':result['actual_wall_seconds']}),flush=True)

from pathlib import Path
import json,hashlib,gc,torch
from train.useful_note_precision import compare_distributions
from train.useful_note_recovery import optimizer_digest
R=Path('/home/ubuntu/dario');C=R/'local/diagnostics/checkpoint-export-restart-attribution-v8-v1';O=R/'local/nvme-study/diagnostics/checkpoint-export-restart-attribution-v8-v1/control';V=R/'local/nvme-study/diagnostics/checkpoint-export-SFT-recovery-v8-v2'
def read(p):return json.loads(Path(p).read_text())

def rows(p):return [json.loads(l) for l in Path(p).read_text().splitlines() if l.strip()]

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()

def equal(a,b):
 if isinstance(a,torch.Tensor):return isinstance(b,torch.Tensor) and a.dtype==b.dtype and a.shape==b.shape and torch.equal(a.contiguous().reshape(-1).view(torch.uint8),b.contiguous().reshape(-1).view(torch.uint8))
 if type(a)!=type(b):return False
 if isinstance(a,dict):return a.keys()==b.keys() and all(equal(v,b[k]) for k,v in a.items())
 if isinstance(a,(list,tuple)):return len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
 return a==b
assert read(C/'completion.json')['status']=='CONTROL_RESTART_ATTRIBUTION_STAGE_COMPLETE_PENDING_AUDIT'
assert read(C/'control-resume14/exit.json')['exit_code']==0
close=read(C/'control-resume14/closure.json');assert close['all_original_session_processes_absent'] and close['assigned_physical8_empty'] and close['checkpoint_COMPLETE_sha256']==sha(O/'step-0000014/COMPLETE.json')
manifest=read(O/'step-0000014/COMPLETE.json')
for name,v in manifest['files'].items():assert (O/'step-0000014'/name).stat().st_size==v['bytes'] and sha(O/'step-0000014'/name)==v['sha256']
load=lambda p:torch.load(p,map_location='cpu',weights_only=True,mmap=True)
paths={'uninterrupted_control':V/'control','resumed_candidate':V/'candidate','resumed_control':O};metrics={k:rows(p/'metrics.jsonl') for k,p in paths.items()}
model={k:load(p/'step-0000014/model.pt')['model'] for k,p in paths.items()}
comparisons={}
for left,right in [('resumed_control','resumed_candidate'),('resumed_control','uninterrupted_control')]:
 key=left+'_vs_'+right
 comparisons[key]=dict(full_terminal_active_model_bytes_equal=equal(model[left],model[right]),losses_and_norms=[dict(step=i+1,loss_equal=metrics[left][i]['loss']==metrics[right][i]['loss'],gradient_norm_equal=metrics[left][i]['gradient_norm']==metrics[right][i]['gradient_norm'],loss_difference=metrics[left][i]['loss']-metrics[right][i]['loss'],gradient_norm_difference=metrics[left][i]['gradient_norm']-metrics[right][i]['gradient_norm']) for i in (12,13)],rank_states=[])
del model;gc.collect()
for rank in range(4):
 states={k:load(p/f'step-0000014/rank-{rank:05d}.pt') for k,p in paths.items()}
 for left,right in [('resumed_control','resumed_candidate'),('resumed_control','uninterrupted_control')]:
  comparisons[left+'_vs_'+right]['rank_states'].append(dict(rank=rank,full_optimizer_bytes_equal=equal(states[left]['optimizer'],states[right]['optimizer']),full_saved_CPU_device_RNG_equal=equal(states[left]['rng'],states[right]['rng']),counters_equal=states[left]['counters']==states[right]['counters']))
 saved12=load(O/f'step-0000012/rank-{rank:05d}.pt');actual=read(O/f'recoveries/restart-attribution-001/restored-rank-{rank:05d}.json');assert actual['full_optimizer_bytes_equal'] and actual['full_active_model_bytes_equal'] and actual['tied_parameter_objects_preserved'] and actual['optimizer_digest']==optimizer_digest(saved12['optimizer']);del saved12,states;gc.collect()
 print('Rank',rank,'all terminal comparisons and actual restored12 digest complete',flush=True)
p=read(R/'local/diagnostics/checkpoint-export-SFT-recovery-v8-v2/protocol-control.json');base=read(p['base_protocol']);reference=load(Path(base['native_reference_output'])/'next-logps.pt');checks=[]
for rank in range(4):
 d=O/'recoveries/restart-attribution-001/initialization';q=read(d/f'qualification-rank-{rank:05d}.json');actual=load(d/f'forward-rank-{rank:05d}.pt');x=compare_distributions(reference[q['reference_index']:q['reference_index']+1],actual)[0];assert x['passed'];checks.append(x)
resume=comparisons['resumed_control_vs_resumed_candidate'];same=resume['full_terminal_active_model_bytes_equal'] and all(v['full_optimizer_bytes_equal'] and v['full_saved_CPU_device_RNG_equal'] and v['counters_equal'] for v in resume['rank_states']) and all(v['loss_equal'] and v['gradient_norm_equal'] for v in resume['losses_and_norms'])
report=dict(status='COMPLETE_CONTROL_RESTART_ATTRIBUTION',comparisons=comparisons,two_resumed_trajectories_fully_identical=same,actual_control12_restored_digests_recomputed=True,actual_native_checks=checks,source_fork_lineage_sha256=sha(O.parent/'FORK_LINEAGE.json'),plan_sha256=sha(C/'plan.json'),audit_script_sha256=sha(__file__),original_strict_uninterrupted_gate_relaxed=False,limitation='Engineering attribution only. Original source run completed14; control12was copied as explicit readback-verified fork. Original strict14gate remains a failure if any endpoint differs.')
with (C/'independent-attribution.json').open('x') as f:json.dump(report,f,indent=2)
print(json.dumps(dict(status=report['status'],two_resumed_trajectories_fully_identical=same,comparisons=comparisons),indent=2),flush=True)

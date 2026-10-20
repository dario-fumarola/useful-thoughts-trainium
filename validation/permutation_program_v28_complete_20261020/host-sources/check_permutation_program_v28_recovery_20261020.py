"""Independently compare all actual restored64 optimizer digests before rotation."""
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,gc
import torch
from train.partial_state_driver import ROOT
from train.useful_note_recovery import optimizer_digest
R=Path('/home/ubuntu/dario');W=R/'local/worktrees/controlled-computation-v28-study';assert ROOT==W
C=R/'local/diagnostics/permutation-program-v28-study-v1'
read=lambda p:json.loads(Path(p).read_text())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
plan=read(C/'plan.json');release=read(C/'terminal-release-seed2026102028.json');torch.set_num_threads(4)
results={}
for key in release['jobs']:
 out=Path(plan['jobs'][key]['output']);checkpoint=out/'step-0000064';manifest=read(checkpoint/'COMPLETE.json')
 assert sha(checkpoint/'COMPLETE.json')==release['jobs'][key]['resume_COMPLETE_sha256']
 reports=[]
 for rank in range(4):
  saved=torch.load(checkpoint/f'rank-{rank:05d}.pt',weights_only=True,map_location='cpu',mmap=True)
  actual=read(out/f'recoveries/planned-restart-64/restored-rank-{rank:05d}.json')
  assert all(actual[k] for k in ['full_active_model_bytes_equal','full_optimizer_bytes_equal','tied_parameter_objects_preserved'])
  expected=optimizer_digest(saved['optimizer']);assert expected==actual['optimizer_digest']
  assert saved['step']==64 and saved['counters']==manifest['metadata']['counters']
  reports.append(dict(rank=rank,full_digest_equal=True,restored_receipt_sha256=sha(out/f'recoveries/planned-restart-64/restored-rank-{rank:05d}.json')))
  del saved,actual,expected;gc.collect()
 report=dict(status='PASS_INDEPENDENT_ACTUAL64_RESTORE_DIGESTS',utc=datetime.now(timezone.utc).isoformat(),model=key,
  original64_COMPLETE_sha256=sha(checkpoint/'COMPLETE.json'),all4_actual_optimizer_digests_independently_recomputed=True,ranks=reports,
  source_sha256=sha(__file__),limits='Full state before reset verified;no uninterrupted trajectory or independent immediate device-RNG readback claim.')
 path=C/key/'terminal/recovery-independent-audit.json'
 with path.open('x') as f:json.dump(report,f,indent=2)
 results[key]=report
 print(key,'all4 actual64 restore digests PASS',flush=True)
with (C/'first-seed-recovery-independent-audit.json').open('x') as f:json.dump(dict(status='PASS_ALL_THREE_ACTUAL64_RESTORES',models=results),f,indent=2)

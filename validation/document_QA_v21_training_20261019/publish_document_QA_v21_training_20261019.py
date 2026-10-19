"""Publish actual terminal training/state/protection evidence,not capability."""
from pathlib import Path
from datetime import datetime,timezone
import gzip,hashlib,json,shutil
import numpy as np
import torch
R=Path('/home/ubuntu/dario');C=R/'local/diagnostics/document-QA-v21-study-v1'
read=lambda p:json.loads(Path(p).read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert read(C/'terminal-pipeline-completion.json')['status']=='ALL_SIX_TERMINAL256_MODELS_AUDITED_PROTECTED_AND_BOUND'
a=read(C/'terminal-independent-audit.json');protection=read(C/'terminal-protection.json');assert a['status']=='PASS_ALL_SIX_TERMINAL256_QA_TRAINING_STATES_NATIVE48_ORDER_COUNTERS_AND_LABELS'
assert protection['status']=='PASS_ALL_SIX_TERMINAL_MODELS_AND_LATEST_FULL256_PROTECTED'
P=R/'validation/document_QA_v21_training_20261019';P.mkdir(exist_ok=False);payloads={}
for j in read(C/'plan.json')['jobs']:
 source=Path(j['output'])/'metrics.jsonl';target=P/(j['key']+'-all256-updates.jsonl.gz');target.write_bytes(gzip.compress(source.read_bytes(),mtime=0));assert gzip.decompress(target.read_bytes())==source.read_bytes();payloads[str(target.relative_to(R))]=dict(rows=256,bytes=target.stat().st_size,sha256=sha(target),original_sha256=sha(source))
 for step in (128,256):
  source=Path(j['output'])/f'step-{step:07d}/COMPLETE.json';shutil.copyfile(source,P/(j['key']+f'-full{step}-manifest.json'))
arrays={};lookup={};mapping=[]
for j in read(C/'plan.json')['jobs']:
 O=Path(j['output'])
 for rank in range(8):
  native=torch.load(O/f'forward-rank-{rank:05d}.pt',weights_only=True,map_location='cpu').numpy();digest=hashlib.sha256(native.tobytes()).hexdigest()
  if digest not in lookup:
   name='actual_'+str(len(lookup));arrays[name]=native;lookup[digest]=name
  receipt=read(O/f'qualification-rank-{rank:05d}.json');mapping.append(dict(model=j['key'],rank=rank,reference_index=receipt['reference_index'],actual_array=lookup[digest],observation_sha256=sha(O/f'forward-rank-{rank:05d}.pt'),receipt=receipt))
base=read(read(C/'plan.json')['jobs'][0]['protocol']);reference=torch.load(Path(read(base['base_protocol'])['native_reference_output'])/'next-logps.pt',weights_only=True,map_location='cpu').numpy();arrays['CPU_reference']=reference
np.savez_compressed(P/'actual-initial-native-arrays.npz',**arrays)
(P/'actual48-initial-native-mapping.json').write_text(json.dumps(dict(observations=48,unique_actual_arrays=len(lookup),canonical_contexts=len(reference),mapping=mapping,limits='Initial forward checks only,not48independentdocuments or capability. Rawarrays de-duplicated only by exactbytes;mapping preserves everyobservation.'),indent=2))
shutil.copyfile(Path(__file__),P/Path(__file__).name)
report=dict(status='ALL_SIX_TERMINAL256_TRAINING_STATES_AUDITED_PROTECTED_AND_BOUND',utc=datetime.now(timezone.utc).isoformat(),actual_updates=1536,models=6,seeds=2,release_sha256=sha(C/'release.json'),independent_audit=a,protection=protection,payloads=payloads,native_arrays=dict(path=str((P/'actual-initial-native-arrays.npz').relative_to(R)),sha256=sha(P/'actual-initial-native-arrays.npz'),observations=48,unique_actual_arrays=len(lookup),canonical_contexts=len(reference)),publisher_sha256=sha(__file__),limits=['Actual training,state,order,counter and protection evidence only. No held capability readout.','Allruns start from released1.7B;2,048 examples/fourpasses. Moretasktargetlabels in augmentedarms;equalupdates not equal usefulsupervision.','Two data-order seeds,not broadtrainingseedvariance. Same-instance EBS protection,not off-instance.','Numericalparity and successfulupdates do not establish reasoning ornovelty relative topretraining.'])
F=R/'validation/document_QA_v21_training_20261019.json';assert not F.exists();F.write_text(json.dumps(report,indent=2));print(json.dumps(dict(status=report['status'],updates=1536,unique_native_arrays=len(lookup)),indent=2),flush=True)

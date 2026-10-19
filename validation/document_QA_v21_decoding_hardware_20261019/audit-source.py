"""Wait for decoder closure;independently audit native arrays/actions,not accuracy."""
from pathlib import Path
from datetime import datetime,timezone
from collections import Counter
import hashlib,json,os,shutil,subprocess,time
import torch
from transformers import AutoTokenizer
from train.document_qa_answers import parse_answer
from train.useful_note_precision import compare_distributions
R=Path('/home/ubuntu/dario');C=R/'local/diagnostics/document-QA-v21-FP32-decode-qualification-v2';W=R/'local/worktrees/document-qa-eval-v21';O=C/'execution'
read=lambda p:json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for block in iter(lambda:f.read(8*2**20),b''):h.update(block)
 return h.hexdigest()
started=time.monotonic()
while not (C/'closure.json').exists():
 live=subprocess.run(['tmux','has-session','-t','document-QA-v21-decode-qualification'],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).returncode==0
 if (C/'failure.json').exists() or not live and time.monotonic()-started>35:
  raise RuntimeError('Decode supervisor stopped before complete closure;no retry')
 if datetime.now(timezone.utc)>=datetime.fromisoformat('2026-10-19T17:10:00+00:00'):raise TimeoutError('No automatic retry')
 time.sleep(3)
p=read(O/'protocol.json');d=O/'answers/shard-0';assert all(sha(W/n)==v for n,v in p['source_files'].items())
closure=read(C/'closure.json');assert closure['all_original_session_processes_absent'] and closure['physical14_empty']
assert read(C/'exit.json')['exit_code']==0 and read(C/'exit.json')['external_stop'] is None
reference=torch.load(O/'native-reference.pt',map_location='cpu',weights_only=True);actual=torch.load(d/'native-logps.pt',map_location='cpu',weights_only=True)
canonical=compare_distributions(reference[0:1],actual)[0];assert canonical['passed'] and actual.dtype==torch.float32
vectors=torch.load(d/'cache-prefill-logps.pt',map_location='cpu',weights_only=True)
for v in vectors.values():assert v.shape==(1,151936) and v.dtype==torch.float32 and torch.isfinite(v).all()
cache=compare_distributions(vectors['ordinary'],vectors['cached'])[0];assert cache['max_logp_error']<=1e-3
cpu_receipt=read(C/'long-context-CPU-reference.json');assert cpu_receipt['reference_sha256']==sha(C/'long-context-CPU-logps.pt') and cpu_receipt['signature']==p['signature']
cpu=torch.load(C/'long-context-CPU-logps.pt',map_location='cpu',weights_only=True)
long_checks={k:compare_distributions(cpu,v)[0] for k,v in vectors.items()};assert all(v['max_logp_error']<=1e-3 for v in long_checks.values())
rows=[json.loads(l) for l in (d/'attempts.jsonl').read_text().splitlines()];assert len(rows)==6 and all(v['status']=='ok' for v in rows)
assert Counter(v['mode'] for v in rows)==Counter({m:2 for m in p['modes']})
assert closure['attempts_sha256']==sha(d/'attempts.jsonl')
tok=AutoTokenizer.from_pretrained(p['model_dir'],local_files_only=True,trust_remote_code=False);defined=set(tok.get_vocab().values());formats=Counter();caps=Counter();timings={}
for row in rows:
 actions,content=row['actions'],row['content_ids'];assert row['n_actions']==len(actions)<=256 and all(v in defined for v in actions)
 assert tok.decode(content,skip_special_tokens=False,clean_up_tokenization_spaces=False)==row['raw_text']
 stop=row['stop_type']
 if stop.startswith('token:'):
  assert int(stop[6:]) in p['stop_ids'] and actions[-1]==int(stop[6:]) and content==actions[:-1]
 elif stop=='qa_answer_line':
  assert actions==content;parsed=parse_answer(row['raw_text'],row['parser_mode']);assert parsed.status=='complete' and row['text']==row['raw_text'][:parsed.end]
 elif stop=='length_cap':assert actions==content and len(actions)==256
 else:raise ValueError('Unsupported actual decoder stop type')
 parsed=parse_answer(row['text'],row['parser_mode'],terminated=stop.startswith('token:') or stop=='qa_answer_line',capped=stop=='length_cap')
 formats[parsed.status]+=1;caps[row['mode']]+=int(stop=='length_cap');timings.setdefault(row['mode'],[]).append(row['wall_seconds'])
# Reconstruct/preserve the unused prelaunch V1 source exactly;the sole change
# before V2 was resolving caller paths. V1 never launched on hardware.
v1=R/'local/diagnostics/document-QA-v21-FP32-decode-qualification-v1';old=read(v1/'execution/protocol.json')
s=(W/'train/document_qa_evaluate.py').read_text()
for line in ["    output,data_dir,model_dir,base_protocol=(Path(v).resolve() for v in (output,data_dir,model_dir,base_protocol))\n","    binding=Path(binding).resolve() if binding else None\n","    release=Path(release).resolve() if release else None\n","    original,output=Path(original).resolve(),Path(output).resolve()\n","    output=Path(output).resolve()\n"]:s=s.replace(line,'')
assert hashlib.sha256(s.encode()).hexdigest()==old['source_files']['train/document_qa_evaluate.py']
(v1/'unused-prelaunch-source.py').write_text(s)
A=R/'local/backups/engineering'/C.name;A.mkdir(parents=True,exist_ok=False);files={}
for label,folder in [('actual',C),('unused-V1-prelaunch',v1)]:
 for src in sorted(folder.rglob('*')):
  if not src.is_file():continue
  dst=A/label/src.relative_to(folder);dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst)
  assert sha(src)==sha(dst)
  with dst.open('rb') as f:os.fsync(f.fileno())
  files[str(dst.relative_to(A))]=dict(bytes=dst.stat().st_size,sha256=sha(dst))
for src in [Path(__file__),R/'local/analysis/prepare_document_QA_v21_decode_supervisor_20261019.py',R/'local/analysis/document_QA_v21_CPU_long_reference_20261019.py']:
 dst=A/'source'/src.name;dst.parent.mkdir(exist_ok=True);shutil.copy2(src,dst);assert sha(src)==sha(dst);files[str(dst.relative_to(A))]=dict(bytes=dst.stat().st_size,sha256=sha(dst))
for directory in [v for v in A.rglob('*') if v.is_dir()]+[A]:
 fd=os.open(directory,os.O_RDONLY|os.O_DIRECTORY);os.fsync(fd);os.close(fd)
archive=dict(status='ALL_ACTUAL_DECODE_ARRAYS_ATTEMPTS_CLOSURE_AND_SOURCE_READBACK_FSYNC',files=files,limits='Same-instance engineering protection. No accuracy result.')
(A/'ARCHIVE.json').write_text(json.dumps(archive,indent=2))
result=dict(status='PASS_ACTUAL_TRAIN_ONLY_LONG_CONTEXT_FP32_QA_DECODING',utc=datetime.now(timezone.utc).isoformat(),source_commit=read(C/'release.json')['source_commit'],source_files=p['source_files'],protocol_sha256=sha(O/'protocol.json'),canonical_native=canonical,actual_cache_ordinary=cache,long_CPU_checks=long_checks,
 attempts=6,execution_errors=0,parser_status_counts=dict(formats),caps_by_mode=dict(caps),actual_attempt_seconds=timings,logical_core=56,physical_device=14,physical_closure=closure,
 EBS_archive=str(A),EBS_archive_sha256=sha(A/'ARCHIVE.json'),audit_source_sha256=sha(__file__),
 limits=['Two disposable TRAIN questions,six attempts,not an accuracy/capability sample. Parser failures/caps recorded but not a mechanics veto.',
 'One actual long-context CPU reference;canonical native check repeats one preexisting reference. No all-512 CPU agreement claim.',
 'FP32 prefill3584/cache3840 only;new shapes/weights need their own native and cached checks.','Training qualification and scientific release remain separate.'])
(C/'independent-audit.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2),flush=True)

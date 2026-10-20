"""Record all three first periodic saves/readbacks and actual resumed updates."""
from pathlib import Path
from datetime import datetime,timezone
import json,os,shutil,hashlib
R=Path('/home/ubuntu/dario');C=R/'local/diagnostics/permutation-program-v28-study-v1'
P=R/'validation/permutation_program_v28_periodic_recovery_20261020';P.mkdir(exist_ok=False)
read=lambda p:json.loads(Path(p).read_text())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
files={};models={}
def copy(src,name):
 dst=P/name;shutil.copyfile(src,dst);assert sha(src)==sha(dst)
 with dst.open('rb') as stream:os.fsync(stream.fileno())
 files[str(dst.relative_to(R))]=sha(dst)
for arm in ('direct','trace','pause'):
 key='program-'+arm+'-seed2026102028-v1';o=R/'local/science-runs/controlled-computation-v28'/key;d=o/'step-0001024'
 latest=json.loads((o/'metrics.jsonl').read_text().splitlines()[-1]);sample=json.loads((C/key/'terminal/host-samples.jsonl').read_text().splitlines()[-1])
 receipt=read(d/'READBACK.json');manifest=read(d/'COMPLETE.json');retired=read(o/'step-0000064/PAYLOAD_RETIRED.json')
 assert receipt['status']=='FULL_STATE_READBACK_FSYNC' and receipt['step']==1024
 assert receipt['COMPLETE_sha256']==sha(d/'COMPLETE.json') and receipt['files']==manifest['files']
 assert latest['step']>1024 and retired['status']=='ROLLING_PAYLOAD_RETIRED'
 assert retired['replacement']['COMPLETE_sha256']==receipt['COMPLETE_sha256']
 assert not (C/key/'terminal/failure.json').exists()
 for name in ('COMPLETE.json','READBACK.json'):copy(d/name,key+'-'+name)
 copy(o/'step-0000064/PAYLOAD_RETIRED.json',key+'-old64-retirement.json')
 snapshot=dict(latest_update={k:latest[k] for k in ('step','time_utc','elapsed_seconds','counters')},
  host_sample_utc=sample['utc'],group_PSS_bytes=sum((v.get('memory') or {}).get('Pss',0) for v in sample['processes']),
  host_available_bytes=sample['MemAvailable_bytes'])
 path=P/(key+'-resumed-snapshot.json');path.write_text(json.dumps(snapshot,indent=2)+'\n');files[str(path.relative_to(R))]=sha(path)
 models[key]=snapshot
copy(R/'local/diagnostics/permutation-program-v28-evaluation-CPU-v1/fixed-report-example-identities.json','fixed-report-example-identities.json')
report=dict(status='ALL_FIRST_PERIODIC1024_READBACKS_AND_RESUMPTIONS_OBSERVED',utc=datetime.now(timezone.utc).isoformat(),
 experiment_id='controlled-computation-v28',models=models,source_commit='ce04236ecd4d7eaa9dae8471821e600561748069',
 held_model_outcomes=0,root_free_bytes=shutil.disk_usage(R).free,files=files,
 limits=['Runtime hashes/fsync receipts and actual resumed updates; independent whole-state tensor audits remain the first64 and terminal checks.',
 'Observed first periodic save stability is not indefinite stability or learned capability.',
 'Only new-study old intermediate payloads rotate; development64 models/manifests remain. Historical originals/cache untouched.',
 'Same-instance EBS; not off-instance protection. Fixed terminal endpoints and all-six-model held guard remain unchanged.'])
path=R/'validation/permutation_program_v28_periodic_recovery_20261020.json';path.write_text(json.dumps(report,indent=2)+'\n')
b=R/'local/backups/diagnostics/permutation-program-v28-periodic-recovery-v1';b.mkdir(parents=True,exist_ok=False)
for src in [path,*P.iterdir()]:
 dst=b/src.name;shutil.copyfile(src,dst);assert sha(src)==sha(dst)
 with dst.open('rb') as stream:os.fsync(stream.fileno())
fd=os.open(b,os.O_RDONLY);os.fsync(fd);os.close(fd)
print(json.dumps(dict(status=report['status'],files=len(files),sha256=sha(path))))

"""Publish actual first64 audits and recovery receipts, without held outcomes."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,os,shutil

R=Path('/home/ubuntu/dario');C=R/'local/diagnostics/permutation-program-v28-study-v1'
P=R/'validation/permutation_program_v28_continuation_20261020'
P.mkdir(exist_ok=False)
read=lambda p:json.loads(Path(p).read_text())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
files={}
def copy(src,name):
    dst=P/name;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src,dst)
    assert sha(src)==sha(dst)
    with dst.open('rb') as f:os.fsync(f.fileno())
    files[str(dst.relative_to(R))]=sha(dst)
release=read(C/'terminal-release-seed2026102028.json')
assert release['status']=='RELEASED_V28_SCIENCE_TERMINAL4096'
assert read(C/'first-seed-recovery-independent-audit.json')['status']=='PASS_ALL_THREE_ACTUAL64_RESTORES'
for key in release['jobs']:
    a=C/key/'first64/independent-all-state-audit.json'
    b=C/key/'terminal/recovery-independent-audit.json'
    assert read(a)['status']=='PASS_V28_FIRST64_ALL_STATE'
    assert read(b)['status']=='PASS_INDEPENDENT_ACTUAL64_RESTORE_DIGESTS'
    assert sha(a)==release['first64_independent_audits'][key]['sha256']
    copy(a,key+'-first64-audit.json');copy(b,key+'-recovery-audit.json')
    copy(C/key/'terminal/launch.json',key+'-continuation-launch.json')
copy(C/'terminal-release-seed2026102028.json','terminal-release.json')
copy(C/'first-seed-recovery-independent-audit.json','all-recoveries.json')
for name in ('permutation_program_state_audit_v28_20261020.py','audit_permutation_program_v28_job_20261020.py','check_permutation_program_v28_recovery_20261020.py','release_permutation_program_v28_continuation_20261020.py'):
    copy(R/'local/analysis'/name,name)
report=dict(status='PASS_FIRST_SEED_ACTUAL64_AUDITS_AND_RESTORES_CONTINUATION_ACTIVE',
 experiment_id='controlled-computation-v28',published_utc=datetime.now(timezone.utc).isoformat(),
 source_commit=release['source_commit'],deadline_utc=release['deadline_utc'],
 jobs=list(release['jobs']),first64_all_model_master_moment_and_order_audits=True,
 actual_fresh_process_restores=12,all_optimizer_digests_independently_recomputed=True,
 planned_terminal_step=4096,held_model_outcomes=0,files=files,
 limits=['Qualification and recovery are not scientific utility results.','No uninterrupted-trajectory or immediate independent device-RNG readback claim.','All six fixed terminal models are required before held model outcomes.','Full checkpoints remain same-instance EBS; development64 models are also retained on NVMe.'])
rp=R/'validation/permutation_program_v28_continuation_20261020.json'
with rp.open('x') as f:json.dump(report,f,indent=2)
backup=R/'local/backups/diagnostics/permutation-program-v28-continuation-v1';backup.mkdir(parents=True,exist_ok=False)
for src in [rp,*P.iterdir()]:
    assert src.is_file();dst=backup/src.name;shutil.copyfile(src,dst);assert sha(src)==sha(dst)
    with dst.open('rb') as f:os.fsync(f.fileno())
fd=os.open(backup,os.O_RDONLY);os.fsync(fd);os.close(fd)
print(json.dumps(dict(status=report['status'],files=len(files),report_sha256=sha(rp))))

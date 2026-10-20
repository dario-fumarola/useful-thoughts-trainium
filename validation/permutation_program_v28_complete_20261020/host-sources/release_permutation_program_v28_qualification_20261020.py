"""Release only one bounded disposable4-worker all-view/restart check."""
from pathlib import Path
from datetime import datetime,timezone,timedelta
import hashlib,json,shutil,subprocess,importlib.util,py_compile
from train.partial_state_driver import ROOT,source_files,validate
R=Path('/home/ubuntu/dario');W=R/'local/worktrees/controlled-computation-v28';C=R/'local/diagnostics/permutation-program-v28-world4-qualification-v1'
assert ROOT==W
read=lambda f:json.loads(Path(f).read_text())
def sha(f):return hashlib.sha256(Path(f).read_bytes()).hexdigest()
def write(f,v):
 with Path(f).open('x') as g:json.dump(v,g,indent=2)
p=read(C/'protocol.json');validate(p)
assert p['source_files']==source_files()
assert sha(p['data_file'])==p['data_sha256'] and sha(p['data_manifest_file'])==p['data_manifest_sha256']
assert not subprocess.check_output(['git','status','--porcelain'],cwd=W,text=True).strip()
assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=W,text=True).strip().startswith('ea926ca')
assert '1838 passed' in (W/'local/checks/tests.472JZc/tests.txt').read_text()
assert read(R/'local/diagnostics/permutation-program-v28-CPU-v1/packing-readiness-v2.json')['native_packs']==131072
assert read(R/'local/diagnostics/permutation-program-v28-CPU-v1/protection.json')['status']=='PASS_READBACK_AND_FSYNC_CPU_DATA_SOURCE_AND_EVIDENCE'
old=read(R/'local/diagnostics/document-evidence-format-weight-v26-world8-qualification-v1/protocol.json')
exceptions={'train/partial_state_driver.py','train/permutation_program_data.py','train/permutation_program_packing.py'}
assert {k:v for k,v in p['source_files'].items() if k not in exceptions}=={k:v for k,v in old['source_files'].items() if k not in exceptions}
for f in ['qualify.py','independent-audit.py']:py_compile.compile(str(C/f),doraise=True)
spec=importlib.util.spec_from_file_location('program_qualification_supervisor',C/'qualify.py');h=importlib.util.module_from_spec(spec);spec.loader.exec_module(h)
top=h.topology();device=[v for v in top if v['neuron_device']==12];assert len(device)==1 and not device[0]['neuron_processes'] and device[0]['logical_neuroncore_config']==2
assert h.available()>768*2**30
out=R/'local/nvme-study/diagnostics/permutation-program-v28-world4-qualification-v1';assert not out.exists()
# Both full checkpoints remain, using conservative16bytes/parameter per state.
assert (R/'local/nvme-study').is_mount() and shutil.disk_usage(R/'local/nvme-study').free>35*2**30+2*16*p['base_signature']['parameters']+2*2**30
assert shutil.disk_usage(R).free>35*2**30+2*2**30
assert shutil.disk_usage(R/'local/nvme-cpt').free>35*2**30+8*2**30
science=read(R/'local/diagnostics/document-evidence-format-weight-v26-control-v1/terminal-protection.json')
assert sha(Path(science['latest_full']['EBS'])/'COMPLETE.json')==science['latest_full']['COMPLETE_sha256']
now=datetime.now(timezone.utc)
r=dict(status='RELEASED_BOUNDED_PROGRAM_ALL_VIEW_MECHANICS_ONLY',utc=now.isoformat(),source_root=str(W),source_commit='ea926ca',protocol_sha256=sha(C/'protocol.json'),supervisor_sha256=sha(C/'qualify.py'),audit_sha256=sha(C/'independent-audit.py'),release_source_sha256=sha(__file__),physical_devices=[12],logical_cores='48-51',deadline_utc=(now+timedelta(minutes=55)).isoformat(),maximum_reserved_chip_hours=1.,output=str(out),original_full_states=[4,8],checkpoint_policy='Both disposablefull4/full8 originals retained on NVMe-study. Only small actual engineering evidence copied to EBS. Latest complete V26 science backup remains untouched.',CPU_checks='1838tests and smoke/compile;38912independent data/token checks;131072native TRAIN packs.',numerical_scope='Numerical kernels/ZeRO/export/restore sources unchanged from V26; only new packer and bounded driver dispatch. Four views in every update. Native/model/master/moment/order/mask/full-state evidence required.',decision='Eight-update actual mechanics only; disposable model cannot enter science. No longer learning study or evaluation release.',no_automatic_retry=True,not_capability_evidence=True,storage_free_bytes={str(d):shutil.disk_usage(d).free for d in [R,R/'local/nvme',R/'local/nvme-cpt',R/'local/nvme-study']})
write(C/'qualification-release.json',r)
print(json.dumps({k:r[k] for k in ['status','utc','deadline_utc','physical_devices','output']}))

"""Protect the completed independently checked presentation and actual parity arrays."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib
import json
import os
import shutil
import numpy as np
import torch

R=Path('/home/ubuntu/dario');C=R/'local/diagnostics/real-source-eval-v17-study-v1'
read=lambda p:json.loads(Path(p).read_text())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert read(C/'final-readout-completion.json')['status']=='ALL_ELEVEN_COMPLETE_ANALYZED_INDEPENDENTLY_CHECKED_RENDERED_AND_EXAMPLED'
p=R/'validation/real_source_balanced_v17_complete_20261019.json';r=read(p);out=p.with_suffix('')
check=read(out/'independent-readout-check.json');assert check['report_sha256']==sha(p)
assert check['status']=='PASS_ALL_PUBLIC_NATIVE_TEXT_VALUES_AST_GOLD_COUNTS_CIS_AND_LITERAL_GATES'
D=out/'analysis-source';D.mkdir(exist_ok=False)
for name in ['analyze_real_source_v17_20261019.py','check_real_source_readout_v17_20261019.py',
             'render_real_source_v17_20261019.py','save_real_source_examples_v17_20261019.py',
             'watch_real_source_final_readout_v17_20261019.py','account_real_source_scalar_errors_v17_20261019.py','protect_real_source_readout_v17_20261019.py']:
    source=R/'local/analysis'/name;shutil.copyfile(source,D/name)
assert sha(D/'analyze_real_source_v17_20261019.py')==r['analysis_source_sha256']
assert sha(D/'check_real_source_readout_v17_20261019.py')==check['checker_sha256']
parities=[]
for item in read(C/'plan.json')['models']:
    registry=read(item['registry'])
    for panel in ['DEV_confirm512','DEV_ordinary256']:
        folder=Path(registry['score_output_root'])/panel;run=read(folder/'run.json')
        assert run['status']=='COMPLETE_ALL_FIXED_DOCUMENT_LIKELIHOODS'
        for name,digest in run['span_reference_files'].items():assert sha(folder/name)==digest
        cpu=torch.load(folder/'CPU-fixed-first4-each-mode.pt',map_location='cpu',weights_only=True).numpy().astype(np.float64)
        native=torch.load(folder/'XLA-fixed-first4-each-mode.pt',map_location='cpu',weights_only=True).numpy().astype(np.float64)
        assert cpu.shape==native.shape==(20 if panel=='DEV_confirm512' else 4,16)
        assert np.isfinite(cpu).all() and np.isfinite(native).all()
        maximum=float(np.abs(native-cpu).max());mean_drift=float(abs((native-cpu).sum(1).mean()))
        assert maximum<=.001 and mean_drift<=.001 and run['fixed_first4_per_mode_parity']['passed']
        parities.append(dict(model_id=item['id'],panel=panel,
            fixed_order=run['span_reference_order'],CPU_logps16=cpu.tolist(),XLA_logps16=native.tolist(),
            independently_recomputed_max_token_error=maximum,independently_recomputed_absmean_span_drift=mean_drift,
            original_result=run['fixed_first4_per_mode_parity'],original_reference_sha256=run['span_reference_files']))
assert len(parities)==22
q=out/'actual-fixed-CPU-XLA-parity-arrays.json'
q.write_text(json.dumps(dict(status='PASS_ALL_ELEVEN_MODELS_BOTH_FIXED_REFERENCE_PANELS',
    rows=264,token_pairs=4224,panels=parities,
    limits='Four fixed documents per mode per model;repeated references,not264 independent documents orall-panelCPU parity. No new model forwards.'),indent=2))
A=R/'local/backups/evaluation/real-source-balanced-v17-complete/PUBLIC-READOUT';A.mkdir(exist_ok=False)
receipts={}
for source in [p,*[f for f in out.rglob('*') if f.is_file()]]:
    assert shutil.disk_usage(R).free>35*2**30+source.stat().st_size
    target=A/source.relative_to(R/'validation');target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,target)
    with target.open('rb') as stream:os.fsync(stream.fileno())
    assert sha(source)==sha(target)
    receipts[str(source.relative_to(R))]=dict(bytes=source.stat().st_size,sha256=sha(source))
manifest=dict(status='ALL_COMPLETE_RAW_DERIVED_OUTPUTS_AND_CURRENT_ANALYSIS_SOURCES_READBACK_PROTECTED',
    utc=datetime.now(timezone.utc).isoformat(),files=receipts,primary_report_sha256=sha(p),
    independent_check_sha256=sha(out/'independent-readout-check.json'),
    limits='Same-instance EBS readback/fsync protection;not off-instance backup. Original raw artifacts stay in the separate unchanged archive.')
(A/'READBACK.json').write_text(json.dumps(manifest,indent=2))
with (A/'READBACK.json').open('rb') as stream:os.fsync(stream.fileno())
(out/'readout-publication-manifest.json').write_text(json.dumps(manifest,indent=2))
print(json.dumps(dict(status=manifest['status'],files=len(receipts),parity_token_pairs=4224),indent=2),flush=True)

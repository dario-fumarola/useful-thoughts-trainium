"""Publish the closed TRAIN-only diagnostic, retaining its limited interpretation."""
from pathlib import Path
from datetime import datetime,timezone
from collections import Counter
import gzip,hashlib,json,os,shutil
R=Path('/home/ubuntu/dario');C=R/'local/diagnostics/selector-TRAIN-decisions-CPU-v1'
F=R/'validation/selector_TRAIN_decisions_CPU_complete_20261020.json';A=F.with_suffix('')
read=lambda p:json.loads(p.read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
result=read(C/'result.json');check=read(C/'independent-check.json');protection=read(C/'protection.json')
assert check['status']=='PASS_INDEPENDENT_TRAIN_CATEGORIES_TOKEN_BOUNDARIES_COUNTS_AND_FIXED_HF_FORWARD_SPOTS'
assert check['result_sha256']==sha(C/'result.json') and read(C/'exit.json')['exit_code']==read(C/'independent-check-exit.json')['exit_code']==0
assert protection['status']=='PASS_CPU_DIAGNOSTIC_ORIGINAL_OUTPUTS_AND_SOURCES_READBACK_FSYNC'
assert sha(Path(protection['EBS'])/'ARCHIVE.json')==protection['archive_sha256']
assert not F.exists() and not A.exists();A.mkdir();files={}
def copy(p,name=None):
    dst=A/(name or p.name);shutil.copyfile(p,dst);assert sha(p)==sha(dst);files[dst.name]=dict(bytes=dst.stat().st_size,sha256=sha(dst),original_sha256=sha(p))
def compressed(p,name):
    dst=A/name;dst.write_bytes(gzip.compress(p.read_bytes(),mtime=0));assert gzip.decompress(dst.read_bytes())==p.read_bytes();files[name]=dict(bytes=dst.stat().st_size,sha256=sha(dst),original_sha256=sha(p))
for name in ['design.json','result.json','independent-check.json','check-contract.json','actual-CPU-model-check.json','launch.json','exit.json','independent-check-exit.json']:copy(C/name)
compressed(C/'inputs.jsonl','fixed-TRAIN256-tokenized-inputs.jsonl.gz');compressed(C/'observations.jsonl','all-TRAIN256-observations.jsonl.gz')
for name in ['selector_train_decisions_cpu_20261019.py','check_selector_train_decisions_cpu_20261019.py','watch_selector_train_decisions_cpu_20261019.py',Path(__file__).name]:copy(R/'local/analysis'/name)
old=R/'local/diagnostics/document-evidence-weight-v25-ablation-v1/TRAIN-fit'
native={v['native']['item_id']:v for v in [json.loads(l) for l in (old/'native-and-scored.jsonl').read_text().splitlines()]}
labels={v['item_id']:v for v in [json.loads(l) for l in (old/'TRAIN256-labels.jsonl').read_text().splitlines()]}
observed={v['item_id']:v for v in [json.loads(l) for l in (C/'observations.jsonl').read_text().splitlines()]}
examples=[]
for key in read(C/'check-contract.json')['fixed_item_ids']:
    examples.append(dict(item_id=key,question=labels[key]['question']['question'],annotated_support_indices=labels[key]['support_indices'],native=native[key],CPU_decisions=observed[key]))
p=A/'fixed-four-TRAIN-examples.json';p.write_text(json.dumps(dict(selection=read(C/'check-contract.json')['selection'],fixed_item_ids=read(C/'check-contract.json')['fixed_item_ids'],examples=examples),indent=2));files[p.name]=dict(bytes=p.stat().st_size,sha256=sha(p))
categories=Counter()
for v in result['strata'].values():categories.update(v['native_categories'])
assert sum(categories.values())==256 and categories['exact_annotated_set']==203
summary=dict(status='COMPLETE_CHECKED_FROZEN_TRAIN_POINTER_DECISION_DIAGNOSTIC_CPU_ONLY',utc=datetime.now(timezone.utc).isoformat(),experiment_id=C.name,
    question=read(C/'design.json')['question'],model=read(C/'design.json')['model'],items=256,strata=result['strata'],all_native_categories=dict(categories),
    first_divergent_token_comparisons=dict(required_continuation_decisions=416,closure_token_preferred=8,token_pairs=check['token_surface_pairs']),
    interpretation=['Only5 of53 annotation-set mismatches are a correct canonical ID prefix followed by early closure. Thirty omit an earlier annotated ID;17 include a nonannotated ID;one is unsupported.',
        'This weakens a simple correct-prefix-then-stop account of most mismatches. It does not identify a unique cause or adjudicate whether the author annotations are semantically sufficient.',
        'Under correct annotated prefixes the chosen closure serialization token outranks the required token at8/416 boundaries. All boundaries compare standalone closing bracket with bracket-plus-newline: these are not total field-termination probabilities.',
        'CPU FP32 and native BF16 are different numerical paths; conditioning on correct history is different from the sampled policy history. No causal decoding or generalization claim follows.',
        'No model update, new held question, revised V25 fitting gate or changed V26 comparison. This is a descriptive optimization diagnostic.'],
    independent_checks=check,CPU_compute_wall_seconds=result['wall_seconds'],CPU_threads=16,accelerator_calls=0,
    protected_originals={k:v for k,v in protection.items() if k!='files'},payloads=files)
with F.open('x') as f:json.dump(summary,f,indent=2)
B=R/'local/backups/engineering/selector-TRAIN-decisions-CPU-v1/PUBLICATION';B.mkdir(exist_ok=False);saved={}
for p in [F,*A.iterdir()]:
    dest=B/p.relative_to(R/'validation');dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dest)
    with dest.open('rb') as f:os.fsync(f.fileno())
    assert sha(p)==sha(dest);saved[str(p.relative_to(R/'validation'))]=sha(dest)
for d in [*[p for p in B.rglob('*') if p.is_dir()],B]:
    fd=os.open(d,os.O_RDONLY|os.O_DIRECTORY);os.fsync(fd);os.close(fd)
with (C/'publication-protection.json').open('x') as f:json.dump(dict(status='PASS_CPU_DIAGNOSTIC_PUBLICATION_READBACK_FSYNC',report=str(F),report_sha256=sha(F),files=saved,EBS=str(B),same_instance=True),f,indent=2)
print(json.dumps(dict(status=summary['status'],report=str(F),report_sha256=sha(F),protected_files=len(saved))))

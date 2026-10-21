"""Package all V30 actions and token scores without original source windows."""
import argparse
import json
import os
from pathlib import Path
import shutil
import tarfile

from scripts.bundle_real_cpt_v29 import project,sha,write
from train.real_cpt_readout import numeric_masks


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--control',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();root=Path(__file__).resolve().parents[1];c=a.control
    result=json.loads((c/'readout-v1/readout.json').read_text())
    assert json.loads((c/'independent-v1/independent-check.json').read_text())['status']=='PASS_ALL_CONTEXTS_AND_PRIMARY_STATISTICS_FIXED_HF'
    a.output.mkdir(parents=True,exist_ok=False);e=a.output/'v30-evidence';e.mkdir()
    projections={}
    for stage in ('generate','score-T','score-B0'):
        for shard in range(4):
            name='notes.jsonl' if stage=='generate' else 'scores.jsonl'
            source=c/f'runs/{stage}-{shard}'/name;target=e/f'outputs/{stage}-{shard}.jsonl'
            projections[str(target.relative_to(e))]=project(source,target)
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(root/'local/assets/qwen3-1.7b-base',local_files_only=True)
    with (e/'cohort-math.jsonl').open('x') as f:
        for line in (c/'diagnostic-labels.jsonl').open():
            row=json.loads(line)
            value={k:row[k] for k in ('item_id','document_id','normalized_expression','numeric_value','full_calculation_text')}
            value['native_masks']=numeric_masks(row,tok)
            f.write(json.dumps(value,separators=(',',':'))+'\n')
        f.flush();os.fsync(f.fileno())
    shutil.copytree(root/'validation/candidate_utility_v30_complete_20261021',e/'results')
    source_names=set(json.loads((c/'release.json').read_text())['sources'])|{
        'scripts/report_candidate_utility_v30.py','scripts/check_candidate_utility_v30_results.py',
        'scripts/check_candidate_sampling_v30.py','scripts/replay_candidate_utility_v30.py',
        'weekend_thinking/__init__.py'}
    for name in sorted(source_names):
        dest=e/'host-sources'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(root/name,dest)
    write(e/'projection-receipts.json',projections)
    (e/'README.txt').write_text('All2304 generated attempts and3840 token-score rows are retained. Original source context/target arrays are replaced by their lengths and canonical hashes. Exact mathematics, source-independent masks, numerical results and frozen source are included. Run host-sources/scripts/replay_candidate_utility_v30.py --evidence . --manifest ../release-manifest.json --output ../replay. This replays statistics, not source reconstruction or model execution.\n')
    files={str(f.relative_to(e)):sha(f) for f in sorted(e.rglob('*')) if f.is_file()}
    archive=a.output/'v30-candidate-evidence.tar.gz'
    with tarfile.open(archive,'x:gz',compresslevel=6) as tf:
        for name in files:tf.add(e/name,arcname='v30-evidence/'+name,recursive=False)
    seen=set()
    with tarfile.open(archive,'r:gz') as tf:
        import hashlib
        for member in tf:
            name=member.name.removeprefix('v30-evidence/');assert member.isfile() and name in files and name not in seen
            assert hashlib.sha256(tf.extractfile(member).read()).hexdigest()==files[name];seen.add(name)
    assert seen==set(files)
    with archive.open('rb') as f:os.fsync(f.fileno())
    write(a.output/'release-manifest.json',dict(status='PASS_V30_ARCHIVE_FULL_MEMBER_READBACK_AND_FSYNC',
        archive=archive.name,bytes=archive.stat().st_size,sha256=sha(archive),files=files,
        report_sha256=sha(e/'results/readout.json'),attempts=2304,score_rows=3840,
        limits='Full source windows, targets and model weights are not redistributed. Original local evidence remains unchanged.'))
    print('VERIFIED_FILES',len(files),'ARCHIVE_BYTES',archive.stat().st_size,flush=True)


if __name__=='__main__':main()

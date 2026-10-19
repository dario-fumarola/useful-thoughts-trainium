"""Run unchanged v9 equation eligibility on the separately identified new source."""
from pathlib import Path
from datetime import datetime,timezone
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from itertools import islice
import hashlib
import json
import multiprocessing
import os
import time
from train import computable_equations_raw as curator

R=Path('/home/ubuntu/dario');W=R/'local/worktrees/real-prefix-partial-v9'
assert Path(curator.__file__).resolve()==W/'train/computable_equations_raw.py'
D=R/'local/nvme-study/data/real-source-balanced-v17-fresh-source-v1'
O=R/'local/nvme-study/data/real-source-balanced-v17-fresh-curation-v1'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text())


def initialize(model_dir,source_sha):
    os.environ['TOKENIZERS_PARALLELISM']='false'
    curator.initialize(model_dir)
    # The legacy callable's constant is used only for source provenance in
    # base.curate. Bind it to the new raw file, retaining the original function,
    # parser,bounds,selection hash and eligibility checks byte-for-byte.
    curator.RAW_SHA=source_sha


def worker(task):
    meta,line,source=task
    return curator.curate_document(meta,line,curator.TOKENIZER,source,minimum_binary_ops=2)


if __name__=='__main__':
    fixed=read(D/'filtered-complete.json');raw=Path(fixed['raw_file']);members=Path(fixed['membership_file'])
    assert sha(raw)==fixed['raw_sha256'] and sha(members)==fixed['membership_sha256']
    old=read(R/'local/nvme-study/data/finemath-raw-compound-prefix-v9/manifest.json')
    assert curator.sources()==old['source_files']
    O.mkdir(exist_ok=False);deadline=datetime.fromisoformat('2026-10-19T12:30:00+00:00').timestamp()
    plan=dict(status='FROZEN_FRESH_SOURCE_CURATION_NO_MODEL_RELEASE',utc=datetime.now(timezone.utc).isoformat(),
        raw_sha256=fixed['raw_sha256'],membership_sha256=fixed['membership_sha256'],
        source_files=curator.sources(),source_script_sha256=sha(__file__),minimum_binary_ops=2,
        workers=4,deadline_utc='2026-10-19T12:30:00+00:00',
        bounds=dict(raw_characters=curator.MAX_CHARACTERS,native_tokens=curator.MAX_TOKENS,candidate_cut=curator.MAX_CUT),
        provenance_binding='Per-worker RAW_SHA is rebound to this actual new raw file solely for emitted source provenance. Eligibility/selection source bytes are unchanged.',
        limits='Source TRAIN-derived confirmation preparation;no old TEST/unlisted text or model outcomes used. Fixed held selection is a later step.')
    (O/'plan.json').write_text(json.dumps(plan,indent=2))
    count=Counter();candidates=Counter();processed=0;accepted=0;started=time.monotonic()
    def tasks():
        with raw.open('rb') as source,members.open() as membership:
            for lineno,(line,meta) in enumerate(zip(source,membership,strict=True),1):
                m=json.loads(meta);assert m['source_line']==lineno and m['split']=='dev'
                yield m,line,str(raw)
    with ((O/'accepted.jsonl').open('x') as accept,(O/'audit.jsonl').open('x') as audit,
          ProcessPoolExecutor(max_workers=4,mp_context=multiprocessing.get_context('spawn'),
              initializer=initialize,initargs=(str(R/'local/assets/qwen3-1.7b-base'),fixed['raw_sha256'])) as pool):
        iterator=tasks()
        while True:
            assert time.time()<deadline,'Finite data curation deadline;partials preserved.'
            batch=list(islice(iterator,512))
            if not batch:break
            for row,entry,stats in pool.map(worker,batch,chunksize=8):
                processed+=1;count[entry['status']]+=1;candidates.update(stats)
                if row is not None:accepted+=1;accept.write(json.dumps(row,separators=(',',':'))+'\n')
                audit.write(json.dumps(entry,separators=(',',':'))+'\n')
            accept.flush();audit.flush()
            print(json.dumps(dict(documents=processed,accepted=accepted,elapsed_seconds=round(time.monotonic()-started,2))),flush=True)
    assert processed==fixed['counts']['novel_rows']
    assert sha(raw)==fixed['raw_sha256'] and sha(members)==fixed['membership_sha256']
    assert curator.sources()==plan['source_files']
    result=dict(status='COMPLETE_FRESH_COMPOUND_CURATION_NO_COHORT_OR_MODEL_RELEASE',utc=datetime.now(timezone.utc).isoformat(),
        documents=processed,accepted=accepted,counts=dict(count),candidate_counts=dict(candidates),
        accepted_sha256=sha(O/'accepted.jsonl'),audit_sha256=sha(O/'audit.jsonl'),
        plan_sha256=sha(O/'plan.json'),elapsed_seconds=time.monotonic()-started,
        limits='Same exact source-equation filter and prefix-computable interpretation asv9;future-aware RHS eligibility is declared. No heldmodel execution.')
    (O/'completion.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2),flush=True)

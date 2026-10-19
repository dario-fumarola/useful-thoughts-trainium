"""Fetch one predetermined public TRAIN shard at the existing pinned revision."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib
import json
import os
import shutil
import urllib.request

R=Path('/home/ubuntu/dario')
import argparse
parser=argparse.ArgumentParser();parser.add_argument('--shard',type=int,choices=[62,61,60],required=True);args=parser.parse_args()
D=R/f'local/nvme-study/data/real-source-balanced-v17-fresh-source-shard{args.shard:02d}-v1'
D.mkdir(exist_ok=False)
revision='e92b25a616738fe95dc186b64dfb19f9c8525594'
filename=f'finemath-4plus/train-{args.shard:05d}-of-00064.parquet'
base='https://huggingface.co/datasets/HuggingFaceTB/finemath'
plan=dict(status='PREDECLARED_PUBLIC_TRAIN_SOURCE_DOWNLOAD_ONLY',utc=datetime.now(timezone.utc).isoformat(),
    dataset='HuggingFaceTB/finemath',config='finemath-4plus',split='train',revision=revision,
    filename=filename,selection='Declared descending shard extension;no model-outcome selection.',
    license='ODC-By-1.0;Common Crawl terms also apply',
    limitations='No cohort or model release. Exclude prior URLs/document hashes/ordered arithmetic ASTs before confirmation selection. Old TEST/unlisted text remains opaque.')
(D/'download-plan.json').write_text(json.dumps(plan,indent=2))
api=f'https://huggingface.co/api/datasets/HuggingFaceTB/finemath/tree/{revision}/finemath-4plus?recursive=false&limit=1000'
with urllib.request.urlopen(api,timeout=60) as response:entries=json.load(response)
entry=next(v for v in entries if v['path']==filename)
size=entry['size'];assert size<512*2**20 and shutil.disk_usage(D).free>35*2**30+size
(D/'public-file-metadata.json').write_text(json.dumps(entry,indent=2))
target=D/Path(filename).name;temporary=target.with_suffix('.download-part')
url=f'{base}/resolve/{revision}/{filename}'
digest=hashlib.sha256();written=0
try:
    with urllib.request.urlopen(url,timeout=180) as response,temporary.open('xb') as stream:
        while block:=response.read(8*2**20):stream.write(block);digest.update(block);written+=len(block)
        stream.flush();os.fsync(stream.fileno())
except Exception as error:
    raise RuntimeError(f'Public shard download failed ({type(error).__name__});partial retained,no automatic retry.') from None
assert written==size
expected=entry.get('lfs',{}).get('oid')
if expected is not None:assert digest.hexdigest()==expected
temporary.rename(target)
result=dict(status='PINNED_PUBLIC_TRAIN_SHARD_DOWNLOADED_SIZE_AND_SHA_VERIFIED',utc=datetime.now(timezone.utc).isoformat(),
    path=str(target),bytes=written,sha256=digest.hexdigest(),metadata_lfs_sha256=expected,
    revision=revision,filename=filename,limits='Source bytes only;no rows selected,no models evaluated,old TEST untouched.')
(D/'download-complete.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2),flush=True)

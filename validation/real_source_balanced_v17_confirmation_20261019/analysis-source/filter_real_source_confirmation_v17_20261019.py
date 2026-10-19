"""Exclude prior retained pages/content using metadata;old source text stays opaque."""
from collections import Counter
from pathlib import Path
from datetime import datetime,timezone
import hashlib
import json
from urllib.parse import urlsplit,unquote
import pyarrow as pa
import pyarrow.parquet as pq

R=Path('/home/ubuntu/dario');D=R/'local/nvme-study/data/real-source-balanced-v17-fresh-source-v1'
read=lambda p:json.loads(Path(p).read_text())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
source=read(D/'download-complete.json');parquet=Path(source['path']);assert sha(parquet)==source['sha256']
def page(url):
    parsed=urlsplit(url)
    return parsed.netloc.casefold().removeprefix('www.')+unquote(parsed.path).rstrip('/').casefold()
def docid(text):return hashlib.sha256(' '.join(text.split()).encode()).hexdigest()
metadata_files=sorted((R/'local/data/finemath').glob('*.documents.jsonl'))
old_pages,old_hashes=set(),set();pins={};metadata_rows=0
for file in metadata_files:
    pins[str(file)]=sha(file)
    for line in file.open():
        row=json.loads(line)
        assert not {'text','prefix','future','answer'}&set(row)
        metadata_rows+=1;old_pages.add(page(row['url']))
        old_hashes.update(row[k] for k in ['document_id','content_sha256'] if row.get(k))
plan=dict(status='FROZEN_METADATA_EXCLUSION_PLAN_NO_MODEL_OUTCOMES',utc=datetime.now(timezone.utc).isoformat(),
    source_sha256=source['sha256'],source_file=source['filename'],metadata_files=pins,
    metadata_rows=metadata_rows,prior_page_keys=len(old_pages),prior_content_keys=len(old_hashes),
    page_normalization='Ignore scheme,www,query,fragment,trailing slash;casefold host/path and URL-decode path. Conservative exclusion.',
    content_normalization='SHA256 of UTF8 text after whitespace collapse,matching upstream document identity.',
    limits='Prior identifiers from metadata only,including reserved-split identifiers solely for exclusion. Old TEST/unlisted raw text and labels are not opened. Metadata coverage is retained audited documents,not every rejected raw line.')
(D/'exclusion-plan.json').write_text(json.dumps(plan,indent=2))
counts=Counter();seen=set();raw=D/'novel-TRAIN-source.jsonl';members=D/'novel-membership.jsonl'
with raw.open('x') as output,members.open('x') as membership:
    position=0;kept=0
    for batch in pq.ParquetFile(parquet).iter_batches(batch_size=1024,columns=['url','text']):
        urls=batch.column(0).to_pylist()
        eligible=[i for i,u in enumerate(urls) if page(u) not in old_pages]
        counts['source_rows']+=len(urls);counts['prior_page_excluded']+=len(urls)-len(eligible)
        # Filter before decoding any text to Python. Known prior pages only
        # expose URL metadata; their UTF8 text buffers are never inspected.
        selected=batch.take(pa.array(eligible,type=pa.int64())).to_pylist()
        for row,index in zip(selected,eligible):
            text=row['text'];identifier=docid(text)
            if identifier in old_hashes:counts['prior_content_excluded']+=1;continue
            if identifier in seen:counts['within_new_shard_duplicate']+=1;continue
            seen.add(identifier);kept+=1;parquet_row=position+index
            value=dict(text=text,url=row['url'],source=dict(dataset='HuggingFaceTB/finemath',
                revision=source['revision'],file=source['filename'],parquet_row=parquet_row))
            output.write(json.dumps(value,separators=(',',':'),ensure_ascii=False)+'\n')
            meta=dict(source_line=kept,document_id=identifier,url=row['url'],content_sha256=identifier,
                record_id=f'finemath-v17-confirmation:{parquet_row}',split='dev',parquet_row=parquet_row)
            membership.write(json.dumps(meta,separators=(',',':'))+'\n');counts['novel_rows']+=1
        position+=len(urls)
assert counts['source_rows']==pq.ParquetFile(parquet).metadata.num_rows
for file,digest in pins.items():assert sha(file)==digest
result=dict(status='FRESH_PUBLIC_TRAIN_SOURCE_FILTERED_NO_COHORT_OR_MODEL_RELEASE',
    utc=datetime.now(timezone.utc).isoformat(),counts=dict(counts),
    raw_file=str(raw),raw_sha256=sha(raw),membership_file=str(members),membership_sha256=sha(members),
    exclusion_plan_sha256=sha(D/'exclusion-plan.json'),script_sha256=sha(__file__),
    limits=['No source/model outcome selection. One deterministic public TRAIN shard;no officialTEST or old quarantined text opened.',
            'Prior retained page/content overlap is excluded;unknown semantic near duplicates and pretraining exposure remain possible.',
            'All candidate rows are source-TRAIN-derived development;arithmetic eligibility and fixed held selection still pending.'])
(D/'filtered-complete.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2),flush=True)

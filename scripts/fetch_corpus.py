#!/usr/bin/env python3
"""Explicit bounded FineMath streaming download; no tokenization or training.

This is a buffered-shuffle slice, not an exactly uniform sample of the corpus.
Licensing and benchmark/near-duplicate audits remain separate required work.
"""
from pathlib import Path
import argparse,json,os,re,sys,tempfile
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from weekend_thinking.artifacts import atomic_json,sha256_file,utc_now


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--max-documents',type=int,required=True)
    p.add_argument('--revision',default='main')
    p.add_argument('--seed',type=int,default=1234)
    p.add_argument('--shuffle-buffer',type=int,default=10000)
    a=p.parse_args()
    if a.max_documents<1 or a.shuffle_buffer<1:raise SystemExit('Positive limits required.')
    if a.out.exists() or a.out.with_suffix('.source.json').exists():raise SystemExit('Refusing to overwrite an existing corpus.')
    try:
        from huggingface_hub import HfApi
        from datasets import load_dataset
    except ImportError:
        raise SystemExit('datasets/huggingface_hub missing. Use a separate helper environment, not an unpinned Neuron upgrade.')
    repo='HuggingFaceTB/finemath';commit=HfApi().dataset_info(repo,revision=a.revision).sha
    if not commit or not re.fullmatch(r'[0-9a-f]{40,64}',commit):raise SystemExit('Immutable revision unavailable.')
    stream=load_dataset(repo,'finemath-4plus',split='train',streaming=True,revision=commit)
    stream=stream.shuffle(seed=a.seed,buffer_size=a.shuffle_buffer)
    a.out.parent.mkdir(parents=True,exist_ok=True)
    fd,tmp=tempfile.mkstemp(prefix='.'+a.out.name+'.',dir=a.out.parent)
    count=0
    try:
        with os.fdopen(fd,'w') as f:
            for row in stream:
                text=row.get('text')
                if not isinstance(text,str) or not text.strip():continue
                obj={'text':text,'url':row.get('url') or '', 'domain':'mathematics',
                     'source_metadata':{k:v for k,v in row.items() if k not in {'text','url'}}}
                f.write(json.dumps(obj,ensure_ascii=False)+'\n');count+=1
                if count>=a.max_documents:break
            f.flush();os.fsync(f.fileno())
        if count==0:raise RuntimeError('No usable documents returned.')
        os.link(tmp,a.out)
    finally:Path(tmp).unlink(missing_ok=True)
    atomic_json(a.out.with_suffix('.source.json'),{'dataset_id':repo,'config':'finemath-4plus',
        'resolved_revision':commit,'created_utc':utc_now(),'documents':count,
        'requested_max_documents':a.max_documents,'seed':a.seed,'shuffle_buffer':a.shuffle_buffer,
        'sha256':sha256_file(a.out),'data_audit_passed':False,
        'license':'ODC-By-1.0; Common Crawl Terms of Use also apply',
        'source_card':f'https://huggingface.co/datasets/{repo}/blob/{commit}/README.md',
        'limitations':['Buffered-shuffle stream slice, not an exact uniform sample.',
                       'No benchmark-overlap or near-duplicate filter performed here.',
                       'Too-short documents will reduce usable training examples after tokenization.']})
    print(json.dumps({'documents':count,'resolved_revision':commit,'out':str(a.out)},indent=2))
if __name__=='__main__':main()

#!/usr/bin/env python3
"""Conservative exact normalized word n-gram overlap audit.

Writes a filtered corpus and aggregate report without printing benchmark text.
Not a full decontamination guarantee or near-duplicate algorithm. The benchmark
question set used here is for exclusion only, never for hyperparameter tuning.
"""
from pathlib import Path
import argparse,hashlib,json,re,sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from weekend_thinking.artifacts import atomic_bytes,atomic_json,sha256_file


def shingles(text,n):
    words=re.findall(r'\w+',text.lower())
    return {hashlib.sha256(' '.join(words[i:i+n]).encode()).digest() for i in range(len(words)-n+1)}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--documents',type=Path,required=True)
    p.add_argument('--benchmark-questions',type=Path,required=True,help='JSONL with question fields only; do not use as training data.')
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--n',type=int,default=13)
    a=p.parse_args()
    if a.n<3:raise SystemExit('Use n>=3; default13 is a screening choice, not a proof.')
    if a.out.exists():raise SystemExit('Refusing overwrite.')
    needles=set();questions=0
    with a.benchmark_questions.open() as f:
        for line in f:
            if not line.strip():continue
            row=json.loads(line);needles.update(shingles(row['question'],a.n));questions+=1
    if not needles:raise SystemExit('No benchmark shingles; do not call an empty audit a pass.')
    # Stream output: large corpora must not accumulate in memory.
    tmp=a.out.with_name('.'+a.out.name+'.incomplete');a.out.parent.mkdir(parents=True,exist_ok=True)
    kept=removed=0
    try:
        with a.documents.open() as src,tmp.open('x') as dst:
            for line in src:
                if not line.strip():continue
                row=json.loads(line)
                if needles.intersection(shingles(row['text'],a.n)):removed+=1
                else:dst.write(json.dumps(row,ensure_ascii=False)+'\n');kept+=1
        import os
        os.link(tmp,a.out)
    finally:tmp.unlink(missing_ok=True)
    report={'source_sha256':sha256_file(a.documents),'benchmark_sha256':sha256_file(a.benchmark_questions),
            'output_sha256':sha256_file(a.out),'n':a.n,'benchmark_questions':questions,
            'kept_documents':kept,'excluded_documents':removed,
            'limitations':['Exact normalized word overlap only; paraphrases and many near-duplicates are not detected.',
                           'May exclude generic shared phrasing. This filter changes the selected data distribution.',
                           'Does not remove contamination in pretrained weights.']}
    atomic_json(a.out.with_suffix('.overlap.json'),report)
    print(json.dumps(report,indent=2))
if __name__=='__main__':main()

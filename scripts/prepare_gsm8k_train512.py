"""Extend the existing question-only GSM8K TRAIN hash order, without models.

The first128 questions remain exactly the earlier development panel. Their
outcomes are already known; this512 panel is development, never official TEST.
Gold is opened only after question-only membership is published.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import unicodedata

from weekend_thinking.artifacts import atomic_json, sha256_file, utc_now

ROOT=Path(__file__).resolve().parents[1]
TRAIN=ROOT/'local/data/benchmark_exclusion/sources/openai--gsm8k/main/train-00000-of-00001.parquet'
TRAIN_SHA='ea82612ea9582142387730c793eb67d3b12849002bc0b7fa6f8efafa7351419d'
PARENT=ROOT/'local/evaluation/held-dev128-v1'
PARENT_SELECTION_SHA='044bf30b847cd3294f30d8622b358e891563be9081787a61b24221b68d27780f'
REVISION='740312add88f781978c0658806c59bc2815b9866'


def parent_hash(value):
    """The original publication's exact JSON escaping convention."""
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),
                                     allow_nan=False).encode()).hexdigest()


def normalized(question):
    return ' '.join(unicodedata.normalize('NFKC',question).casefold().split())


def select_questions(questions,parent,count=512):
    all_rows=[dict(item_id=f'gsm8k:train:{i:05d}',question=q) for i,q in enumerate(questions)]
    if any(not isinstance(q,str) or not q.strip() for q in questions):
        raise ValueError('Every canonical TRAIN question must be nonempty')
    byid={r['item_id']:r for r in all_rows}
    excluded=set(parent['excluded_ids'])
    if not excluded<=set(byid):raise ValueError('Parent exclusions not canonical TRAIN IDs')
    norms={normalized(byid[key]['question']) for key in excluded}
    if parent_hash(sorted(norms))!=parent['excluded_question_normal_forms_sha256']:
        raise ValueError('Parent excluded question identities changed')
    eligible=[r for r in all_rows if r['item_id'] not in excluded and normalized(r['question']) not in norms]
    eligible.sort(key=lambda r:(hashlib.sha256((parent['salt']+':'+r['item_id']).encode()).hexdigest(),r['item_id']))
    chosen=[];seen=set()
    for row in eligible:
        norm=normalized(row['question'])
        if norm not in seen:
            chosen.append(row);seen.add(norm)
            if len(chosen)==count:break
    if len(chosen)!=count:raise ValueError('Full question quota unavailable; never shrink it')
    if [r['item_id'] for r in chosen[:len(parent['ordered_item_ids'])]]!=parent['ordered_item_ids']:
        raise ValueError('Extension changed the existing ordered question panel')
    return chosen,len(eligible)


def prepare(output):
    import pyarrow.parquet as pq
    output=Path(output)
    if output.exists():raise FileExistsError(output)
    if TRAIN.name!='train-00000-of-00001.parquet' or sha256_file(TRAIN)!=TRAIN_SHA:
        raise ValueError('Only the exact pinned TRAIN Parquet may be opened')
    parent_file=PARENT/'selection.json'
    if sha256_file(parent_file)!=PARENT_SELECTION_SHA:raise ValueError('Original panel changed')
    parent=json.loads(parent_file.read_text())
    source=pq.read_table(TRAIN,columns=['question'])['question'].to_pylist()
    if len(source)!=7473:raise ValueError('Canonical TRAIN source count changed')
    selected,eligible=select_questions(source,parent)
    original=[json.loads(s) for s in (PARENT/'questions.jsonl').read_text().splitlines() if s.strip()]
    if [dict(item_id=r['item_id'],question=r['question']) for r in original]!=selected[:128]:
        raise ValueError('Original128 question projection changed')
    prompt=PARENT/'prompt.json';fmt=ROOT/'local/resolved_format.json'
    input_pins={str(p):sha256_file(p) for p in (TRAIN,parent_file,PARENT/'questions.jsonl',prompt,fmt)}
    output.mkdir(parents=True)
    selection=dict(schema='gsm8k-train512-question-selection-v1',created_utc=utc_now(),
        dataset='openai/gsm8k',revision=REVISION,split='train',source_sha256=TRAIN_SHA,
        parent_selection_sha256=PARENT_SELECTION_SHA,salt=parent['salt'],
        rule='Extend the original question-only hash order to512 eligible unique normalized questions; first128 unchanged.',
        selection_uses_answers=False,items=512,eligible=eligible,
        ordered_item_ids=[r['item_id'] for r in selected],
        ordered_item_ids_sha256=parent_hash([r['item_id'] for r in selected]))
    atomic_json(output/'selection.json',selection)
    with (output/'questions.jsonl').open('x') as stream:
        for row in selected:stream.write(json.dumps(dict(row,document_id=row['item_id'],task='gsm8k',split='train'),ensure_ascii=False)+'\n')
    # Only now load TRAIN gold. No model or generation code receives this file.
    answers=pq.read_table(TRAIN,columns=['answer'])['answer'].to_pylist()
    if len(answers)!=len(source):raise ValueError('TRAIN answer alignment changed')
    with (output/'gold.jsonl').open('x') as stream:
        for row in selected:
            answer=answers[int(row['item_id'].rsplit(':',1)[1])]
            if not isinstance(answer,str) or not answer.strip():raise ValueError('Missing fixed gold; never replace selected items')
            stream.write(json.dumps(dict(item_id=row['item_id'],task='gsm8k',split='train',answer=answer),ensure_ascii=False)+'\n')
    (output/'prompt.json').write_bytes(prompt.read_bytes())
    format_data=json.loads(fmt.read_text());format_data['header_text']='';format_data['format']['header_ids']=[]
    format_data['purpose']='New development interface: no instruction header; native think tags; existing GSM full-question reader suffix.'
    atomic_json(output/'format.json',format_data)
    if any(sha256_file(p)!=digest for p,digest in input_pins.items()):raise ValueError('Preparation inputs changed')
    questions=[json.loads(s) for s in (output/'questions.jsonl').read_text().splitlines()]
    gold=[json.loads(s) for s in (output/'gold.jsonl').read_text().splitlines()]
    assert len(questions)==len(gold)==512
    assert all(not {'answer','solution','target','gold'}&set(row) for row in questions)
    assert [r['item_id'] for r in questions]==[r['item_id'] for r in gold]==selection['ordered_item_ids']
    result=dict(schema='gsm8k-train512-development-v1',status='COMPLETE_CPU_PREPARATION_NOT_MODEL_EVALUATION',
        created_utc=utc_now(),selection=selection,input_hashes=input_pins,
        output_hashes={name:sha256_file(output/name) for name in ('selection.json','questions.jsonl','gold.jsonl','prompt.json','format.json')},
        source_sha256=sha256_file(__file__),planned_modes=['direct','empty_format','one_thought'],
        thought_cap=128,max_answer_tokens=512,compute='fp32',head_precision='fp32',
        parser='Existing strict_final_hash4_number_v1; no recovered-number accuracy.',
        unsupported_definition='Report format_error/truncated/error separately and their combined count, retaining all512 in accuracy denominator.',
        planned_models=['B0','C','T','W'],planned_training_seeds=[20261018,20261019],
        model_binding='Actual terminal model/checkpoint/spec/source identities must be supplied later; no invented future checkpoint hash.',
        limitations=['Development TRAIN panel, never official TEST or a clean external benchmark.',
            'First128 already have earlier development outcomes; membership is not selected on those outcomes.',
            'The new no-header GSM projection repeats the whole question using the existing reader suffix; it is not arithmetic-LHS reanchoring.',
            'Likelihood remains primary; strict parsed accuracies are secondary with unsupported counts.',
            'No model was loaded or evaluated; this preparation does not qualify hardware.'])
    atomic_json(output/'manifest.json',result);return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();result=prepare(args.output)
    print(json.dumps(dict(items=result['selection']['items'],manifest_sha256=sha256_file(args.output/'manifest.json'))))

"""Disclosed deterministic examples after the complete independently checked readout."""
from pathlib import Path
import gzip
import hashlib
import json
from tokenizers import Tokenizer
from train.partial_state_evaluation import donor_permutation

R=Path('/home/ubuntu/dario');p=R/'validation/real_target_sft_v15_complete_20261019.json'
r=json.loads(p.read_text());out=p.with_suffix('');sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert json.loads((out/'independent-readout-check.json').read_text())['report_sha256']==sha(p)
def payload(name):
    f=out/name;assert sha(f)==r['payloads'][str(f.relative_to(R))]['sha256']
    return [json.loads(v) for v in gzip.decompress(f.read_bytes()).splitlines()]
meta=payload('fixed-source-target-metadata.jsonl.gz');order=[v['item_id'] for v in meta];annotations={v['item_id']:v for v in meta}
questions=[json.loads(v) for v in (R/'local/nvme-cpt/data/real-target-sft-v15-evaluation-inputs-v1/DEV_fresh384-questions.jsonl').read_text().splitlines()]
assert [v['item_id'] for v in questions]==order
prefixes={v['item_id']:v['prefix_ids'] for v in questions}
tok=Tokenizer.from_file(str(R/'local/assets/qwen3-1.7b-base/tokenizer.json'))
decode=lambda ids:tok.decode(ids,skip_special_tokens=False)
answers,notes,LL,donors={},{},{},{}
for m in r['tables']:
    values=payload(f'{m}-all1920-scalar-continuations.jsonl.gz');answers[m]={(v['item_id'],v['mode']):v for v in values}
    values=payload(f'{m}-all384-original-notes.jsonl.gz');assert [v['item_id'] for v in values]==order
    notes[m]={v['item_id']:v for v in values}
    perm=donor_permutation([len(v['content_ids']) for v in values],order)
    donors[m]={k:values[d] for k,d in zip(order,perm)}
    values=payload(f'{m}-DEV_fresh384-all1920-logps.jsonl.gz');LL[m]={(v['item_id'],v['mode']):v for v in values}
examples=[]
for seed in r['seeds']:
    direct,cot=[f'{a}-seed{seed}-step320' for a in ['target_direct','target_cot']]
    chosen={k:['first_three_in_frozen_order'] for k in order[:3]}
    def correct(m,k,mode):return answers[m][k,mode]['correct']
    for label,predicate in [
        ('first_CoT_own_correct_direct_raw_wrong',lambda k:correct(cot,k,'own') and not correct(direct,k,'raw')),
        ('first_CoT_own_wrong_direct_raw_correct',lambda k:not correct(cot,k,'own') and correct(direct,k,'raw')),
        ('first_CoT_own_wrong_oracle_correct',lambda k:not correct(cot,k,'own') and correct(cot,k,'oracle'))]:
        found=next((k for k in order if predicate(k)),None)
        if found is not None:chosen.setdefault(found,[]).append(label)
    delta={k:LL[cot][k,'own']['span16_logprob']-LL[direct][k,'raw']['span16_logprob'] for k in order}
    for label,k in [('maximum_secondary_full16_contrast',max(order,key=delta.get)),('minimum_secondary_full16_contrast',min(order,key=delta.get))]:
        chosen.setdefault(k,[]).append(label)
    for k in order:
        if k not in chosen:continue
        a=annotations[k]
        models={}
        for m in ['B0',direct,cot,f'real_raw-seed{seed}-step320',f'real_prior-seed{seed}-step320']:
            models[m]=dict(original_note=notes[m][k],unrelated_donor_note=donors[m][k],
                continuations={mode:answers[m][k,mode] for mode in ['raw','empty','own','other','oracle']},
                full16_logprobs={mode:LL[m][k,mode]['span16_logprob'] for mode in ['raw','empty','own','other','oracle']})
        examples.append(dict(seed=seed,item_id=k,document_id=a['document_id'],selected_by=chosen[k],
            prefix_native_ids=prefixes[k],prefix_text=decode(prefixes[k]),future16_native_ids=a['target16'],future16_text=decode(a['target16']),
            expected_numeric_value=a['expected_numeric_value'],normalized_expression=a['normalized_expression'],
            supplied_teacher_trace=a['teacher_full_calculation'],models=models))
f=out/'disclosed-examples.jsonl.gz';f.write_bytes(gzip.compress(''.join(json.dumps(v)+'\n' for v in examples).encode(),mtime=0))
(out/'example-selection.json').write_text(json.dumps(dict(examples=len(examples),payload_sha256=sha(f),source_sha256=sha(__file__),
    rule='Per seed:first3 frozen order;first in each of3 directional answer discordances;max/min secondary full16 contrast. Ties frozen order;deduplicate within seed. All selection reasons disclosed.',
    limits='Illustrations selected after all fixed outcomes close;not an endpoint or independent evidence. Gold/teacher traces are shown for inspection only;never writer inputs.'),indent=2))
print('Saved',len(examples),'disclosed examples with original native outputs.',flush=True)

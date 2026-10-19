"""CPU feasibility of held-support exclusion;no cohort/model release or answers."""
from collections import Counter
from datetime import datetime,timezone
from pathlib import Path
import hashlib
import json
from tokenizers import Tokenizer
from train.document_evidence_data import question_projection,render

R=Path('/home/ubuntu/dario')
source=R/'local/nvme-study/data/musique-TRAIN-source-v12/musique_ans_train.raw.jsonl'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert sha(source)=='a85aeee64718f08ab9d11472f28b424e3d933794082b6d3c5b2481cd7e733345'
tok=Tokenizer.from_file(str(R/'local/assets/qwen3-1.7b-base/tokenizer.json'))
norm=lambda s:' '.join(s.casefold().split())
fingerprint=lambda s:hashlib.sha256(norm(s).encode()).hexdigest()
records=[]
for line in source.open():
    row=json.loads(line);q=question_projection(row)
    text=render(q)
    visible={fingerprint(c['paragraph_text']) for c in row['contexts']}
    supporting={fingerprint(c['paragraph_text']) for c in row['contexts'] if c['is_supporting']}
    support_titles={norm(c['wikipedia_title']) for c in row['contexts'] if c['is_supporting']}
    assert supporting and supporting<=visible
    records.append(dict(id=q['item_id'],question=norm(q['question']),
        tokens=len(tok.encode(text,add_special_tokens=False).ids),visible=visible,
        visible_titles={norm(c['wikipedia_title']) for c in row['contexts']},
        support=supporting,support_titles=support_titles,
        components={norm(s) for s in row['component_question_texts']},hops=len(row['decomposed_instances']),
        rank=hashlib.sha256(('document-evidence-v16-audit-only:'+q['item_id']).encode()).hexdigest()))
assert len(records)==len({v['id'] for v in records})==19938
results={}
for budget in [2048,3072]:
    eligible=sorted((v for v in records if v['tokens']<=budget),key=lambda v:(v['rank'],v['id']))
    held=[];seen=set()
    for row in eligible:
        if row['question'] not in seen:
            held.append(row);seen.add(row['question'])
        if len(held)==512:break
    assert len(held)==512
    union=lambda field:set().union(*(v[field] for v in held))
    held_support,held_titles,held_components,held_visible=[union(f) for f in ['support','support_titles','components','visible']]
    held_ids={v['id'] for v in held}
    candidates=[v for v in eligible if v['id'] not in held_ids and v['question'] not in seen]
    paragraph_disjoint=[v for v in candidates if not v['visible']&held_support]
    title_disjoint=[v for v in paragraph_disjoint if not v['visible_titles']&held_titles]
    component_disjoint=[v for v in title_disjoint if not v['components']&held_components]
    wholly_visible_disjoint=[v for v in candidates if not v['visible']&held_visible]
    counts={}
    for label,training in [('exclude_held_support_text',paragraph_disjoint),
                           ('also_exclude_held_support_title',title_disjoint),
                           ('also_exclude_held_component_question',component_disjoint),
                           ('exclude_all_held_visible_text',wholly_visible_disjoint)]:
        train_visible=set().union(*(v['visible'] for v in training)) if training else set()
        assert not train_visible&held_support
        counts[label]=dict(training_questions=len(training),unique_question_texts=len({v['question'] for v in training}),
            train_hops=dict(Counter(v['hops'] for v in training)),
            held_questions_with_support_seen_in_train_visible=sum(bool(v['support']&train_visible) for v in held),
            held_questions_with_any_distractor_seen_in_train_visible=sum(bool((v['visible']-v['support'])&train_visible) for v in held),
            training_ids_sha256=hashlib.sha256(('\n'.join(v['id'] for v in training)+'\n').encode()).hexdigest())
    results[str(budget)]=dict(eligible=len(eligible),held_questions=512,
        held_hops=dict(Counter(v['hops'] for v in held)),
        held_ids_sha256=hashlib.sha256(('\n'.join(v['id'] for v in held)+'\n').encode()).hexdigest(),
        candidate_training_questions_after_exact_question_exclusion=len(candidates),policies=counts)
report=dict(status='CPU_DIRECTED_SPLIT_FEASIBILITY_NO_COHORT_MODEL_OR_EVALUATION_RELEASE',
    utc=datetime.now(timezone.utc).isoformat(),source_sha256=sha(source),rows=len(records),
    selection='For each declared complete-input budget,first512distinct normalizedquestions by fixed salted ID hash;no model outcomes or answers. All passages retained.',
    complete_input_token_budgets=results,source_sha256_script=sha(__file__),
    projection_source_sha256=sha(R/'train/document_evidence_data.py'),
    references=['https://arxiv.org/html/2108.00573v2#S5.SS5','https://github.com/StonyBrookNLP/musique'],
    limits=['AuthorsTRAINonly;officialDEV/TESTunopened. The two length budgets are separate feasibility tables,not aselectedcohort.',
            'Offline supporting flags andcomponentquestion labels areused onlytosplit/audit;never policyinputs. Answersarenotread.',
            'Directed exclusion guarantees heldsupport textabsence fromtraining-visibleparagraphs;itdoesnotmakeall visibleparagraphsdisjoint.',
            'Titleexclusionisconservative,butdoesnotprove semantic/evidence independence orpretrainingnovelty. Normalizationcasefolds/collapseswhitespace.',
            'No proposed method,training budget,hardwarequalification ormodelrun isreleased. Choose scientificdirection onlyaftercompleteV15readout.'])
out=R/'local/analysis/document-evidence-directed-split-v16-20261019.json'
with out.open('x') as stream:json.dump(report,stream,indent=2)
print(json.dumps(report,indent=2),flush=True)

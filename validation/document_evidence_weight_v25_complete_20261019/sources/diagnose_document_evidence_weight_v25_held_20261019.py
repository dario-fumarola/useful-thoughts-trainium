"""Fixed descriptive support-count and benefit/harm breakdown; no new gate."""
from collections import Counter
from pathlib import Path
import json
import re
R=Path('/home/ubuntu/dario');C=R/'local/diagnostics/document-evidence-weight-v25-ablation-v1'
D=R/'local/nvme-cpt/data/document-evidence-plan-v23-TRAIN2048-held255-v2'
read=lambda p:json.loads(p.read_text())
rows=lambda p:[json.loads(v) for v in p.read_text().splitlines()]
assert read(C/'held-independent-check.json')['status'].startswith('PASS_')
gold={v['item_id']:v for v in rows(D/'held255-gold-only.jsonl')}
scored=read(C/'held-native-and-replay-scored.json');native=scored['native:selection']
assert set(native)==set(gold) and len(gold)==255
questions={v['item_id']:v for v in rows(D/'held255-questions-only.jsonl')}
# Reconstruct selected sets from the original text rather than trusting a score dict.
for key,value in native.items():
    text=value['native']['raw_text'];fields=list(re.finditer(r'(?im)^[ \t]*Final answer:[ \t]*',text))
    chosen=set()
    if len(fields)==1:
        match=re.fullmatch(r'\s*Evidence:[ \t]*((?:\[[0-9]+\][ \t]*)*)\r?\n\s*',text[:fields[0].start()],re.I)
        if match:
            ids=list(map(int,re.findall(r'\[([0-9]+)\]',match[1])))
            if ids and len(ids)==len(set(ids)) and all(1<=i<=len(questions[key]['contexts']) for i in ids):chosen=set(ids)
    target=set(gold[key]['support_indices']);common=len(chosen&target)
    precision=common/len(chosen) if chosen else 0.;recall=common/len(target)
    expected=dict(precision=precision,recall=recall,exact_match=int(chosen==target),f1=2*precision*recall/(precision+recall) if precision+recall else 0.)
    assert all(abs(value['annotation_overlap'][name]-n)<=1e-12 for name,n in expected.items())
strata={};groups={}
for n in sorted({len(v['support_indices']) for v in gold.values()}):
 ids=[k for k,v in gold.items() if len(v['support_indices'])==n]
 selected=[native[k] for k in ids]
 strata[str(n)]=dict(items=len(ids),exact_sets=sum(v['annotation_overlap']['exact_match'] for v in selected),
    selection_status=dict(Counter(v['selection']['status'] for v in selected)),
    output_ID_counts=dict(Counter(len(v['selection']['indices']) for v in selected)),
    **{metric:sum(v['annotation_overlap'][metric] for v in selected)/len(ids) for metric in ('precision','recall','f1')},
    canonical_answers={mode:sum(values[k]['score']['canonical_em'] for k in ids) for mode,values in scored.items()})
for exact in (False,True):
 ids=[k for k in gold if bool(native[k]['annotation_overlap']['exact_match'])==exact]
 direct=scored['native:direct']
 groups[str(exact)]=dict(items=len(ids),same_model_direct_correct=sum(direct[k]['score']['canonical_em'] for k in ids),
    own_correct=sum(native[k]['score']['canonical_em'] for k in ids),
    oracle_replay_correct=sum(scored['answer-replay:oracle'][k]['score']['canonical_em'] for k in ids),
    corrected=sum(native[k]['score']['canonical_em']>direct[k]['score']['canonical_em'] for k in ids),
    regressed=sum(native[k]['score']['canonical_em']<direct[k]['score']['canonical_em'] for k in ids))
selected=list(native.values())
result=dict(status='COMPLETE_DESCRIPTIVE_V25_SELECTION_AND_ANSWER_BREAKDOWN',
    items=255,strata=strata,exact_sets=sum(v['annotation_overlap']['exact_match'] for v in selected),
    mean_annotation_overlap={metric:sum(v['annotation_overlap'][metric] for v in selected)/255 for metric in ('precision','recall','f1')},
    exact_set_groups_posthoc=groups,
    limits=['Support counts and annotation exactness are descriptive, not new scientific gates.',
        'Subgroups condition on generated selections and question difficulty; not causal comparisons.',
        'Four-support n4 cannot support a depth-generalization claim.',
        'All20 passages remain visible. Imperfect support annotations are not semantic gold.'])
with (C/'held-selection-descriptive.json').open('x') as f:json.dump(result,f,indent=2)
print(json.dumps(result,indent=2))

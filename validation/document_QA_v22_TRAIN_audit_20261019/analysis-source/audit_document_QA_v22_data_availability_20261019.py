"""CPU-only availability and TRAIN grounding sample;no newscience/datarelease.

Use source identities and cached native inputlengths only. No modeloutputs/scores.
Greedy counts are constructive lowerbounds,not optimal independent-set bounds.
"""
from collections import Counter
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,re
from train.document_qa_data import evidence_identity,normalized
R=Path('/home/ubuntu/dario');D=R/'local/diagnostics/document-QA-v22-data-availability-v1';D.mkdir(exist_ok=False)
S=R/'local/nvme-study/data/musique-TRAIN-source-v12/musique_ans_train.raw.jsonl';V=R/'local/nvme-cpt/data/document-QA-v21-TRAIN2048-held512-v1';sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();readrows=lambda p:[json.loads(l) for l in Path(p).read_text().splitlines()]
plan=dict(status='CPU_ONLY_UNRELEASED_DATA_AVAILABILITY_AND_GROUNDING_INSPECTION',utc=datetime.now(timezone.utc).isoformat(),question='Can unchanged TRAIN-only sources support fresh evidence-clean higher-hop confirmation at existing contextwidth,and what traininglabels need semanticgrounding review?',source_sha256=sha(S),old_manifest_sha256=sha(V/'manifest.json'),windows=[2048,3072,4096],filters=['current_train_exclusions','plus_old_held_support','plus_all_old_held_visible','paragraph_only_plus_old_held_support'],within_held=['support_text_page_seed','support_text_seed'],balanced_greedy_quotas={2:256,3:128,4:128},grounding_sample='First32 fixedhash TRAIN examples per2/3/4hops;96examples. Stratified,not representativeunweighted.',limits='No newtrain/heldmodeljob,semanticpass,checkpointchoice oroldcohortmutation.4096 is sizingonly andexceeds currentqualifiedwidth. Repeatedanswers remainallowed andnot officialS5. No modeloutputs areopened.')
assert plan['source_sha256']=='a85aeee64718f08ab9d11472f28b424e3d933794082b6d3c5b2481cd7e733345';(D/'plan.json').write_text(json.dumps(plan,indent=2))
raw={r['id']:r for r in readrows(S)};lengths={r['item_id']:r['native_input_tokens'] for r in readrows(S.parent/'input-inventory-v2.jsonl')};train=readrows(V/'TRAIN2048.jsonl');oldheld=readrows(V/'held512-questions-only.jsonl');assert len(raw)==19938 and len(train)==2048 and len(oldheld)==512
meta={k:dict(evidence_identity(v),hops=len(v['decomposed_instances'])) for k,v in raw.items()}
def union(ids,field):return set().union(*(meta[k][field] for k in ids))
tids={r['item_id'] for r in train};hids={r['item_id'] for r in oldheld};excluded=tids|hids;questions={meta[k]['question'] for k in excluded};T={f:union(tids,f) for f in ('visible_text','visible_titles','seed_ids')};H={f:union(hids,f) for f in ('support_text','support_titles','visible_text','visible_titles','seed_ids')}
ranked=sorted(meta.values(),key=lambda r:(hashlib.sha256(('document-QA-v22-CPU-availability-v1:'+r['item_id']).encode()).hexdigest(),r['item_id']))
def eligible(r,policy):
 if r['item_id'] in excluded or r['question'] in questions or r['support_text']&T['visible_text'] or r['seed_ids']&T['seed_ids']:return False
 if policy!='paragraph_only_plus_old_held_support' and r['support_titles']&T['visible_titles']:return False
 if policy=='current_train_exclusions':return True
 if r['seed_ids']&H['seed_ids']:return False
 if policy=='plus_all_old_held_visible':return not (r['support_text']&H['visible_text'] or r['support_titles']&H['visible_titles'])
 return not (r['support_text']&H['support_text'] or (policy!='paragraph_only_plus_old_held_support' and r['support_titles']&H['support_titles']))
def greedy(candidates,page,balanced=False):
 selected=[];texts=set();titles=set();seeds=set();seen=set();counts=Counter();quotas={2:256,3:128,4:128}
 order=sorted(candidates,key=lambda r:(-r['hops'],ranked_index[r['item_id']])) if balanced else candidates
 for r in order:
  if balanced and counts[r['hops']]>=quotas[r['hops']]:continue
  if r['question'] in seen or r['support_text']&texts or (page and r['support_titles']&titles) or r['seed_ids']&seeds:continue
  selected.append(r['item_id']);counts[r['hops']]+=1;texts.update(r['support_text']);titles.update(r['support_titles']);seeds.update(r['seed_ids']);seen.add(r['question'])
 return selected,dict(counts)
ranked_index={r['item_id']:i for i,r in enumerate(ranked)};conditions=[]
for window in plan['windows']:
 for policy in plan['filters']:
  candidates=[r for r in ranked if lengths[r['item_id']]<=window and eligible(r,policy)]
  for page in (True,False):
   selected,counts=greedy(candidates,page);balanced,bc=greedy(candidates,page,True)
   conditions.append(dict(window=window,policy=policy,within_held='support_text_page_seed' if page else 'support_text_seed',eligible_rows=len(candidates),eligible_hops=dict(Counter(r['hops'] for r in candidates)),greedy_independent_rows=len(selected),greedy_hops=counts,balanced_greedy_rows=len(balanced),balanced_greedy_hops=bc,greedy_ids_sha256=hashlib.sha256(('\n'.join(selected)+'\n').encode()).hexdigest()))
# Review examples are existing TRAIN only;no held policyoutput ornew heldgoldseen.
review=[]
for hops in (2,3,4):
 pool=[r for r in train if meta[r['item_id']]['hops']==hops];pool.sort(key=lambda r:(hashlib.sha256(('document-QA-v22-grounding-TRAIN-audit-v1:'+r['item_id']).encode()).hexdigest(),r['item_id']));assert len(pool)>=32
 for row in pool[:32]:
  parts=re.findall(r'(?m)^Step (\d+): ([^\n]*)\nEvidence \[(\d+)\]: ([^\n]*)',row['trace']);assert len(parts)==hops;previous={};steps=[]
  for n,q,c,a in parts:
   n,c=int(n),int(c);assert n==len(previous)+1;expanded=re.sub(r'#(\d+)',lambda m:previous[int(m[1])],q);previous[n]=a;context=row['question']['contexts'][c-1]
   steps.append(dict(step=n,question_original=q,question_resolved=expanded,canonical_answer=a,citation=c,title=context['title'],paragraph=context['text'],manual_entailment_label=None,manual_note=None))
  review.append(dict(item_id=row['item_id'],hops=hops,question=row['question']['question'],final_answer=row['answer'],steps=steps,source='EXISTING_TRAIN_ONLY',all_visible_contexts=row['question']['contexts']))
assert len(review)==96 and {r['item_id'] for r in review}<=tids and not {r['item_id'] for r in review}&hids
q=D/'TRAIN96-grounding-review.jsonl';q.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in review));rubric=dict(status='UNANNOTATED_MANUAL_REVIEW_TEMPLATE_NOT_A_SEMANTIC_PASS',labels=['entailed_in_cited_paragraph','supported_elsewhere_visible_but_citation_insufficient','insufficient_or_ambiguous_from_visible_context','contradicted_by_visible_context'],instructions='Judge relation between resolvedquestion andanswer,not merely literalpresence. Use citedparagraph first;inspectallvisiblecontext ifinsufficient. Do not useoutsideworldknowledge. Recordtextbasis/ambiguity. Oneagentreview isnot independenthumanannotation. Stratifiedsample cannotyieldunweightedcorpuserrorrate. Do not alterV21labels/scores.',samples=96,per_hops=32)
(D/'manual-review-rubric.json').write_text(json.dumps(rubric,indent=2));report=dict(status='CPU_AVAILABILITY_INVENTORY_COMPLETE_NO_COHORT_OR_TRAINING_RELEASE',utc=datetime.now(timezone.utc).isoformat(),source_rows=19938,raw_hop_counts=dict(Counter(r['hops'] for r in meta.values())),conditions=conditions,review_template=dict(path=str(q),sha256=sha(q),rows=96,status=rubric['status']),plan_sha256=sha(D/'plan.json'),source_sha256=sha(__file__),limits=plan['limits'])
(D/'result.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2),flush=True)

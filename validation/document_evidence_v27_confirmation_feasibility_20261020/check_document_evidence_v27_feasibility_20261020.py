"""Independent aggregate/source-count and power checks; no model or examples."""
from pathlib import Path
from collections import Counter,defaultdict
from statistics import NormalDist
import json,hashlib,math
from transformers import AutoTokenizer
from train.document_evidence_data import render
R=Path('/home/ubuntu/dario');C=R/'local/diagnostics/document-evidence-v27-confirmation-feasibility-v1';D=R/'local/nvme-study/data/musique-DEV-mirror-metadata-v27'
read=lambda p:json.loads(Path(p).read_text());rows=lambda p:[json.loads(s) for s in Path(p).read_text().splitlines()];sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
r=read(C/'DEV-mirror-metadata-result.json');a=read(D/'acquisition.json');assert a['receipts'][0]['sha256']==a['receipts'][1]['sha256']==r['DEV_sha256']
for v in a['receipts']:assert sha(D/v['file'])==v['sha256']
raw=rows(D/a['receipts'][0]['file']);assert len(raw)==r['source_rows']==2417
meta={v['item_id']:v for v in rows(D/'input-metadata-v2.jsonl')};assert len(meta)==len(raw) and sha(D/'input-metadata-v2.jsonl')==r['metadata_inventory_sha256']
norm=lambda s:' '.join(s.casefold().split());fp=lambda s:hashlib.sha256(norm(s).encode()).hexdigest()
S=R/'local/nvme-study/data/musique-TRAIN-source-v12';trainraw=rows(S/'musique_ans_train.raw.jsonl');T=R/'local/nvme-cpt/data/document-evidence-plan-v23-TRAIN2048-held255-v2';V=R/'local/nvme-cpt/data/document-QA-v21-TRAIN2048-held512-v1'
old={v['item_id'] for p in [T/'TRAIN2048.jsonl',V/'held512-questions-only.jsonl',T/'held255-questions-only.jsonl'] for v in rows(p)}
oldraw=[v for v in trainraw if v['id'] in old];assert len(oldraw)==2815
blocktext={fp(p['paragraph_text']) for v in oldraw for p in v['contexts']};blockpage={norm(p['wikipedia_title']) for v in oldraw for p in v['contexts']};blockseed={str(h['id']) for v in oldraw for h in v['decomposed_instances']};blockq={norm(v['composed_question_text']) for v in oldraw}
features={};eligible=[];causes=Counter();overlaps=Counter()
for v in raw:
 assert v['id'] in meta
 text={fp(p['paragraph_text']) for p in v['paragraphs'] if p['is_supporting']};page={norm(p['title']) for p in v['paragraphs'] if p['is_supporting']};seed={str(h['id']) for h in v['question_decomposition']}
 overlaps['support_text']+=bool(text&blocktext);overlaps['support_titles']+=bool(page&blockpage);overlaps['seed_ids']+=bool(seed&blockseed)
 reason='too_long' if meta[v['id']]['native_input_tokens']>3072 else 'old_question' if norm(v['question']) in blockq else 'old_component_question_bare_ID_conservative' if seed&blockseed else 'old_visible_paragraph' if text&blocktext else 'old_visible_page' if page&blockpage else None
 if reason:causes[reason]+=1;continue
 eligible.append(v);features[v['id']]={(f,x) for f,xs in [('support_text',text),('support_titles',page),('seed_ids',seed)] for x in xs}
assert len(eligible)==r['strict_clean_eligible_rows']==875 and dict(causes)==r['first_exclusion_counts'] and dict(overlaps)==r['overlaps_any_length']
assert Counter(len(v['question_decomposition']) for v in eligible)==Counter({int(k):v for k,v in r['eligible_hops'].items()})
# Explicit adjacency/BFS rather than producer's union-find.
owners=defaultdict(set)
for k,fs in features.items():
 for f in fs:owners[f].add(k)
unseen=set(features);sizes=[]
while unseen:
 stack=[unseen.pop()];seen=set(stack)
 while stack:
  k=stack.pop();neighbors=set().union(*(owners[f] for f in features[k]))-seen
  seen.update(neighbors);unseen.difference_update(neighbors);stack.extend(neighbors)
 sizes.append(len(seen))
assert len(sizes)==r['support_overlap_components']==333 and max(sizes)==r['largest_component']==37
assert Counter(sizes)==Counter({int(k):v for k,v in r['component_size_histogram'].items()})
occupied=set();qseen=set();chosen=[]
for v in sorted(eligible,key=lambda v:(hashlib.sha256(('document-evidence-v27-feasibility-only-v1:'+v['id']).encode()).hexdigest(),v['id'])):
 if occupied&features[v['id']] or norm(v['question']) in qseen:continue
 occupied|=features[v['id']];qseen.add(norm(v['question']));chosen.append(v)
assert len(chosen)==r['fixed_hash_greedy_independent_rows']==343
seeds={x for fs in features.values() for f,x in fs if f=='seed_ids'};assert len(seeds)==r['distinct_original_IDs']==1229
assert all(len({str(h['id']) for h in v['question_decomposition']})>=2 for v in eligible)
assert min(len(eligible),len(seeds)//2)==r['independent_count_upper_bound']==614<r['desired_independent_items']==1024
# Four fixed source-hash examples: token lengths only, content never printed.
tok=AutoTokenizer.from_pretrained(R/'local/assets/qwen3-1.7b-base',local_files_only=True,trust_remote_code=False)
spots=sorted(raw,key=lambda v:hashlib.sha256(('v27-token-length-independent-spot:'+v['id']).encode()).hexdigest())[:4]
for v in spots:
 q=dict(item_id=v['id'],question=v['question'],contexts=[dict(title=p['title'],text=p['paragraph_text']) for p in v['paragraphs']]);text=render(q)
 assert len(tok.encode(text,add_special_tokens=False))==meta[v['id']]['native_input_tokens']
 assert hashlib.sha256(text.encode()).hexdigest()==meta[v['id']]['input_sha256']
# Independent closed-form normal CDF for each planning point, not observed power.
planning=read(C/'result.json')
for p in planning['normal_power_planning']:
 se=((p['discordance']-p['true_gain']**2)/p['n'])**.5;t=max(.03,NormalDist().inv_cdf(.975)*se)
 prob=1-NormalDist(p['true_gain'],se).cdf(t)
 assert abs(prob-p['approx_marginal_pass'])<1e-12 and abs(max(0,4*prob-3)-p['joint_four_lower_bound'])<1e-12
result=dict(status='PASS_INDEPENDENT_FRESH_COUNTS_COMPONENTS_UPPER_BOUND_TOKEN_SPOTS_AND_POWER_ARITHMETIC',DEV_rows=2417,clean_eligible=875,components=333,largest_component=37,independent_greedy=343,independent_upper_bound=614,desired_independent=1024,independent_token_length_spots=4,power_grid_points=len(planning['normal_power_planning']),DEV_result_sha256=sha(C/'DEV-mirror-metadata-result.json'),TRAIN_result_sha256=sha(C/'result.json'),checker_sha256=sha(__file__),limit='No semantic review, prediction outcomes or model evaluation; planning approximation only.')
(C/'independent-check.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))

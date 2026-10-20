"""CPU-only strict fresh-source availability and prospective power arithmetic.
No model output, new held cohort, official DEV/TEST acquisition, or training.
"""
from pathlib import Path
from collections import Counter
from datetime import datetime,timezone
import hashlib,json,math
from train.document_qa_data import evidence_identity
R=Path('/home/ubuntu/dario');C=R/'local/diagnostics/document-evidence-v27-confirmation-feasibility-v1'
C.mkdir(exist_ok=False)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
readrows=lambda p:[json.loads(s) for s in Path(p).read_text().splitlines()]
S=R/'local/nvme-study/data/musique-TRAIN-source-v12/musique_ans_train.raw.jsonl'
V=R/'local/nvme-cpt/data/document-QA-v21-TRAIN2048-held512-v1'
D=R/'local/nvme-cpt/data/document-evidence-plan-v23-TRAIN2048-held255-v2'
plan=dict(status='CPU_ONLY_AVAILABILITY_AND_POWER_NO_CONFIRMATION_RELEASE',utc=datetime.now(timezone.utc).isoformat(),
 question='Can the remaining already-downloaded TRAIN source support a fresh, source-separated powered confirmation?',
 raw_sha256=sha(S),old_manifests={str(p.relative_to(R)):sha(p) for p in [V/'manifest.json',D/'manifest.json']},
 exclusion='Candidate ID/normalized question/original component question may not overlap TRAIN2048 or either inspected held512/255. Candidate supporting paragraph/page may not occur anywhere visible in those cohorts.',
 window=3072,independence='Greedy fixed hash order, no shared support paragraph/page/component ID among retained questions. Constructive lower bound only; eligible count is an upper bound.',
 ranking_salt='document-evidence-v27-feasibility-only-v1:',
 planning='Normal approximation for paired mean difference; practical gate point>=.03 and two-sided95% lower>0. Four required comparisons (two controls times two seeds). Union-bound planning: each marginal>=.95 gives joint>=.80 without assuming independence.',
 effect_grid=[.03,.04,.05,.07],discordance_grid=[.10,.15,.20,.30],sample_grid=[128,255,512,768,1024,1536,2048,4096],
 limits='No model outputs or scores read. No cohort selection/release, official files, numerical jobs or old result changes. Planning approximation is not actual bootstrap power or a promise of semantic novelty.')
assert plan['raw_sha256']=='a85aeee64718f08ab9d11472f28b424e3d933794082b6d3c5b2481cd7e733345'
(C/'plan.json').write_text(json.dumps(plan,indent=2))
raw={v['id']:v for v in readrows(S)};meta={k:dict(evidence_identity(v),hops=len(v['decomposed_instances'])) for k,v in raw.items()}
lengths={v['item_id']:v['native_input_tokens'] for v in readrows(S.parent/'input-inventory-v2.jsonl')}
train=readrows(D/'TRAIN2048.jsonl');old512=readrows(V/'held512-questions-only.jsonl');old255=readrows(D/'held255-questions-only.jsonl')
assert (len(train),len(old512),len(old255))==(2048,512,255)
exclude={v['item_id'] for v in train+old512+old255};questions={meta[k]['question'] for k in exclude}
seen={f:set().union(*(meta[k][f] for k in exclude)) for f in ('visible_text','visible_titles','seed_ids')}
counts=Counter();eligible=[]
for k,v in meta.items():
 if lengths[k]>3072:counts['too_long']+=1;continue
 if k in exclude or v['question'] in questions:counts['old_question']+=1;continue
 if v['seed_ids']&seen['seed_ids']:counts['old_component_question']+=1;continue
 if v['support_text']&seen['visible_text']:counts['old_visible_paragraph']+=1;continue
 if v['support_titles']&seen['visible_titles']:counts['old_visible_page']+=1;continue
 eligible.append(v)
ranked=sorted(eligible,key=lambda v:(hashlib.sha256((plan['ranking_salt']+v['item_id']).encode()).hexdigest(),v['item_id']))
chosen=[];texts=set();titles=set();seeds=set();qs=set()
for v in ranked:
 if v['support_text']&texts or v['support_titles']&titles or v['seed_ids']&seeds or v['question'] in qs:continue
 chosen.append(v);texts.update(v['support_text']);titles.update(v['support_titles']);seeds.update(v['seed_ids']);qs.add(v['question'])
# Neither IDs nor gold are released as a candidate confirmation cohort.
power=[]
for d in plan['effect_grid']:
 for q in plan['discordance_grid']:
  assert 0<d<=q<1
  for n in plan['sample_grid']:
   se=math.sqrt((q-d*d)/n);threshold=max(.03,1.959963984540054*se)
   marginal=.5*math.erfc((threshold-d)/(se*math.sqrt(2)))
   power.append(dict(true_gain=d,discordance=q,n=n,approx_marginal_pass=marginal,joint_four_lower_bound=max(0,1-4*(1-marginal))))
summary=dict(status='CPU_FRESH_SOURCE_AVAILABILITY_AND_POWER_ARITHMETIC_COMPLETE',utc=datetime.now(timezone.utc).isoformat(),
 source_rows=len(raw),excluded_question_ids=len(exclude),disjoint_fresh_eligible_rows=len(eligible),eligible_hops=dict(Counter(v['hops'] for v in eligible)),
 exclusion_first_failure_counts=dict(counts),fixed_greedy_independent_rows=len(chosen),greedy_hops=dict(Counter(v['hops'] for v in chosen)),
 greedy_ids_sha256=hashlib.sha256(('\n'.join(v['item_id'] for v in chosen)+'\n').encode()).hexdigest(),
 maximum_possible_independent_rows_upper_bound=len(eligible),normal_power_planning=power,
 practical_threshold_caveat='At true gain exactly.03, the point>=.03 requirement caps asymptotic marginal pass probability at.5. Do not claim80% power at the decision threshold.',
 plan_sha256=sha(C/'plan.json'),source_sha256=sha(__file__),limits=plan['limits'])
(C/'result.json').write_text(json.dumps(summary,indent=2))
print(json.dumps({k:v for k,v in summary.items() if k!='normal_power_planning'},indent=2))

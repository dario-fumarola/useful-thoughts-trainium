"""Freeze complete TRAIN-derived QA cohorts and audit native supervised views."""
from collections import Counter
from datetime import datetime,timezone
from pathlib import Path
import hashlib,json,os
from transformers import AutoTokenizer
from train.document_qa_data import evidence_identity,select_cohort,supervised_trace,normalized
from train.document_evidence_data import question_projection,render
from train.document_qa_packing import training_pack,SOURCE_LABELS,EOS
R=Path('/home/ubuntu/dario');W=R/'local/worktrees/document-qa-v21'
D=R/'local/nvme-cpt/data/document-QA-v21-TRAIN2048-held512-v1';D.mkdir(parents=True,exist_ok=False)
S=R/'local/nvme-study/data/musique-TRAIN-source-v12/musique_ans_train.raw.jsonl'
M=R/'local/nvme-study/data/musique-TRAIN-mirror-v19'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert sha(S)=='a85aeee64718f08ab9d11472f28b424e3d933794082b6d3c5b2481cd7e733345'
plan=dict(status='CPU_DATA_FREEZE_NO_MODEL_OR_SCIENCE_RELEASE',utc=datetime.now(timezone.utc).isoformat(),
 question='Can complete, native single-model QA targets be trained on evidence-clean real documents?',
 cohort='At most3072 complete rendered input tokens;greedy salted hash chooses512 distinct questions with disjoint supporting paragraph text,page title and actual original dataset:seedID. Exclude held support from ALL train-visible passages/titles and held original seedIDs;deduplicate normalized composed questions;first2048 remaining by same hash.',
 qualification='Separate first128 ordered TRAIN rows only;no held model outputs before protocol release.',
 native_contract=dict(width=3328,target_slots=512,source_labels=SOURCE_LABELS),
 labels='Canonical author-converter-compatible complete decomposition including root/citations/#earlier-step references;canonical answer and attributed mirror aliases gold only.',
 views='direct answerEOS;prior complete trace+Final answer+answerEOS;post answerEOS+Post-answer reasoning+complete traceEOS. Every augmented arm uses plain direct on even passes,augmentation on odd passes.',
 loss='Half per-example meanCE over256 fixed uniformly sampled source target positions plus half per-example meanCE over complete task targets. Source positions identical across arms;not full-source CE.',
 references=['https://arxiv.org/html/2108.00573v3#S5','https://arxiv.org/html/2509.20186v1'],
 pass='512 evidence/seed-disjoint held,2048 distinct evidence-clean train,all6144 native views fit3328/512 without truncation;prefix/masks/causal targets/source-sampling exact.',
 limits='Authors TRAIN only,not official benchmarkDEV/TEST. Directed evidence split does not remove pretraining or semantic duplicates. Mirror aliases not authenticated against official ZIP. No training or model evaluation released.')
(D/'plan.json').write_text(json.dumps(plan,indent=2))
raw={r['id']:r for l in S.open() if (r:=json.loads(l))}
lengths={r['item_id']:r for l in (S.parent/'input-inventory-v2.jsonl').open() if (r:=json.loads(l))}
alias={r['item_id']:r for l in (M/'answer-aliases-by-author-raw-id.jsonl').open() if (r:=json.loads(l))}
reverse={r['official_id']:k for k,r in alias.items()}
canonical={reverse[r['id']]:r for l in (M/'musique_ans_v1.0_train.jsonl').open() if (r:=json.loads(l))}
records=[evidence_identity(row) for k,row in raw.items() if lengths[k]['native_input_tokens']<=3072]
training,held=select_cohort(records,held_count=512,train_count=2048)
expected=[json.loads(l)['item_id'] for l in (R/'local/diagnostics/document-QA-cohort-feasibility-v21-v1/3072-greedy_support_title_and_seed_id_disjoint-held512/candidate-held-groups.jsonl').open()]
assert [r['item_id'] for r in held]==expected
assert not ({r['item_id'] for r in training}&set(expected))
U=lambda rs,f:set().union(*(r[f] for r in rs))
assert not U(training,'visible_text')&U(held,'support_text')
assert not U(training,'visible_titles')&U(held,'support_titles')
assert not U(training,'seed_ids')&U(held,'seed_ids')
for field in ('support_text','support_titles','seed_ids'):
 assert sum(len(r[field]) for r in held)==len(U(held,field))
tok=AutoTokenizer.from_pretrained(R/'local/assets/qwen3-1.7b-base',local_files_only=True,trust_remote_code=False)
counts=Counter();maxima=Counter();trainrows=[];questions=[];golds=[];ungrounded=[]
for split,cohort in [('train',training),('held',held)]:
 for meta in cohort:
  r=raw[meta['item_id']];q=question_projection(r);text=render(q)
  assert hashlib.sha256(text.encode()).hexdigest()==lengths[r['id']]['input_sha256']
  label=supervised_trace(r,canonical[r['id']])
  for h in canonical[r['id']]['question_decomposition']:
   c=r['contexts'][h['paragraph_support_idx']]
   if normalized(h['answer']) not in normalized(c['wikipedia_title']+'\n'+c['paragraph_text']):
    ungrounded.append(dict(item_id=r['id'],split=split,hop_id=h['id'],answer=h['answer'],support_index=h['paragraph_support_idx']))
  gold=dict(item_id=r['id'],answer=r['answer_text'],answer_aliases=alias[r['id']]['answer_aliases'],trace=label,
            hops=len(r['decomposed_instances']),support_indices=[i for i,c in enumerate(r['contexts']) if c['is_supporting']],
            original_seed_ids=sorted(meta['seed_ids']))
  if split=='train':
   item=dict(item_id=r['id'],split='train',question=q,answer=gold['answer'],trace=label)
   trainrows.append(item);source_labels=None
   for view in ('direct','prior','post'):
    p=training_pack(item,tok,view);source=p['note_close_mask'];target=p['future_mask']
    active=source|target;pos=p['target_positions'][active]
    assert p['input_ids'][0,pos].tolist()==p['labels'][active].tolist()
    actualsource=p['labels'][source].tolist();assert len(actualsource)==256
    if source_labels is not None:assert actualsource==source_labels
    source_labels=actualsource
    counts[view+'_source_labels']+=len(actualsource);counts[view+'_task_labels']+=int(target.sum())
    maxima[view+'_active_slots']=max(maxima[view+'_active_slots'],int(active.sum()))
    maxima[view+'_sequence_tokens']=max(maxima[view+'_sequence_tokens'],int(pos.max())+1)
  else:questions.append(q);golds.append(gold)
  counts[split+'_hop'+str(gold['hops'])]+=1

def save_jsonl(name,rows):
 p=D/name
 with p.open('x') as f:
  for row in rows:f.write(json.dumps(row,ensure_ascii=False)+'\n')
 return dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p),rows=len(rows))
files=dict(training=save_jsonl('TRAIN2048.jsonl',trainrows),
           held_questions=save_jsonl('held512-questions-only.jsonl',questions),
           held_gold=save_jsonl('held512-gold-only.jsonl',golds),
           grounding_exceptions=save_jsonl('literal-grounding-exceptions.jsonl',ungrounded))
sourcefiles={n:sha(W/n) for n in ('train/document_evidence_data.py','train/document_qa_data.py','train/document_qa_packing.py','train/useful_note_sft_data.py')}
report=dict(status='PASS_ALL_6144_NATIVE_QA_VIEWS_AND_EVIDENCE_CLEAN_COHORTS_CPU_ONLY',utc=datetime.now(timezone.utc).isoformat(),
 plan_sha256=sha(D/'plan.json'),script_sha256=sha(__file__),source_sha256=sha(S),files=files,source_files=sourcefiles,
 items=2048,held=512,distinct_training_questions=len({r['question'] for r in training}),
 independent_held_groups=dict(support_text=512,support_page=512,original_seed_id=512),
 directed_overlap=dict(held_support_text_in_train_visible=0,held_support_titles_in_train_visible=0,original_seed_id=0),
 held_ids_sha256=hashlib.sha256(('\n'.join(expected)+'\n').encode()).hexdigest(),
 counts=dict(counts),maxima=dict(maxima),literal_grounding_exceptions=len(ungrounded),
 limits=plan['limits']+' Literal-grounding exceptions retained,not selected away;presence is not semantic entailment. No full-context backward/save/restore/timing qualification yet.')
(D/'manifest.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2),flush=True)

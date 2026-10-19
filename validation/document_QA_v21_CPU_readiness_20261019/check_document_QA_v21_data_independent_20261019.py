"""Independent raw-ID/evidence and native-label check; no policy/model calls."""
from pathlib import Path
from collections import Counter
from datetime import datetime,timezone
import hashlib,json
from transformers import AutoTokenizer
R=Path('/home/ubuntu/dario');D=R/'local/nvme-cpt/data/document-QA-v21-TRAIN2048-held512-v1'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();norm=lambda s:' '.join(s.casefold().split())
m=json.loads((D/'manifest.json').read_text());raw={r['id']:r for l in (R/'local/nvme-study/data/musique-TRAIN-source-v12/musique_ans_train.raw.jsonl').open() if (r:=json.loads(l))}
readrows=lambda p:[json.loads(l) for l in p.open()]
train=readrows(D/'TRAIN2048.jsonl');held=readrows(D/'held512-questions-only.jsonl');gold=readrows(D/'held512-gold-only.jsonl')
assert len(train)==2048 and len(held)==len(gold)==512
assert len({r['item_id'] for r in train+held})==2560
for f in m['files'].values():assert sha(f['path'])==f['sha256']
TS,TT,TQ=set(),set(),set();HS,HT,HQ=set(),set(),set()
for i,q in enumerate(held):
 r=raw[q['item_id']];assert r['composed_question_text']==q['question'] and r['answer_text']==gold[i]['answer']
 contexts=[{'title':c['wikipedia_title'],'text':c['paragraph_text']} for c in r['contexts']]
 assert contexts==q['contexts'] and set(q)=={'item_id','question','contexts'}
 s={norm(c['paragraph_text']) for c in r['contexts'] if c['is_supporting']};t={norm(c['wikipedia_title']) for c in r['contexts'] if c['is_supporting']};ids={(h['dataset'],h['id']) for h in r['decomposed_instances']}
 assert not(s&HS or t&HT or ids&HQ)
 HS|=s;HT|=t;HQ|=ids
for row in train:
 r=raw[row['item_id']];TS.update(norm(c['paragraph_text']) for c in r['contexts']);TT.update(norm(c['wikipedia_title']) for c in r['contexts']);TQ.update((h['dataset'],h['id']) for h in r['decomposed_instances'])
 assert row['answer']==r['answer_text'] and row['question']['contexts']==[{'title':c['wikipedia_title'],'text':c['paragraph_text']} for c in r['contexts']]
assert not(TS&HS or TT&HT or TQ&HQ)
tok=AutoTokenizer.from_pretrained(R/'local/assets/qwen3-1.7b-base',local_files_only=True,trust_remote_code=False)
# Independently reconstruct all task targets, source boundaries and inventories.
counts=Counter();maximum=Counter()
for row in train:
 q=row['question'];paragraphs=[f'[{i}] {c["title"]}\n{c["text"]}' for i,c in enumerate(q['contexts'],1)]
 source='Documents:\n'+'\n\n'.join(paragraphs)+'\n\nQuestion: '+q['question']+'\n'
 src=tok.encode(source,add_special_tokens=False);assert len(src)<=3072
 assert row['trace'].count('\nEvidence [')==len(raw[row['item_id']]['decomposed_instances'])
 for view in ('direct','prior','post'):
  opening=source+('Reasoning:\n' if view=='prior' else 'Answer:\n');ctx=tok.encode(opening,add_special_tokens=False)
  target=(row['trace']+'\nFinal answer:\n'+row['answer']) if view=='prior' else row['answer']
  seq=tok.encode(opening+target,add_special_tokens=False)+[151643]
  if view=='post':seq+=tok.encode('\nPost-answer reasoning:\n'+row['trace'],add_special_tokens=False)+[151643]
  assert seq[:len(src)]==src and seq[:len(ctx)]==ctx and len(seq)<=3328
  labels=len(seq)-len(ctx);counts[view+'_task_labels']+=labels;counts[view+'_source_labels']+=256
  maximum[view+'_active_slots']=max(maximum[view+'_active_slots'],256+labels)
  maximum[view+'_sequence_tokens']=max(maximum[view+'_sequence_tokens'],len(seq))
assert all(m['counts'][k]==v for k,v in counts.items()) and dict(maximum)==m['maxima']
result=dict(status='PASS_INDEPENDENT_RAW_IDENTITIES_EVIDENCE_DISJOINTNESS_AND_ALL_NATIVE_TARGET_INVENTORIES',utc=datetime.now(timezone.utc).isoformat(),manifest_sha256=sha(D/'manifest.json'),script_sha256=sha(__file__),training=2048,held=512,within_held_support_and_seed_overlap=0,held_support_in_training_visible_overlap=0,all_task_counts=dict(counts),maxima=dict(maximum),limits='No model results or hardware execution;semantic/pretraining independence not established.')
p=D/'independent-check.json'
with p.open('x') as f:json.dump(result,f,indent=2)
print(json.dumps(result,indent=2),flush=True)

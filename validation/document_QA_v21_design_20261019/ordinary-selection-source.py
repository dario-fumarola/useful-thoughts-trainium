"""Freeze a no-QA-format Wikipedia continuation panel,without reading model outputs."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json
from tokenizers import Tokenizer
R=Path('/home/ubuntu/dario');D=R/'local/nvme-cpt/data/document-QA-v21-ordinary256-v2';D.mkdir(exist_ok=False)
Q=R/'local/nvme-cpt/data/document-QA-v21-TRAIN2048-held512-v1';S=R/'local/nvme-study/data/musique-TRAIN-source-v12/musique_ans_train.raw.jsonl'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();norm=lambda s:' '.join(s.casefold().split());fp=lambda s:hashlib.sha256(norm(s).encode()).hexdigest()
plan=dict(status='PROSPECTIVE_CPU_ORDINARY_PANEL_ONLY',utc=datetime.now(timezone.utc).isoformat(),source_sha256=sha(S),QA_manifest_sha256=sha(Q/'manifest.json'),items=256,prefix_tokens=256,target_tokens=32,
 selection='Unique normalizedparagraphtext;exclude EVERY QAtrain/held visible paragraph text AND page title. Rank remaining complete paragraph sequences with>=288native tokens by salted normalizedtext hash;first256 distinct page titles,one paragraph perpage. No model outcomes. Prefix first256tokens,target next32;ordinary quality excerpt only,not QA context truncation.',
 pass_threshold='256 distinct text/title-disjoint paragraphs,exact32native targets;no support/answer labels needed.',limits='MuSiQue authorTRAIN source paragraphs,not officialDEV/TEST. Exactdisjointness doesnotestablishpretrainingorsemanticnovelty.')
(D/'plan.json').write_text(json.dumps(plan,indent=2))
texts,titles=set(),set()
for f in (Q/'TRAIN2048.jsonl',Q/'held512-questions-only.jsonl'):
 for line in f.open():
  r=json.loads(line);q=r.get('question') if isinstance(r.get('question'),dict) else r
  for c in q['contexts']:texts.add(fp(c['text']));titles.add(norm(c['title']))
tok=Tokenizer.from_file(str(R/'local/assets/qwen3-1.7b-base/tokenizer.json'));pool={}
for line in S.open():
 r=json.loads(line)
 for c in r['contexts']:
  text,title=c['paragraph_text'],c['wikipedia_title'];key=fp(text)
  if key in texts or norm(title) in titles or key in pool:continue
  ids=tok.encode(title+'\n'+text,add_special_tokens=False).ids
  if len(ids)<288:continue
  pool[key]=dict(item_id=key,title=title,paragraph_text=text,source_question_id=r['id'],prefix_ids=ids[:256],target_ids=ids[256:288],paragraph_native_tokens=len(ids))
ranked=sorted(pool.values(),key=lambda r:hashlib.sha256(('document-QA-v21-ordinary:'+r['item_id']).encode()).hexdigest())
selected=[];seen_pages=set()
for row in ranked:
 if norm(row['title']) in seen_pages:continue
 selected.append(row);seen_pages.add(norm(row['title']))
 if len(selected)==256:break
assert len(selected)==len({r['item_id'] for r in selected})==256
# One paragraph per page avoids treating neighboring pages as independent rows.
# Choose the first paragraph perpage in fixed hash order before any model run.
assert len({norm(r['title']) for r in selected})==256
p=D/'ordinary256.jsonl'
with p.open('x') as f:
 for row in selected:f.write(json.dumps(row,ensure_ascii=False)+'\n')
m=dict(status='FROZEN_CPU_ORDINARY256_DISJOINT_QA_VISIBLE_PAGES_AND_TEXT',utc=datetime.now(timezone.utc).isoformat(),plan_sha256=sha(D/'plan.json'),source_sha256=sha(S),QA_manifest_sha256=sha(Q/'manifest.json'),script_sha256=sha(__file__),eligible_unique_paragraphs=len(pool),items=256,unique_pages=256,
 data_file=str(p),bytes=p.stat().st_size,data_sha256=sha(p),prefix_tokens=256,target_tokens=32,total_target_tokens=8192,limits=plan['limits'])
(D/'manifest.json').write_text(json.dumps(m,indent=2));print(json.dumps(m,indent=2),flush=True)

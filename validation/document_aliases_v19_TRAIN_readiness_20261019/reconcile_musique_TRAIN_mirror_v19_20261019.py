"""Reconcile the already downloaded TRAIN using the authors' documented ID rename."""
from pathlib import Path
from datetime import datetime,timezone
from collections import Counter
import ast,hashlib,json,urllib.request
R=Path('/home/ubuntu/dario');D=R/'local/nvme-study/data/musique-TRAIN-mirror-v19';sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
source=D/'musique_ans_v1.0_train.jsonl';assert sha(source)=='83a75b1e11e4e9bb8f8308e72ac40ca617ae4431b3a0d955b61cab259248490a'
rawfile=R/'local/nvme-study/data/musique-TRAIN-source-v12/musique_ans_train.raw.jsonl';assert sha(rawfile)=='a85aeee64718f08ab9d11472f28b424e3d933794082b6d3c5b2481cd7e733345'
with urllib.request.urlopen('https://api.github.com/repos/StonyBrookNLP/musique/commits/main',timeout=20) as r:commit=json.load(r)['sha']
with urllib.request.urlopen(f'https://raw.githubusercontent.com/StonyBrookNLP/musique/{commit}/utils/common.py',timeout=20) as r:code=r.read()
(D/'author-utils-common.py').write_bytes(code)
node=next(n for n in ast.parse(code.decode()).body if isinstance(n,ast.FunctionDef) and n.name=='translate_id')
mapping=ast.literal_eval(node.body[0].value)
assert mapping=={'double':'2hop','triple_ii':'3hop1','triple_io':'3hop2','quadruple_iii':'4hop1','quadruple_iot':'4hop2','quadruple_ioh1':'4hop3','quadruple_ioh2':'4hop4'}
raw={}
for line in rawfile.open():
 row=json.loads(line);kind,tail=row['id'].split('__');key=mapping[kind]+'__'+tail;assert key not in raw;raw[key]=row
seen=set();aliases=Counter();fields=Counter();hops=Counter();not_visible=[];gold=[]
for line in source.open():
 row=json.loads(line);key=row['id'];assert key in raw and key not in seen;seen.add(key);base=raw[key]
 assert row['question']==base['composed_question_text'];fields['question']+=1
 assert row['answer']==base['answer_text'];fields['canonical_answer']+=1
 assert row['answerable']==base['answerable']==True;fields['answerable']+=1
 assert len(row['paragraphs'])==len(base['contexts'])
 for index,(a,b) in enumerate(zip(row['paragraphs'],base['contexts'])):
  assert a['idx']==index and a['title']==b['wikipedia_title'] and a['paragraph_text']==b['paragraph_text'] and a['is_supporting']==b['is_supporting']
 fields['all_visible_paragraphs_order_and_support_flags']+=1
 assert len(row['question_decomposition'])==len(base['decomposed_instances'])
 for a,b in zip(row['question_decomposition'],base['decomposed_instances']):assert a['id']==b['id'] and a['answer']==b['answer_text']
 fields['hop_ids_answers_and_count']+=1;hops[len(row['question_decomposition'])]+=1
 values=row['answer_aliases'];assert isinstance(values,list) and all(isinstance(v,str) for v in values)
 aliases['rows_with_aliases']+=bool(values);aliases['alias_strings']+=len(values);aliases['empty_alias_strings']+=sum(not v.strip() for v in values)
 visible='  '.join(c['wikipedia_title']+' '+c['paragraph_text'] for c in base['contexts']).lower()
 for value in values:
  if value.lower().strip() not in visible:not_visible.append(dict(item_id=base['id'],official_id=key,alias=value))
 gold.append(dict(item_id=base['id'],official_id=key,canonical_answer=row['answer'],answer_aliases=values))
assert seen==raw.keys() and len(seen)==19938
for name,values in [('aliases-outside-literal-visible-text.jsonl',not_visible),('answer-aliases-by-author-raw-id.jsonl',gold)]:
 with (D/name).open('x') as f:f.write(''.join(json.dumps(v)+'\n' for v in values))
report=dict(status='ALL_19938_MIRROR_TRAIN_CANONICAL_RECORDS_RECONCILE_WITH_AUTHOR_RAW',utc=datetime.now(timezone.utc).isoformat(),repo='bdsaglam/musique',revision='22873a405dd809893b22ada0b499299fb612d2df',TRAIN_file=source.name,TRAIN_sha256=sha(source),TRAIN_bytes=source.stat().st_size,author_raw_sha256=sha(rawfile),rows=len(seen),matching_fields=dict(fields),hops=dict(hops),alias_counts=dict(aliases),aliases_outside_literal_visible_text=len(not_visible),aliases_by_raw_id_sha256=sha(D/'answer-aliases-by-author-raw-id.jsonl'),author_ID_translation=dict(commit=commit,source_sha256=sha(D/'author-utils-common.py'),mapping=mapping),script_sha256=sha(__file__),limits=['OnlyTRAINfileopened;officialDEV/TEST/globalaliasesunread. No model/cohort/budgetreleased.','ThefirstreconciliationassertedunchangedIDsandstopped;originalscript/logretained. Authorsdocumenttheexactshape-prefixrename;thispassappliesonlythatbijectiveIDmap. No redownloadorcanonicaldataedit.','Authoredquestions,canonicalanswers,visibleparagraphorder/text/titles/supportflags,andhopIDs/answersmatchtheauthorrawrecords. Decompositionquestionstringsarenotclaimedbyte-identical.','Aliasstringsarefromapinnedmirror;originalofficialZIPbytescouldnotbeacquired. Canonicalfieldsareindependentlycrosschecked;aliasannotationauthenticityisnotindependentlyproven.','AnyfutureQAmetricmuststatecanonical/aliasprovenance;stringagreementisnotsemanticentailmentorpretrainingnovelty. Aliaslabelsremainofflinegold,neverpolicyinputs.'])
with (D/'reconciliation.json').open('x') as f:json.dump(report,f,indent=2)
print(json.dumps(report,indent=2),flush=True)

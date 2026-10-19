"""Fetch one pinned TRAIN file;reconcile canonicaldata withtheauthorrawTRAIN."""
from pathlib import Path
from datetime import datetime,timezone
from collections import Counter
import hashlib,json,os,shutil,urllib.request
R=Path('/home/ubuntu/dario');D=R/'local/nvme-study/data/musique-TRAIN-mirror-v19';D.mkdir(exist_ok=False)
repo='bdsaglam/musique';revision='22873a405dd809893b22ada0b499299fb612d2df';filename='musique_ans_v1.0_train.jsonl'
expected='83a75b1e11e4e9bb8f8308e72ac40ca617ae4431b3a0d955b61cab259248490a';size=241046755
url=f'https://huggingface.co/datasets/{repo}/resolve/{revision}/{filename}'
(D/'plan.json').write_text(json.dumps(dict(status='PINNED_TRAIN_ONLY_DATA_ACQUISITION_NO_COHORT_OR_MODEL',utc=datetime.now(timezone.utc).isoformat(),repo=repo,revision=revision,file=filename,expected_bytes=size,expected_sha256=expected,scope='OnlynamedTRAINpayloadrequested. NoofficialDEV/TEST/globalaliasesfile. Mirrororiginisdisclosed;canonicaldatareconciliationrequiredbeforeuse.'),indent=2))
assert shutil.disk_usage(D).free>35*2**30+2*size
h=hashlib.sha256();total=0;destination=D/filename
with urllib.request.urlopen(url,timeout=30) as response,destination.open('xb') as out:
 while block:=response.read(4*1024**2):
  total+=len(block);assert total<=size;h.update(block);out.write(block)
 out.flush();os.fsync(out.fileno())
assert total==size and h.hexdigest()==expected
print('PinnedTRAINfilehashverified',total,flush=True)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
rawfile=R/'local/nvme-study/data/musique-TRAIN-source-v12/musique_ans_train.raw.jsonl'
assert sha(rawfile)=='a85aeee64718f08ab9d11472f28b424e3d933794082b6d3c5b2481cd7e733345'
raw={r['id']:r for line in rawfile.open() if (r:=json.loads(line))}
seen=set();aliases=Counter();fields=Counter();hops=Counter();alias_not_visible=[]
for line in destination.open():
 row=json.loads(line);key=row['id'];assert key in raw and key not in seen;seen.add(key);base=raw[key]
 assert row['question']==base['composed_question_text'];fields['question']+=1
 assert row['answer']==base['answer_text'];fields['canonical_answer']+=1
 assert row['answerable']==base['answerable']==True;fields['answerable']+=1
 assert len(row['paragraphs'])==len(base['contexts'])
 for index,(a,b) in enumerate(zip(row['paragraphs'],base['contexts'])):
  assert a['idx']==index and a['title']==b['wikipedia_title'] and a['paragraph_text']==b['paragraph_text'] and a['is_supporting']==b['is_supporting']
 fields['all_visible_paragraphs_order_and_support_flags']+=1
 assert len(row['question_decomposition'])==len(base['decomposed_instances'])
 for a,b in zip(row['question_decomposition'],base['decomposed_instances']):
  assert a['id']==b['id'] and a['answer']==b['answer_text']
 fields['hop_ids_answers_and_count']+=1;hops[len(row['question_decomposition'])]+=1
 values=row['answer_aliases'];assert isinstance(values,list) and all(isinstance(v,str) and v.strip() for v in values)
 aliases['rows_with_aliases']+=bool(values);aliases['alias_strings']+=len(values)
 visible='  '.join(c['wikipedia_title']+' '+c['paragraph_text'] for c in base['contexts']).lower()
 for value in values:
  if value.lower().strip() not in visible:alias_not_visible.append(dict(item_id=key,alias=value))
assert seen==raw.keys() and len(seen)==19938
(D/'aliases-outside-literal-visible-text.jsonl').write_text(''.join(json.dumps(v)+'\n' for v in alias_not_visible))
report=dict(status='ALL_19938_MIRROR_TRAIN_CANONICAL_RECORDS_RECONCILE_WITH_AUTHOR_RAW',utc=datetime.now(timezone.utc).isoformat(),repo=repo,revision=revision,TRAIN_file=filename,TRAIN_sha256=expected,TRAIN_bytes=size,author_raw_sha256=sha(rawfile),rows=len(seen),matching_fields=dict(fields),hops=dict(hops),alias_counts=dict(aliases),aliases_outside_literal_visible_text=len(alias_not_visible),script_sha256=sha(__file__),limits=['OnlyTRAINfileopened;officialDEV/TEST/globalaliasannotationsunread. No model/cohort/budgetreleased.','IDs,authoredquestions,canonicalanswers,visibleparagraphorder/text/titles/supportflags,andhopIDs/answersmatchtheauthorrawrecords.','Aliasstringscome fromapinnedmirror;originalofficialZIPbytescouldnotbeindependentlyacquired. Canonicalfieldsarecrosschecked,aliasannotationauthenticityisnotindependentlyproven.','Exactannotation/stringagreementisnotsemanticentailmentorpretrainingnovelty. FutureQAmetricsmuststatecanonical/aliasprovenanceseparately.'])
(D/'reconciliation.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2),flush=True)

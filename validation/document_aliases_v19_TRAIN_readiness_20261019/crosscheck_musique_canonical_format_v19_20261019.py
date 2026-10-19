"""Execute only reviewed pure author conversion functions;comparecanonicalTRAINfields."""
from pathlib import Path
from datetime import datetime,timezone
from typing import Dict,List
import ast,hashlib,json,urllib.request
R=Path('/home/ubuntu/dario');D=R/'local/nvme-study/data/musique-TRAIN-mirror-v19';read=lambda p:json.loads(Path(p).read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
prior=read(D/'reconciliation.json');commit=prior['author_ID_translation']['commit']
code=(D/'author-raw-format-converter.py').read_bytes()
common=ast.parse((D/'author-utils-common.py').read_text());conversion=ast.parse(code.decode());names={'translate_id','step_placeholder','filter_answers_in_context','raw_instance_to_official_format'}
functions=[n for tree in [common,conversion] for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names];assert len(functions)==4
constants={}
for n in ast.parse((D/'author-utils-constants.py').read_text()).body:
 pairs=([(target,n.value) for target in n.targets] if isinstance(n,ast.Assign)
        else [(n.target,n.value)] if isinstance(n,ast.AnnAssign) else [])
 for target,value in pairs:
  if isinstance(target,ast.Name) and target.id in ['CONSTITUENT_QUESTION_START','CONSTITUENT_QUESTION_END','REPLACEMENT_QUESTION_START','REPLACEMENT_QUESTION_END']:constants[target.id]=ast.literal_eval(value)
assert set(constants)=={'CONSTITUENT_QUESTION_START','CONSTITUENT_QUESTION_END','REPLACEMENT_QUESTION_START','REPLACEMENT_QUESTION_END'}
assert not any(isinstance(n,(ast.Import,ast.ImportFrom)) for f in functions for n in ast.walk(f))
namespace=dict(Dict=Dict,List=List,**constants)
exec(compile(ast.Module(body=functions,type_ignores=[]),'<pinned-author-pure-converter>','exec'),namespace)
raw={}
for line in (R/'local/nvme-study/data/musique-TRAIN-source-v12/musique_ans_train.raw.jsonl').open():
 row=json.loads(line);raw[namespace['translate_id'](row['id'])]=row
count=0;differences=[]
for line in (D/'musique_ans_v1.0_train.jsonl').open():
 row=json.loads(line);expected=namespace['raw_instance_to_official_format'](raw[row['id']],{})
 actual={k:v for k,v in row.items() if k!='answer_aliases'};expected.pop('answer_aliases')
 if actual!=expected:differences.append(dict(id=row['id'],different_fields=[k for k in actual.keys()|expected.keys() if actual.get(k)!=expected.get(k)]))
 count+=1
assert count==19938
report=dict(status='ALL_CANONICAL_TRAIN_FIELDS_EXCEPT_ALIAS_ANNOTATIONS_MATCH_AUTHOR_CONVERTER' if not differences else 'CANONICAL_FORMAT_DIFFERENCES_REVIEW_REQUIRED',utc=datetime.now(timezone.utc).isoformat(),rows=count,differing_rows=len(differences),differences=differences,author_commit=commit,author_converter_sha256=sha(D/'author-raw-format-converter.py'),author_common_sha256=sha(D/'author-utils-common.py'),author_constants_sha256=sha(D/'author-utils-constants.py'),pure_functions=sorted(names),script_sha256=sha(__file__),limits=['Twoearlypure-functionloadersstoppedbeforedatawhileconstantslocationwasunresolved. Finalpassreadspinnedutils.constantswithoutchangingannotations;originalscripts/logspreserved.','NoofficialDEV/TEST/globalaliasesread. Emptyaliasmapisintentional;onlyallcanonicalfieldsincludingdecompositionquestions/paragraphindicesarechecked.','Thischecksannotationtransformation,notsemanticcorrectnessorlearnedcapability. Mirroredaliasprovenanceisunchanged. No cohort/model release.'])
with (D/'canonical-format-crosscheck.json').open('x') as f:json.dump(report,f,indent=2)
print(json.dumps(report,indent=2),flush=True)

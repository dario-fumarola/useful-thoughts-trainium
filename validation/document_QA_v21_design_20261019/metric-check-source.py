"""Compare our reimplementation with pinned author's pure metric functions only."""
from pathlib import Path
import ast,collections,hashlib,json,random,re,string,urllib.request
from train.document_qa_answers import normalize_answer,exact_match,token_f1
R=Path('/home/ubuntu/dario');D=R/'local/diagnostics/document-QA-v21-official-metric-CPU-v1';D.mkdir(exist_ok=False)
url='https://raw.githubusercontent.com/stonybrooknlp/musique/922ac98f19a201998dbdae6d7f2887a5258dbdeb/metrics/answer.py'
body=urllib.request.urlopen(url,timeout=30).read();(D/'author-answer.py').write_bytes(body)
module=ast.parse(body);wanted={'normalize_answer','get_tokens','compute_exact','compute_f1'};nodes=[n for n in module.body if isinstance(n,ast.FunctionDef) and n.name in wanted]
assert {n.name for n in nodes}==wanted
namespace=dict(re=re,string=string,collections=collections)
exec(compile(ast.Module(body=nodes,type_ignores=[]),str(D/'author-answer.py'), 'exec'),namespace)
rng=random.Random(2026101921);words=['New','York','City','THE','an','answer','Lyon','Cedar','Northport','United','States','NYC','','é','Å','İ']
strings=['','The, United States!','NYC','New York City','***','the','a']
for _ in range(512):strings.append(' '.join(rng.choice(words)+rng.choice(['',',','.','!','-']) for _ in range(rng.randrange(8))))
for a,b in zip(strings,strings[1:]+strings[:1]):
 assert normalize_answer(a)==namespace['normalize_answer'](a)
 assert exact_match(a,b)==namespace['compute_exact'](b,a)
 assert abs(token_f1(a,b)-namespace['compute_f1'](b,a))<1e-14
result=dict(status='PASS_PINNED_OFFICIAL_METRIC_NORMALIZATION_EM_F1',source_url=url,source_sha256=hashlib.sha256(body).hexdigest(),synthetic_pairs=len(strings),limits='Code-onlydownload. No datasetDEV/TEST ormodeloutcomes. Only four reviewed purefunctions executed;not the repository evaluation loader.')
(D/'report.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2),flush=True)

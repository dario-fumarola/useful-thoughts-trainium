"""CPU-only TRAIN annotation grounding andlabel-length feasibility;no cohort/model."""
from pathlib import Path
from datetime import datetime,timezone
from collections import Counter
import hashlib,json,re,unicodedata
from tokenizers import Tokenizer
from train.document_evidence_data import question_projection,supplied_support
R=Path('/home/ubuntu/dario');S=R/'local/nvme-study/data/musique-TRAIN-source-v12/musique_ans_train.raw.jsonl'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert sha(S)=='a85aeee64718f08ab9d11472f28b424e3d933794082b6d3c5b2481cd7e733345'
T=Tokenizer.from_file(str(R/'local/assets/qwen3-1.7b-base/tokenizer.json'))
I=R/'local/nvme-study/data/musique-TRAIN-source-v12/input-inventory-v2.jsonl'
lengths={r['item_id']:r['native_input_tokens'] for l in I.open() if (r:=json.loads(l))}
def tokens(s):return re.findall(r'\w+',unicodedata.normalize('NFKC',s).casefold())
def contains(needle,hay):
 return bool(needle) and any(hay[i:i+len(needle)]==needle for i in range(len(hay)-len(needle)+1))
def grounded(answer,paragraphs):
 a=tokens(answer)
 return any(contains(a,p) for p in paragraphs)
# Positiveandnegativecross-boundary/Unicodecasesarenottiedtothedataset.
assert grounded('New York',[tokens('In NEW YORK, this happened.')])
assert not grounded('Yorkshire',[tokens('York')])
assert not grounded('York City',[tokens('York'),tokens('City')])
assert grounded('Ａｌｐｈａ',[tokens('Alpha')]) and not grounded('',[tokens('anything')])
counts={};bybudget={str(b):Counter() for b in [2048,3072]};hops=Counter();support_counts=Counter();label_lengths=[];issues=[];total_hops=0
for line in S.open():
 row=json.loads(line);q=question_projection(row);hop=row['decomposed_instances']
 assert row['setname']=='train' and row['answerable'] and len(hop) in (2,3,4)
 visible=[tokens(c['wikipedia_title']+' '+c['paragraph_text']) for c in row['contexts']]
 support=[tokens(c['wikipedia_title']+' '+c['paragraph_text']) for c in row['contexts'] if c['is_supporting']]
 assert support;hops[len(hop)]+=1;support_counts[len(support)]+=1
 sub=[grounded(v['answer_text'],visible) for v in hop];subsupport=[grounded(v['answer_text'],support) for v in hop]
 final=grounded(row['answer_text'],visible);finalsupport=grounded(row['answer_text'],support)
 terminal=any(tokens(row['answer_text'])==tokens(v['answer_text']) for v in hop)
 note=supplied_support(row,q);n=len(T.encode(note,add_special_tokens=False).ids);label_lengths.append(n)
 values=dict(rows=1,final_answer_in_visible=int(final),final_answer_in_support=int(finalsupport),all_hop_answers_in_visible=int(all(sub)),all_hop_answers_in_support=int(all(subsupport)),final_matches_a_hop_answer=int(terminal),hop_answers=len(hop),hop_answers_in_visible=sum(sub),hop_answers_in_support=sum(subsupport))
 total_hops+=len(hop)
 for k,v in values.items():counts[k]=counts.get(k,0)+v
 for budget,c in bybudget.items():
  if lengths[row['id']]<=int(budget):
   c.update(values)
   for bound in [128,256,512,1024]:c[f'verbatim_support_note_within_{bound}']=c.get(f'verbatim_support_note_within_{bound}',0)+int(n<=bound)
 if not(finalsupport and all(subsupport) and terminal):
  issues.append(dict(item_id=row['id'],visible_final=final,support_final=finalsupport,missing_visible_hop_indices=[i for i,v in enumerate(sub) if not v],missing_support_hop_indices=[i for i,v in enumerate(subsupport) if not v],terminal_matches=terminal))
assert counts['rows']==len(lengths)==19938 and counts['hop_answers']==total_hops
label_lengths.sort()
D=R/'local/diagnostics/document-grounding-v18-CPU-v1';D.mkdir(exist_ok=False)
(D/'annotation-exception-ids.jsonl').write_text(''.join(json.dumps(v)+'\n' for v in issues))
report=dict(status='CPU_TRAIN_ANNOTATION_GROUNDING_AUDIT_NO_COHORT_MODEL_OR_EVALUATION_RELEASE',utc=datetime.now(timezone.utc).isoformat(),source_sha256=sha(S),inventory_sha256=sha(I),script_sha256=sha(__file__),rows=counts,by_complete_input_budget=dict(bybudget),hops=dict(hops),support_paragraph_counts=dict(support_counts),verbatim_support_note_native_token_quantiles={str(p):label_lengths[round((len(label_lengths)-1)*p)] for p in [0,.25,.5,.75,.95,1]},exception_ids_rows=len(issues),exception_ids_sha256=sha(D/'annotation-exception-ids.jsonl'),normalization='NFKC+casefold+Unicodewordtokens;exactcontiguouswordsequencewithinonevisibleparagraphincludingitstitle;neveracrossparagraphs.',limits=['AuthorsTRAINonly;officialDEV/TESTunopened. Globalansweraliasesfilenotdownloadedorread.','Answer-stringpresenceisnotsemanticentailment,correctevidenceselectionorlearnedreasoning. Titlesarevisibleandcancontainanswers.','RawTRAINlacksperitemofficialansweraliases;thesearecanonicalannotationcoveragecounts,notofficialEM/F1.','Verbatimfullsupportnotesareanofflineoracle,lengthinventoryonly. Noprompt,cohort,supervisedtarget,budgetorlongcontextmodelreleased.','TheexistingV17study/frozencomparisonsareunchanged;nextscientificdirectionchosenafteritscompletereadout.'])
(D/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2),flush=True)

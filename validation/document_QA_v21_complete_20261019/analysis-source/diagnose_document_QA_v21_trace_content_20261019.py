"""Posthoc descriptive diagnosis of alreadyclosed traces;no newmodel calls/gates.

Canonical trace fidelity and literal citation presence are limited diagnostics,
not semantic entailment or a semantic regrading of benchmark answers.
"""
from collections import Counter
from pathlib import Path
from datetime import datetime,timezone
import csv,gzip,hashlib,json,re
from train.document_qa_answers import normalize_answer
R=Path('/home/ubuntu/dario');F=R/'validation/document_QA_v21_complete_20261019.json';P=F.with_suffix('');C=R/'local/diagnostics/document-QA-v21-study-v1'
read=lambda p:json.loads(Path(p).read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert read(C/'final-readout-completion.json')['report_sha256']==sha(F)
def payload(name):return [json.loads(l) for l in gzip.decompress((P/name).read_bytes()).splitlines()]
qs={v['item_id']:v for v in map(json.loads,(R/'local/nvme-cpt/data/document-QA-v21-TRAIN2048-held512-v1/held512-questions-only.jsonl').read_text().splitlines())};gold={v['item_id']:v for v in payload('fixed-held-gold-metadata.jsonl.gz')}
def trace(text):
 text=re.split(r'(?im)^[ \t]*Final answer:',text)[0]
 steps=re.findall(r'(?m)^Step (\d+):[^\n]*\nEvidence \[(\d+)\]:[ \t]*([^\n]*)',text)
 return [dict(index=int(a),citation=int(b),answer=c.strip()) for a,b,c in steps]
results={};raw=[]
for seed in (2026101921,2026101922):
 model=f'qa_prior-seed{seed}';attempts={v['item_id']:v for v in payload(model+'-all-generation-attempts.jsonl.gz') if v['mode']=='prior'}
 scores={}
 for v in payload(model+'-all-likelihood-rows.jsonl.gz'):
  if v['panel']=='QA':scores[(v['item_id'],v['mode'])]=v
 for item,v in attempts.items():
  t=trace(v['text']);g=trace(gold[item]['trace']);assert len(g)==gold[item]['hops']
  indices=[s['index'] for s in t];complete=indices==list(range(1,len(g)+1))
  same=[normalize_answer(a['answer'])==normalize_answer(b['answer']) for a,b in zip(t,g)] if complete else []
  literal=[];matches=[]
  for step in t:
   index=step['citation']-1;valid=0<=index<len(qs[item]['contexts']);content=qs[item]['contexts'][index]['text'] if valid else ''
   norm=normalize_answer(step['answer']);present=valid and bool(norm) and ' '+norm+' ' in ' '+normalize_answer(content)+' ';literal.append(present)
   expected=g[step['index']-1] if 0<step['index']<=len(g) else None
   matches.append(dict(**step,citation_in_range=valid,claim_literal_in_cited_paragraph=present,canonical_gold_step_answer=expected['answer'] if expected else None,canonical_gold_citation=expected['citation'] if expected else None))
  own=scores[(item,'own_replay')];empty=scores[(item,'empty')]
  raw.append(dict(model=model,item_id=item,gold_hops=len(g),generated_steps=len(t),sequential_complete_trace=complete,canonical_intermediate_answers_all_match=complete and all(same[:-1]),canonical_all_step_answers_match=complete and all(same),all_claims_literal_in_own_citations=bool(t) and all(literal),claims_literal=sum(literal),claims=len(t),final_copies_last_claim=bool(t) and normalize_answer(t[-1]['answer'])==normalize_answer(v['parsed']['answer']),canonical_em=v['canonical_em'],attributed_alias_em=v['attributed_alias_em'],gold_answer_span_gain_own_minus_empty=own['full_logp']-empty['full_logp'],step_details=matches))
 vs=[v for v in raw if v['model']==model];byhop={}
 for hops in sorted({v['gold_hops'] for v in vs}):
  selected=[v for v in vs if v['gold_hops']==hops];byhop[str(hops)]=dict(items=len(selected),generated_step_counts=dict(Counter(v['generated_steps'] for v in selected)),sequential_complete=sum(v['sequential_complete_trace'] for v in selected),canonical_correct=sum(v['canonical_em'] for v in selected))
 groups={}
 for name in ('canonical_correct','alias_only','neither_canonical_nor_attributed_alias'):
  selected=[v for v in vs if (v['canonical_em']==1 if name=='canonical_correct' else v['canonical_em']==0 and v['attributed_alias_em']==1 if name=='alias_only' else v['attributed_alias_em']==0)]
  groups[name]=dict(items=len(selected),mean_gold_answer_span_gain_own_minus_empty=sum(v['gold_answer_span_gain_own_minus_empty'] for v in selected)/len(selected) if selected else None)
 results[str(seed)]=dict(items=512,by_gold_hops=byhop,sequential_complete_trace=sum(v['sequential_complete_trace'] for v in vs),all_canonical_intermediates_match=sum(v['canonical_intermediate_answers_all_match'] for v in vs),all_canonical_step_answers_match=sum(v['canonical_all_step_answers_match'] for v in vs),all_claims_literal_in_own_citations=sum(v['all_claims_literal_in_own_citations'] for v in vs),literal_claims=sum(v['claims_literal'] for v in vs),total_claims=sum(v['claims'] for v in vs),final_copies_last_claim=sum(v['final_copies_last_claim'] for v in vs),canonical_correct_with_noncanonical_intermediate=sum(v['canonical_em'] and not v['canonical_intermediate_answers_all_match'] for v in vs),likelihood_by_answer_match_group=groups)
assert len(raw)==1024
out=dict(status='POSTHOC_STORED_TRACE_STRUCTURE_CANONICAL_FIDELITY_LITERAL_CITATION_AND_CONFIDENCE_DIAGNOSIS',utc=datetime.now(timezone.utc).isoformat(),report_sha256=sha(F),source_sha256=sha(__file__),results=results,limits=['All1024 originalprior generations retained;no newmodelforwards,primaryscores or gateschanged.','Canonicalintermediate fidelity rejectsparaphrases/alternativevaliddecompositions;not semanticcorrectness.','Literalanswerpresence incitedparagraph doesnotestablish therelation,entailment,citationfaithfulness orreasoning.','Correctfinals withnoncanonicalintermediate are diagnosticcandidates,not automaticallyfalsechains.','Likelihoodgrouping conditionson generatedanswer matches;descriptive,not causal or a calibration theorem.'])
(P/'stored-trace-content-diagnosis.json').write_text(json.dumps(out,indent=2));(P/'all1024-stored-trace-diagnostics.jsonl.gz').write_bytes(gzip.compress((''.join(json.dumps(v)+'\n' for v in raw)).encode(),mtime=0));print(json.dumps(out,indent=2),flush=True)

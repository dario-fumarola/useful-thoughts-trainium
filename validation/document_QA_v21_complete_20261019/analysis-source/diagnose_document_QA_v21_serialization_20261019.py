"""Posthoc serialization diagnostic on unchanged outputs;not a new primary metric.

Gold-independent rule for every model/mode:one anchored Final answer field when
present;otherwise first complete nonempty outputline. Multiple Finalfields still
unsupported. Errors/caps retained. No modelcalls orprimaryreport edits.
"""
from pathlib import Path
from datetime import datetime,timezone
import gzip,hashlib,json
from dataclasses import asdict
from train.document_qa_answers import ParsedAnswer,parse_answer,score
from analyze_document_QA_v21_20261019 import contrast
R=Path('/home/ubuntu/dario');F=R/'validation/document_QA_v21_complete_20261019.json';P=F.with_suffix('');r=json.loads(F.read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();original_sha=sha(F)
def payload(name):return [json.loads(l) for l in gzip.decompress((P/name).read_bytes()).splitlines()]
gold={v['item_id']:v for v in payload('fixed-held-gold-metadata.jsonl.gz')};tables={};data={};allrows=[]
for model in r['tables']:
 data[model]={};tables[model]={}
 for v in payload(model+'-all-generation-attempts.jsonl.gz'):
  terminated=v['stop_type'].startswith('token:');capped=v['stop_type']=='length_cap';parsed=parse_answer(v['text'],'prior',terminated=terminated,capped=capped);kind='anchored_final'
  if parsed.reason=='missing_final_field':parsed=parse_answer(v['text'],'direct',terminated=terminated,capped=capped);kind='first_line_fallback'
  if v['status']!='ok':parsed=ParsedAnswer('execution_error',reason='execution_error')
  result=dict(model=model,item_id=v['item_id'],mode=v['mode'],serialization=kind,parsed=asdict(parsed),frozen_primary_canonical_em=v['canonical_em'],**score(parsed,gold[v['item_id']]['answer'],gold[v['item_id']]['answer_aliases']))
  data[model].setdefault(v['mode'],{})[v['item_id']]=result;allrows.append(result)
 for mode,byid in data[model].items():
  assert len(byid)==512;vs=list(byid.values());tables[model][mode]=dict(items=512,correct=sum(v['canonical_em'] for v in vs),attributed_alias_correct=sum(v['attributed_alias_em'] for v in vs),canonical_F1=sum(v['canonical_f1'] for v in vs)/512,complete=sum(v['parsed']['status']=='complete' for v in vs),first_line_fallbacks=sum(v['serialization']=='first_line_fallback' for v in vs),net_extra_correct_vs_frozen=sum(v['canonical_em']-v['frozen_primary_canonical_em'] for v in vs))
cs={}
for seed in r['seeds']:
 a=data[f'qa_direct-seed{seed}']['reference_cot'];b=data[f'qa_prior-seed{seed}']['prior'];cs[seed]=contrast({k:v['canonical_em'] for k,v in a.items()},{k:v['canonical_em'] for k,v in b.items()})
assert len(allrows)==20992 and sha(F)==original_sha
out=dict(status='POSTHOC_GOLD_INDEPENDENT_ALTERNATIVE_SERIALIZATION_DIAGNOSTIC_PRIMARY_UNCHANGED',utc=datetime.now(timezone.utc).isoformat(),original_primary_report_sha256=original_sha,source_sha256=sha(__file__),rule=__doc__,tables=tables,posthoc_prior_generated_minus_direct_reference_contrasts=cs,limits=['Thisserializationrule waswritten afterinspectingformatfailures;notpreregistered,not a repairedprimaryscore ornewpass.','Firstlinefallback doesnot establish that a lineis a semantically validanswer;wholelineEM scored,no goldwordsearch.','Bareanswers underCoTdemonstrations are not evidence thecontrol generatedreasoning.','Alloriginaloutputs/caps/errors retained,no modelexecutionsrerun,no gatechanges. Freshconfirmation needed.'])
(P/'posthoc-serialization-diagnostic.json').write_text(json.dumps(out,indent=2));(P/'all20992-posthoc-serialization-diagnostics.jsonl.gz').write_bytes(gzip.compress((''.join(json.dumps(v)+'\n' for v in allrows)).encode(),mtime=0));print(json.dumps(dict(status=out['status'],direct_reference={seed:tables[f'qa_direct-seed{seed}']['reference_cot'] for seed in r['seeds']},posthoc_contrasts=cs),indent=2),flush=True)

"""One CPU cohort/evidence/trace feasibility question;no model orfinalcohortrelease."""
from collections import Counter,defaultdict
from datetime import datetime,timezone
from pathlib import Path
import hashlib,json,math
from tokenizers import Tokenizer
from train.document_evidence_data import question_projection,render
R=Path('/home/ubuntu/dario');D=R/'local/diagnostics/document-QA-cohort-feasibility-v21-v1';D.mkdir(exist_ok=False)
S=R/'local/nvme-study/data/musique-TRAIN-source-v12/musique_ans_train.raw.jsonl';I=S.parent/'input-inventory-v2.jsonl'
M=R/'local/nvme-study/data/musique-TRAIN-mirror-v19';V16=R/'validation/document_evidence_v16_directed_split_20261019.json'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();read=lambda p:json.loads(Path(p).read_text())
assert sha(S)=='a85aeee64718f08ab9d11472f28b424e3d933794082b6d3c5b2481cd7e733345'
assert read(M/'canonical-format-crosscheck.json')['differing_rows']==0
plan=dict(status='CPU_PROSPECTIVE_FEASIBILITY_ONLY',utc=datetime.now(timezone.utc).isoformat(),question='Do true original seed-question IDs permit evidence-clean document QA cohorts without conflating generic relation templates with repeated facts?' ,source_sha256=sha(S),inventory_sha256=sha(I),script_sha256=sha(__file__),conditions=dict(input_token_budgets=[2048,3072],held_questions=[256,384,512],selection=['V16fixedhash','greedy_support_title_and_seed_id_disjoint']),selection_salt='document-evidence-v16-audit-only:',pass_if_any='Atleastone exactrequestedcohort:held>=256,strictretainedTRAIN>=2048,andheldcombinedsupporttext/title/originalseedIDclusters>=256. Prefer512held;allconditionsreportedbeforechoosinganyscientificcohort.',correction='V16/V20 raw component_question_texts sometimes contain generic relation programs, not original question identities. V21 uses dataset:id from decomposed_instances; old feasibility artifacts remain unchanged.',trace_inventory='Provisionalexplicitseed-question/answer/citationtraceincludingroot;completeanswerEOS;no truncation. Lengthsareinventory,notafrozenpromptorhardwarequalification.',limits='No model outcomes,weights,evaluationorfinalcohort release. OfficialDEV/TEST/globalaliasesunopened.')
(D/'plan.json').write_text(json.dumps(plan,indent=2))
lengths={r['item_id']:r for line in I.open() if (r:=json.loads(line))}
aliasrows=[json.loads(l) for l in (M/'answer-aliases-by-author-raw-id.jsonl').open()];official_to_raw={v['official_id']:v['item_id'] for v in aliasrows}
canonical={official_to_raw[r['id']]:r for line in (M/'musique_ans_v1.0_train.jsonl').open() if (r:=json.loads(line))}
tok=Tokenizer.from_file(str(R/'local/assets/qwen3-1.7b-base/tokenizer.json'));norm=lambda s:' '.join(s.casefold().split());fp=lambda s:hashlib.sha256(norm(s).encode()).hexdigest()
records=[]
for line in S.open():
 row=json.loads(line);q=question_projection(row);text=render(q);prior=lengths[row['id']];assert hashlib.sha256(text.encode()).hexdigest()==prior['input_sha256']
 support={fp(c['paragraph_text']) for c in row['contexts'] if c['is_supporting']};titles={norm(c['wikipedia_title']) for c in row['contexts'] if c['is_supporting']}
 assert support and titles
 steps=[]
 for i,(hop,linked) in enumerate(zip(row['decomposed_instances'],canonical[row['id']]['question_decomposition']),1):
  assert linked['paragraph_support_idx'] is not None
  steps.append(f'Step {i}: {linked["question"]}\nEvidence [{linked["paragraph_support_idx"]+1}]: {hop["answer_text"]}')
 trace='\n'.join(steps);answer=row['answer_text']
 direct=len(tok.encode(answer,add_special_tokens=False).ids)+1
 prior_n=len(tok.encode(trace+'\nFinal answer:\n'+answer,add_special_tokens=False).ids)+1
 post_n=len(tok.encode(answer,add_special_tokens=False).ids)+1+len(tok.encode('\nPost-answer reasoning:\n'+trace,add_special_tokens=False).ids)+1
 records.append(dict(id=row['id'],question=norm(q['question']),tokens=prior['native_input_tokens'],visible={fp(c['paragraph_text']) for c in row['contexts']},visible_titles={norm(c['wikipedia_title']) for c in row['contexts']},support=support,support_titles=titles,components={str(h['dataset'])+':'+str(h['id']) for h in row['decomposed_instances']},hops=len(row['decomposed_instances']),rank=hashlib.sha256(('document-evidence-v16-audit-only:'+row['id']).encode()).hexdigest(),trace_tokens=len(tok.encode(trace,add_special_tokens=False).ids),direct_target_tokens=direct,prior_target_tokens=prior_n,post_target_tokens=post_n))
assert len(records)==len(lengths)==len(canonical)==19938
print('VerifiedallTRAINinputhashes;inventoriedcompleteprovisionaltraces.',flush=True)

def groups(held,fields):
 parent=list(range(len(held)))
 def find(i):
  while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
  return i
 owners={}
 for i,row in enumerate(held):
  for field in fields:
   for value in row[field]:
    key=(field,value)
    if key in owners:
     a,b=find(i),find(owners[key]);parent[a]=b
    else:owners[key]=i
 clusters=defaultdict(list)
 for i,row in enumerate(held):clusters[find(i)].append(row['id'])
 sizes=sorted(map(len,clusters.values()),reverse=True);mapping={}
 for members in clusters.values():
  group=hashlib.sha256(('v21:'+','.join(fields)+':'+'|'.join(sorted(members))).encode()).hexdigest()
  mapping.update({key:group for key in members})
 return dict(clusters=len(sizes),singletons=sum(n==1 for n in sizes),largest=sizes[0],cluster_size_histogram=dict(Counter(sizes)),perfect_within_cluster_correlation_effective_n=len(held)**2/sum(n*n for n in sizes)),mapping

def quantiles(values):
 v=sorted(values);return {str(p):v[round((len(v)-1)*p)] for p in [0,.5,.9,.95,.99,1]}
results={};v16=read(V16)
for budget in [2048,3072]:
 eligible=sorted((r for r in records if r['tokens']<=budget),key=lambda r:(r['rank'],r['id']))
 for strategy in ['V16fixedhash','greedy_support_title_and_seed_id_disjoint']:
  pool=[];seenq=set();seen_support=set();seen_titles=set();seen_components=set()
  for row in eligible:
   if row['question'] in seenq:continue
   if strategy.startswith('greedy') and (row['support']&seen_support or row['support_titles']&seen_titles or row['components']&seen_components):continue
   pool.append(row);seenq.add(row['question']);seen_support|=row['support'];seen_titles|=row['support_titles'];seen_components|=row['components']
   if len(pool)==512:break
  for wanted in [256,384,512]:
   held=pool[:wanted];key=f'{budget}-{strategy}-held{wanted}';H={r['id'] for r in held};Q={r['question'] for r in held}
   union=lambda f:set().union(*(r[f] for r in held))
   hsupport,htitles,hcomponents=[union(f) for f in ['support','support_titles','components']]
   train=[r for r in eligible if r['id'] not in H and r['question'] not in Q and not(r['visible']&hsupport) and not(r['visible_titles']&htitles) and not(r['components']&hcomponents)]
   assert all(not r['visible']&hsupport and not r['visible_titles']&htitles and not r['components']&hcomponents for r in train)
   grouping={};maps={}
   for label,fields in [('support_text',['support']),('support_title',['support_titles']),('original_seed_id',['components']),('support_text_or_title',['support','support_titles']),('support_or_seed_id',['support','support_titles','components'])]:grouping[label],maps[label]=groups(held,fields)
   held_hash=hashlib.sha256(('\n'.join(r['id'] for r in held)+'\n').encode()).hexdigest();train_hash=hashlib.sha256(('\n'.join(r['id'] for r in train)+'\n').encode()).hexdigest()
   if False: # V16 component-string exclusion was intentionally superseded by actual original seed IDs.
    assert held_hash==v16['complete_input_token_budgets'][str(budget)]['held_ids_sha256']
    assert train_hash==v16['complete_input_token_budgets'][str(budget)]['policies']['also_exclude_held_component_question']['training_ids_sha256']
   cohort=D/key;cohort.mkdir()
   (cohort/'candidate-TRAIN-ids.txt').write_text(''.join(r['id']+'\n' for r in train))
   (cohort/'candidate-held-groups.jsonl').write_text(''.join(json.dumps(dict(item_id=r['id'],input_tokens=r['tokens'],hops=r['hops'],**{name+'_group_id':mapping[r['id']] for name,mapping in maps.items()}))+'\n' for r in held))
   stats=dict(requested_held=wanted,actual_held=len(held),eligible=len(eligible),retained_training=len(train),unique_training_questions=len({r['question'] for r in train}),held_hops=dict(Counter(r['hops'] for r in held)),train_hops=dict(Counter(r['hops'] for r in train)),held_ids_sha256=held_hash,train_ids_sha256=train_hash,groupings=grouping,trace_inventory={field:dict(training=quantiles([r[field] for r in train]),held=quantiles([r[field] for r in held])) for field in ['trace_tokens','direct_target_tokens','prior_target_tokens','post_target_tokens']},full_input_plus_prior_target_max=dict(training=max(r['tokens']+r['prior_target_tokens'] for r in train),held=max(r['tokens']+r['prior_target_tokens'] for r in held)),declared_feasible=bool(len(held)==wanted and len(held)>=256 and len(train)>=2048 and grouping['support_or_seed_id']['clusters']>=256))
   results[key]=stats;print(key,'train',len(train),'heldclusters',grouping['support_text_or_title']['clusters'],'feasible',stats['declared_feasible'],flush=True)
report=dict(status='CPU_COHORT_EVIDENCE_AND_TRACE_FEASIBILITY_COMPLETE_NO_RELEASE',utc=datetime.now(timezone.utc).isoformat(),plan_sha256=sha(D/'plan.json'),source_sha256=sha(S),inventory_sha256=sha(I),projection_sha256=sha(R/'train/document_evidence_data.py'),canonical_crosscheck_sha256=sha(M/'canonical-format-crosscheck.json'),script_sha256=sha(__file__),conditions=results,declared_feasibility_pass=any(v['declared_feasible'] for v in results.values()),supersedes_component_identity_interpretation='V16/V20 raw component strings include generic relation templates. They remain exactly reported; V21 uses true original dataset:id. No model outcome or old scientific result changed.',limits=['Nofinalcohort,prompt,objective,model,trainingorevaluationreleased. AllcountsuseauthorsTRAIN;nomodeloutcomesread.','Groupingbasedonexactnormalizedsupporttext/title(orcomponentquestion)doesnotprove semanticorpretrainingindependence. Effective-nformulaassumesperfectwithingroupcorrelation,isadescriptivebound,notobservedmodelerrorcorrelation.','Greedyheldselectionusesofflinegoldsupportmetadatatosplitandpreventevidenceoverlap;policiesneverreceiveit. Train/held uniquequestionIDsare notautomaticallyindependentdocuments.','Traceformatprovisional:completehumanseed-question/answer/citationsequenceincludingroot. Tokeninventorymustnotbeconfusedwithsourcepacking/readergain/hardwarequalification.','TargetcountsincludeexplicitEOSandfragmenttokenization;actualtrain/evalnativeboundaryalignmentstillneedsimplementation. No truncatedpassages/traces.'])
with (D/'report.json').open('x') as f:json.dump(report,f,indent=2)
print(json.dumps(dict(status=report['status'],feasible=report['declared_feasibility_pass']),indent=2),flush=True)

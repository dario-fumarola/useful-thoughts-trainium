"""Fixed complete QA plots,all paired contrasts,costs and reproducible examples."""
from pathlib import Path
import csv,gzip,hashlib,json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
R=Path('/home/ubuntu/dario');F=R/'validation/document_QA_v21_complete_20261019.json';P=F.with_suffix('');r=json.loads(F.read_text());check=json.loads((P/'independent-readout-check.json').read_text());assert check['report_sha256']==hashlib.sha256(F.read_bytes()).hexdigest()
def csvfile(name,header,values):
 with (P/name).open('x',newline='') as f:
  w=csv.writer(f,lineterminator='\n');w.writerow(header);w.writerows(values)
csvfile('all-answer-failures-costs.csv',['model','mode','n','correct','canonical_EM','canonical_F1','attributed_alias_EM','unsupported','pending','execution_error','capped','actions','prompt_tokens','worker_wall_sum_seconds'],[[model,mode,512,t['correct'],t['canonical_em'],t['canonical_f1'],t['attributed_alias_em'],t['parsed_statuses'].get('unsupported',0),t['parsed_statuses'].get('pending',0),t['execution_errors'],t['capped'],t['generated_actions'],t['prompt_tokens'],t['generation_wall_seconds_sum']] for model,modes in r['tables'].items() for mode,t in modes.items()])
csvfile('all-primary-secondary-paired-contrasts.csv',['seed','endpoint','name','treatment_minus_reference','lower95','upper95','items'],[[seed,endpoint,name,v['difference_b_minus_a'],*v['ci95'],v['items']] for seed,s in r['seeds'].items() for endpoint in ('primary_canonical_EM_contrasts','secondary_canonical_F1_contrasts','ordinary_NLL_cost') for name,v in s[endpoint].items()])
csvfile('all-likelihood-contrasts.csv',['seed','part','name','treatment_minus_reference','lower95','upper95','items'],[[seed,part,name,v['difference_b_minus_a'],*v['ci95'],v['items']] for seed,s in r['seeds'].items() for part,cs in s['secondary_likelihood_contrasts'].items() for name,v in cs.items()])
fig,axes=plt.subplots(1,2,figsize=(13,5.5),sharex=True,sharey=True)
for ax,seed in zip(axes,r['seeds']):
 d=f'qa_direct-seed{seed}';p=f'qa_prior-seed{seed}';post=f'qa_post-seed{seed}'
 conditions=[('B0','reference_cot','Unchanged B0 conventional CoT'),(d,'direct','Ordinary-trained direct'),(d,'reference_cot','Ordinary-trained conventional CoT'),(p,'prior','Prior-trained generated reasoning'),(p,'empty','Prior-trained empty reasoning'),(post,'direct','Post-trained direct')]
 vals=[100*r['tables'][m][mode]['canonical_em'] for m,mode,_ in conditions];colors=['#777777','#2878a5','#3796ba','#cf7e2a','#b5a38c','#4c9760']
 ax.barh(range(len(vals)),vals,color=colors,height=.65)
 for i,(value,(m,mode,label)) in enumerate(zip(vals,conditions)):ax.text(value+.6,i,str(r['tables'][m][mode]['correct']),va='center',fontsize=9)
 ax.set_yticks(range(len(vals)),[v[2] for v in conditions]);ax.set_xlim(0,105);ax.set_title('Seed '+seed);ax.set_xlabel('Canonical complete-answer exact match (%)');ax.grid(axis='x',alpha=.15);ax.spines[['top','right']].set_visible(False)
axes[0].invert_yaxis();fig.suptitle('Evidence-clean document QA: all 512 attempts retained')
fig.text(.5,.02,'Authors’ TRAIN-derived held cohort; official DEV/TEST unopened. Bar labels: correct /512.\nEach path has a 256-action cap;actual costs differ. This is supervised CoT/QA adaptation,not emergence or broad pretraining.',ha='center',fontsize=8)
fig.tight_layout(rect=(0,.1,1,.94))
for ext in ('png','svg'):fig.savefig(P/('complete-answer-comparison.'+ext),dpi=170,bbox_inches='tight')
plt.close(fig)
fig,axes=plt.subplots(1,2,figsize=(13,5.5),sharey=True)
names=['post_direct_minus_direct_direct','prior_prior_minus_prior_empty','prior_prior_minus_direct_direct','prior_prior_minus_direct_prior','prior_prior_minus_direct_reference_cot','prior_prior_minus_B0_reference_cot']
labels=['Post direct − ordinary direct','Prior generated − prior empty','Prior generated − ordinary direct','Prior generated − ordinary generated','Prior generated − ordinary conventional CoT','Prior generated − B0 conventional CoT']
for ax,(seed,s) in zip(axes,r['seeds'].items()):
 for i,name in enumerate(names):
  v=s['primary_canonical_EM_contrasts'][name];point=100*v['difference_b_minus_a'];lo,hi=np.array(v['ci95'])*100
  ax.errorbar(point,i,xerr=[[max(0,point-lo)],[max(0,hi-point)]],fmt='o',color='#2878a5',capsize=3)
 ax.axvline(0,color='#777777',lw=.8);ax.axvline(3,color='#ad7531',lw=.8,ls=':');ax.set_yticks(range(len(labels)),labels);ax.set_title('Seed '+seed);ax.set_xlabel('Canonical EM difference (percentage points)');ax.grid(axis='x',alpha=.15);ax.spines[['top','right']].set_visible(False)
axes[0].invert_yaxis();fig.suptitle('Separate training,inference and practical-value contrasts')
fig.text(.5,.02,'20,000 paired question bootstrap draws;checkpoint-conditional unadjusted95% intervals.\nEach relevant gate requires ≥3pp(point) and lower95%>0 in BOTH seeds. No seed pooling.',ha='center',fontsize=8)
fig.tight_layout(rect=(0,.1,1,.94))
for ext in ('png','svg'):fig.savefig(P/('declared-contrasts.'+ext),dpi=170,bbox_inches='tight')
plt.close(fig)
# Examples fixed by item hash before reading outputs;all selected cases retained.
Q=R/'local/nvme-cpt/data/document-QA-v21-TRAIN2048-held512-v1';qs={v['item_id']:v for v in map(json.loads,(Q/'held512-questions-only.jsonl').read_text().splitlines())};gold={v['item_id']:v for v in map(json.loads,(Q/'held512-gold-only.jsonl').read_text().splitlines())}
selected=sorted(qs,key=lambda k:(hashlib.sha256(('QA21_EXAMPLE_V1|'+k).encode()).hexdigest(),k))[:8]
examples=[]
for item in selected:
 outputs=[]
 for model,modes in r['tables'].items():
  data=[json.loads(v) for v in gzip.decompress((P/(model+'-all-generation-attempts.jsonl.gz')).read_bytes()).splitlines()]
  allowed=['direct','reference_cot'] if model=='B0' else ['direct','prior','empty','reference_cot']
  outputs.extend(dict(model=model,**v) for v in data if v['item_id']==item and v['mode'] in allowed)
 examples.append(dict(item_id=item,question=qs[item],gold=gold[item],outputs=outputs))
(P/'fixed-hash-eight-examples.json').write_text(json.dumps(dict(selection='First8 sorted SHA256(QA21_EXAMPLE_V1|item_id);all conditions shown,no outcome selection.',items=examples),indent=2))
print('Rendered complete frozen QA comparison,all contrasts,costs and fixed-hash examples.',flush=True)

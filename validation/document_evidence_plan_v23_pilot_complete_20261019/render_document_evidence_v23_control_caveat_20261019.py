"""Render three declared contrasts and absolute results after independent check."""
from pathlib import Path
import json,csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path('/home/ubuntu/dario');F=R/'validation/document_evidence_plan_v23_pilot_complete_20261019.json';P=F.with_suffix('')
r=json.loads(F.read_text());check=json.loads((P/'independent-readout-check.json').read_text())
assert check['status']=='PASS_INDEPENDENT_FIELDS_METRICS_POINTERS_BOOTSTRAPS_AND_DECISIONS'
keys=['selected_minus_matched_direct','selected_minus_same_direct','selected_minus_same_empty']
labels=['Selection − matched direct','Selection − same-model direct','Selection − supplied-empty prompt*']
fig,axis=plt.subplots(figsize=(8.5,3.3))
for index,key in enumerate(keys):
 v=r['canonical_EM_contrasts'][key];point=100*v['difference_treatment_minus_control'];low,high=[100*x for x in v['ci95']]
 y=2-index;axis.errorbar(point,y,xerr=[[point-low],[high-point]],fmt='o',color='#164b75' if index==0 else '#62676c',capsize=4,linewidth=1.6)
 axis.annotate(f'{point:+.1f} [{low:+.1f}, {high:+.1f}]',(high,y),xytext=(8,0),textcoords='offset points',va='center',fontsize=9)
axis.axvline(0,color='#888888',linewidth=.9)
axis.set_yticks([2,1,0],labels);axis.set_ylim(-.5,2.5);axis.set_xlabel('Canonical exact-match gain (percentage points)')
axis.set_title('Passage selection pilot · one data-order seed · 255 questions',fontsize=11,pad=14)
low=min(0,*[100*r['canonical_EM_contrasts'][key]['ci95'][0] for key in keys]);high=max(0,*[100*r['canonical_EM_contrasts'][key]['ci95'][1] for key in keys]);span=max(8,high-low)
axis.set_xlim(low-.1*span,high+.55*span)
axis.spines[['top','right','left']].set_visible(False);axis.grid(axis='x',alpha=.12);axis.tick_params(axis='y',length=0)
fig.text(.02,.015,'*The supplied-empty prompt regenerates IDs on all 255 questions; this row does not isolate selection omission.\nPrimary needs ≥3 pp and a positive 95% lower bound. Direct answers contain no generated IDs.\nOne seed; intervals are conditional on these checkpoints. No replicated or cost advantage established.',fontsize=7.8,color='#454545')
fig.tight_layout(rect=(0,.18,1,1))
fig.savefig(P/'primary-contrasts-with-control-caveat.svg');fig.savefig(P/'primary-contrasts-with-control-caveat.png',dpi=180);plt.close(fig)
path=P/'primary-contrasts-with-control-caveat.csv'
expected=[['contrast','items','gain_pp','ci95_low_pp','ci95_high_pp']]
for key in keys:
 v=r['canonical_EM_contrasts'][key];expected.append([key,str(v['items']),str(100*v['difference_treatment_minus_control']),*[str(100*x) for x in v['ci95']]])
if path.exists():
 with path.open() as stream:assert list(csv.reader(stream))==expected
else:
 with path.open('x') as stream:csv.writer(stream).writerows(expected)
print('Rendered fixed primary/local contrasts;no score or threshold changes')

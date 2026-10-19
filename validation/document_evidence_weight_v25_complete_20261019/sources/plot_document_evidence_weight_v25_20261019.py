"""Export fixed fitting, developmental-answer and ordinary-cost figures."""
from pathlib import Path
import csv
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

R=Path('/home/ubuntu/dario');C=R/'local/diagnostics/document-evidence-weight-v25-ablation-v1'
read=lambda p:json.loads(p.read_text())
fit=read(C/'TRAIN-fit/result.json')
held=read(C/'held-native-and-replay-result.json')
ordinary=read(C/'ORDINARY/result.json')
assert read(C/'TRAIN-fit/independent-check.json')['status'].startswith('PASS_')
assert read(C/'held-independent-check.json')['status'].startswith('PASS_')
assert read(C/'ORDINARY/independent-check.json')['status'].startswith('PASS_')
O=C/'figures';O.mkdir(exist_ok=False)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none'})
colors=['#666666','#0072B2']
fig,ax=plt.subplots(figsize=(7,4.5),layout='constrained')
for offset,field,label,color in [(-.18,'control_exact','Weight 0.1',colors[0]),(.18,'exact','Weight 1.0',colors[1])]:
 xs=[n+offset for n in range(3)];ys=[100*fit['strata'][str(n)][field]/fit['strata'][str(n)]['items'] for n in (2,3,4)]
 bars=ax.bar(xs,ys,.34,label=label,color=color)
 for bar,n in zip(bars,(2,3,4)):
  v=fit['strata'][str(n)];ax.annotate(f"{v[field]}/{v['items']}",(bar.get_x()+bar.get_width()/2,bar.get_height()),xytext=(0,4),textcoords='offset points',ha='center',fontsize=9)
ax.axhline(90,ls='--',lw=1,color='#777777',label='90% fitting threshold (2/3 only)')
ax.set(xticks=[0,1,2],xticklabels=['Two supports','Three supports','Four supports\n(descriptive)'],ylabel='Exact annotated sets (%)',ylim=(0,108),title='TRAIN selection fitting — same fixed 256 questions')
ax.legend(loc='lower left',frameon=False,fontsize=9)
fig.savefig(O/'TRAIN-selection-fitting.svg');plt.close(fig)
with (O/'TRAIN-selection-fitting.csv').open('x',newline='') as f:
 w=csv.writer(f);w.writerow(['support_count','questions','weight0.1_exact','weight1.0_exact','paired_gain','paired_CI95_low','paired_CI95_high'])
 for n in (2,3,4):
  v=fit['strata'][str(n)];w.writerow([n,v['items'],v['control_exact'],v['exact'],v['paired_gain'],*v['paired_CI95']])

names={
 ('native:selection','old-matched-direct'):'Native selection − matched direct training',
 ('native:selection','old-selection-native'):'Native selection: weight 1.0 − 0.1',
 ('native:selection','native:direct'):'Native selection − same-model direct',
 ('answer-replay:selected_id_replay','answer-replay:empty'):'Closed-field own IDs − empty IDs',
 ('answer-replay:oracle','answer-replay:selected_id_replay'):'Closed-field oracle IDs − own IDs'}
contrasts=[v for v in held['contrasts'] if (v['treatment'],v['control']) in names]
assert len(contrasts)==5
fig,ax=plt.subplots(figsize=(9,4.4),layout='constrained')
for i,v in enumerate(contrasts):
 x=100*v['paired_gain'];low,high=[100*z for z in v['CI95']]
 color=colors[1] if v['treatment'].startswith('native:') else '#009E73'
 ax.errorbar(x,i,xerr=[[max(0,x-low)],[max(0,high-x)]],fmt='o',capsize=4,color=color)
 ax.annotate(f'{x:+.1f} [{low:+.1f}, {high:+.1f}]',(high,i),xytext=(7,0),textcoords='offset points',va='center',fontsize=9)
ax.axvline(0,lw=1,color='#555555');ax.invert_yaxis()
ax.set(yticks=list(range(5)),yticklabels=[names[(v['treatment'],v['control'])] for v in contrasts],xlabel='Correct-answer gain, percentage points (paired 95% interval)',title='Held255 developmental answers — previously inspected panel')
ax.margins(x=.45,y=.15)
fig.savefig(O/'held-answer-contrasts.svg');plt.close(fig)
with (O/'held-answer-contrasts.csv').open('x',newline='') as f:
 w=csv.writer(f);w.writerow(['treatment','control','gain','CI95_low','CI95_high','corrected','regressed'])
 for v in held['contrasts']:w.writerow([v['treatment'],v['control'],v['paired_gain'],*v['CI95'],v['corrected'],v['regressed']])
with (O/'absolute-held-answers.csv').open('x',newline='') as f:
 w=csv.writer(f);w.writerow(['path','questions','canonical_correct','canonical_F1','alias_correct','unsupported','errors','capped','actions'])
 for name,v in held['tables'].items():w.writerow([name,v['items'],v['canonical_correct'],v['mean_canonical_F1'],v['alias_correct'],v['unsupported'],v['execution_errors'],v['capped'],v['generated_actions']])

labels={'B0':'Untouched B0','evidence_direct':'Matched direct training','evidence_select':'Selection weight 0.1'}
fig,ax=plt.subplots(figsize=(7.3,3.4),layout='constrained')
for i,(name,v) in enumerate(sorted(ordinary['paired'].items())):
 x=v['mean_nll_change'];lo,hi=v['CI95']
 ax.errorbar(x,i,xerr=[[max(0,x-lo)],[max(0,hi-x)]],fmt='o',color=colors[1],capsize=4)
 ax.annotate(f'{x:+.3f} [{lo:+.3f}, {hi:+.3f}]',(hi,i),xytext=(7,0),textcoords='offset points',va='center',fontsize=9)
ax.axvline(0,lw=1,color='#555555');ax.invert_yaxis();ax.margins(x=.5,y=.25)
ax.set(yticks=list(range(3)),yticklabels=[labels[k] for k in sorted(ordinary['paired'])],xlabel='Weight 1.0 minus reference NLL, nats/token (positive is worse)',title='Ordinary256 continuation cost — paired page intervals')
fig.savefig(O/'ordinary-NLL-cost.svg');plt.close(fig)
print(json.dumps(dict(status='EXPORTED_FIXED_V25_READOUT_FIGURES',directory=str(O),files=[p.name for p in O.iterdir()])))

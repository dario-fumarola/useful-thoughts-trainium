"""Show fixed answer contrasts and ordinary-page costs; no endpoint selection."""
from pathlib import Path
import csv,json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path('/home/ubuntu/dario');C=R/'local/diagnostics/document-evidence-format-weight-v26-control-v1'
read=lambda p:json.loads(p.read_text())
h=read(C/'held-result.json');o=read(C/'ORDINARY/result.json')
assert read(C/'held-independent-check.json')['status'].startswith('PASS_') and read(C/'ORDINARY/independent-check.json')['status'].startswith('PASS_')
O=C/'figures';O.mkdir(exist_ok=False)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none'})
names={'format0.1:direct':'Format 0.1\ndirect','format1:direct':'Format 1.0\ndirect','selector1:direct':'Selection 1.0\ndirect','selector1:own':'Selection 1.0\ngenerated IDs'}
fig,ax=plt.subplots(figsize=(7.5,4.4),layout='constrained')
values=[100*h['tables'][k]['canonical_correct']/255 for k in names]
bars=ax.bar(range(4),values,color=['#888888','#0072B2','#009E73','#E69F00'])
for bar,key in zip(bars,names):
    ax.annotate(f"{int(h['tables'][key]['canonical_correct'])}/255",(bar.get_x()+bar.get_width()/2,bar.get_height()),xytext=(0,4),textcoords='offset points',ha='center')
ax.set(xticks=list(range(4)),xticklabels=list(names.values()),ylabel='Canonical complete-answer accuracy (%)',ylim=(0,100),title='Inspected development panel — one data-order seed')
fig.savefig(O/'absolute-answers.svg');plt.close(fig)
with (O/'absolute-answers.csv').open('x',newline='') as f:
    w=csv.writer(f);w.writerow(['path','items','correct','canonical_F1','alias_correct','unsupported','errors','capped','actions'])
    for name,v in h['tables'].items():w.writerow([name,v['items'],v['canonical_correct'],v['mean_canonical_F1'],v['alias_correct'],v['unsupported'],v['execution_errors'],v['capped'],v['generated_actions']])
labels=['Selection1 direct − format1 direct (primary)','Selection1 generated IDs − format1 direct','Selection1 empty replay − format1 empty replay','Format1 direct − format0.1 direct']
fig,ax=plt.subplots(figsize=(10,4.2),layout='constrained')
for i,v in enumerate(h['contrasts']):
    x=100*v['paired_gain'];lo,hi=[100*z for z in v['CI95']]
    ax.errorbar(x,i,xerr=[[max(0,x-lo)],[max(0,hi-x)]],fmt='o',capsize=4,color='#0072B2' if i==0 else '#666666')
    ax.annotate(f'{x:+.1f} [{lo:+.1f}, {hi:+.1f}]',(hi,i),xytext=(7,0),textcoords='offset points',va='center',fontsize=9)
ax.axvline(0,lw=1,color='#555555');ax.invert_yaxis();ax.margins(x=.5,y=.2)
ax.set(yticks=list(range(4)),yticklabels=labels,xlabel='Answer gain, percentage points (paired 95% interval)',title='Coefficient-matched control — developmental contrasts')
fig.savefig(O/'paired-answer-contrasts.svg');plt.close(fig)
with (O/'paired-answer-contrasts.csv').open('x',newline='') as f:
    w=csv.writer(f);w.writerow(['treatment','control','gain','CI95_low','CI95_high','corrected','regressed'])
    for v in h['contrasts']:w.writerow([v['treatment'],v['control'],v['paired_gain'],*v['CI95'],v['corrected'],v['regressed']])
labels={'B0':'Untouched B0','evidence_direct':'Format0.1','evidence_select':'Selection0.1','selector1':'Selection1.0'}
fig,ax=plt.subplots(figsize=(7.8,4.1),layout='constrained')
for i,(name,v) in enumerate(sorted(o['paired'].items())):
    x=v['mean_nll_change'];lo,hi=v['CI95']
    ax.errorbar(x,i,xerr=[[max(0,x-lo)],[max(0,hi-x)]],fmt='o',capsize=4,color='#0072B2')
    ax.annotate(f'{x:+.3f} [{lo:+.3f}, {hi:+.3f}]',(hi,i),xytext=(7,0),textcoords='offset points',va='center',fontsize=9)
ax.axvline(0,lw=1,color='#555555');ax.invert_yaxis();ax.margins(x=.55,y=.2)
ax.set(yticks=list(range(4)),yticklabels=[labels[k] for k in sorted(o['paired'])],xlabel='Format1 minus reference NLL, nats/token (positive is worse)',title='Same ordinary256 pages — conditional paired intervals')
fig.savefig(O/'ordinary-NLL-cost.svg');plt.close(fig)
with (O/'ordinary-NLL-cost.csv').open('x',newline='') as f:
    w=csv.writer(f);w.writerow(['reference','mean_nll_change','CI95_low','CI95_high'])
    for name,v in sorted(o['paired'].items()):w.writerow([name,v['mean_nll_change'],*v['CI95']])
print(json.dumps(dict(status='EXPORTED_V26_FIXED_FIGURES',files=[p.name for p in O.iterdir()])))

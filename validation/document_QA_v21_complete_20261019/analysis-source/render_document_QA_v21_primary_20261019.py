"""Focused view of the unchanged predeclared primary contrasts."""
from pathlib import Path
import json,hashlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
R=Path('/home/ubuntu/dario');F=R/'validation/document_QA_v21_complete_20261019.json';P=F.with_suffix('');r=json.loads(F.read_text());check=json.loads((P/'independent-readout-check.json').read_text());assert check['report_sha256']==hashlib.sha256(F.read_bytes()).hexdigest()
names=['post_direct_minus_direct_direct','prior_prior_minus_prior_empty','prior_prior_minus_direct_direct'];labels=['Post-answer direct − matched direct','Generated reasoning − same-model empty','Generated reasoning − matched direct']
fig,axes=plt.subplots(1,2,figsize=(12,4.4),sharey=True,sharex=True)
for ax,(seed,s) in zip(axes,r['seeds'].items()):
 for i,name in enumerate(names):
  v=s['primary_canonical_EM_contrasts'][name];point=100*v['difference_b_minus_a'];lo,hi=np.array(v['ci95'])*100
  ax.errorbar(point,i,xerr=[[point-lo],[hi-point]],fmt='o',color='#2878a5',capsize=4)
 ax.axvline(0,color='#555555',lw=.9);ax.axvline(3,color='#a86920',lw=.9,ls=':');ax.set_yticks(range(3),labels);ax.set_title('Seed '+seed);ax.set_xlabel('Canonical EM difference (percentage points)');ax.grid(axis='x',alpha=.16);ax.spines[['top','right']].set_visible(False)
axes[0].invert_yaxis();fig.suptitle('Document QA: local inference gain, overall two-seed gate not met')
fig.text(.5,.02,'20,000 paired question bootstrap draws; checkpoint-conditional, unadjusted 95% intervals.\nEach declared gate requires a point gain ≥3pp and lower95%>0 in each seed; no pooling. All512 attempts retained.',ha='center',fontsize=8)
fig.tight_layout(rect=(0,.15,1,.93))
for ext in ('png','svg'):fig.savefig(P/('primary-contrasts-focused.'+ext),dpi=170,bbox_inches='tight')
plt.close(fig)

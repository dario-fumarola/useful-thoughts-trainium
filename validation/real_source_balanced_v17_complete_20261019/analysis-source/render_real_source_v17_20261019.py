"""Completed V15 results only: all modes, errors, costs and declared contrasts."""
from pathlib import Path
import csv
import hashlib
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

R=Path('/home/ubuntu/dario')
p=R/'validation/real_source_balanced_v17_complete_20261019.json'
r=json.loads(p.read_text());out=p.with_suffix('')
check=json.loads((out/'independent-readout-check.json').read_text())
assert check['status'].startswith('PASS_ALL_PUBLIC_')
assert check['report_sha256']==hashlib.sha256(p.read_bytes()).hexdigest()

def csvfile(name,header,values):
    with (out/name).open('x',newline='') as stream:
        writer=csv.writer(stream,lineterminator='\n');writer.writerow(header);writer.writerows(values)

categories=('valid','unsupported','format_error','incomplete','execution_error')
csvfile('all-modes-accuracy-failures-and-costs.csv',
    ['model','mode','documents','correct','accuracy',*categories,'capped','answer_actions','note_actions_when_own','given_closing_tokens_when_nonraw','answer_worker_wall_sum_seconds'],
    [[m,mode,c['documents'],c['correct'],c['correct']/c['documents'],
      *[c['categories'].get(k,0) for k in categories],c['capped'],c['generated_actions'],
      r['note_costs'][m]['generated_actions'] if mode=='own' else 0,
      0 if mode=='raw' else 2*c['documents'],c['wall_seconds_sum']]
     for m,modes in r['tables'].items() for mode,c in modes.items()])
csvfile('all-likelihood-modes.csv',
    ['model','panel','mode','documents','mean_full16_logprob','mean_token_NLL','note_error_rows'],
    [[m,panel,mode,c['documents'],c['mean_span16_logprob'],c['mean_token_NLL'],c['generation_error_rows']]
     for m,panels in r['absolute_likelihood'].items() for panel,modes in panels.items() for mode,c in modes.items()])
csvfile('all-paired-contrasts.csv',
    ['seed','endpoint','contrast','treatment_minus_reference','ci95_lower','ci95_upper','documents','draws'],
    [[seed,endpoint,name,c['difference_b_minus_a'],*c['ci95'],c['documents'],c['bootstrap_draws']]
     for seed,s in r['seeds'].items() for endpoint in ['accuracy_contrasts','secondary_full16_contrasts','prospective_component_contrasts','ordinary_token_NLL_regression']
     for name,c in s[endpoint].items()])

fig,axes=plt.subplots(1,2,figsize=(14.4,7.2),sharex=True,sharey=True)
labels=['Unchanged B0 (shared)','Prior-calculation CPT','Target-only direct SFT','Target-only CoT SFT','Source-balanced direct','Source-balanced CoT']
for ax,seed in zip(axes,r['seeds']):
    models=['B0']+[f'{arm}-seed{seed}-step320' for arm in ['real_prior','target_direct','target_cot','source_direct','source_cot']]
    y=np.arange(6)
    for mode,offset,color,label in [('raw',-.17,'#2878a5','Raw'),('own',.17,'#d17828','Generated calculation')]:
        cells=[r['tables'][m][mode] for m in models]
        values=[100*c['correct']/512 for c in cells]
        ax.barh(y+offset,values,height=.28,color=color,label=label)
        for a,v,c in zip(y+offset,values,cells):ax.text(v+.6,a,str(c['correct']),va='center',fontsize=8)
    for mode,marker,color,label in [('empty','o','#444444','Empty format'),('other','x','#888888','Unrelated calculation'),('oracle','D','#111111','Supplied full trace')]:
        ax.scatter([100*r['tables'][m][mode]['correct']/512 for m in models],y,
                   marker=marker,color=color,s=25,label=label,zorder=4)
    ax.set_title('Seed '+seed);ax.set_yticks(y,labels);ax.set_xlim(0,106)
    ax.set_xlabel('Complete leading scalar correct (%)');ax.grid(axis='x',alpha=.16)
    ax.spines[['top','right']].set_visible(False)
axes[0].invert_yaxis()
h,l=axes[0].get_legend_handles_labels();fig.legend(h,l,loc='upper center',bbox_to_anchor=(.5,.925),ncol=5,frameon=False,fontsize=9)
fig.suptitle('Fresh real-document computations: all 512 attempts retained',fontsize=14)
fig.text(.5,.025,'Bar labels: correct out of 512. Shared B0; each training seed is reported separately.\n'
         'Raw: up to 32 answer actions. Own: up to 128 note + 32 answer actions; given closing tokens counted separately.\n'
         'Supplied traces include the answer. CSV retains unsupported, format, incomplete and execution-error counts.\n'
         'Source-balanced supervised arithmetic reference; different task supervision and inference budgets, not a pretraining or equal-compute claim.',
         ha='center',fontsize=8.6)
fig.tight_layout(rect=(0,.19,1,.86),w_pad=2)
for ext in ['png','svg']:fig.savefig(out/f'all-models-scalar-accuracy.{ext}',dpi=170,bbox_inches='tight')
plt.close(fig)

fig,axes=plt.subplots(1,2,figsize=(12.4,4.8),sharey=True)
names=['cot_own_minus_direct_raw','cot_own_minus_cot_empty','cot_own_minus_B0raw','direct_raw_minus_B0raw']
labels=['Balanced CoT own − balanced direct raw','CoT own − CoT empty','CoT own − B0 raw','Balanced direct raw − B0 raw']
for ax,(seed,s) in zip(axes,r['seeds'].items()):
    for y,name in enumerate(names):
        c=s['accuracy_contrasts'][name];v=100*c['difference_b_minus_a'];lo,hi=np.array(c['ci95'])*100
        ax.errorbar(v,y,xerr=[[v-lo],[hi-v]],fmt='o',color='#2878a5',capsize=3)
    ax.axvline(0,color='#777777',lw=.8);ax.axvline(3,color='#ad7531',lw=.8,ls=':')
    ax.set_yticks(range(4),labels);ax.set_title('Seed '+seed);ax.set_xlabel('Accuracy difference (percentage points)')
    ax.grid(axis='x',alpha=.16);ax.spines[['top','right']].set_visible(False)
axes[0].invert_yaxis();fig.suptitle('Declared reference contrast and controls')
fig.text(.5,.02,'20,000 paired document bootstrap draws; 95% checkpoint-conditional intervals, no pooled seeds.\n'
         'Dotted line: 3pp minimum applies to Balanced CoT own − balanced direct raw only. See JSON for the complete literal gate.',ha='center',fontsize=8.5)
fig.tight_layout(rect=(0,.12,1,.93),w_pad=1.5)
for ext in ['png','svg']:fig.savefig(out/f'declared-accuracy-contrasts.{ext}',dpi=170,bbox_inches='tight')
plt.close(fig)
for p in out.iterdir():
    if p.suffix in ['.csv','.svg']:p.write_text('\n'.join(v.rstrip() for v in p.read_text().splitlines())+'\n')
print('Rendered fixed comparison, declared contrasts and all-mode CSVs.',flush=True)

csvfile('declared-gates-by-seed.csv',
 ['seed','accuracy_gate','ordinary_quality_gate','combined_gate','ordinary_NLL_upper95','ordinary_NLL_upper_bound'],
 [[seed,s['declared_primary_gate_pass'],s['declared_ordinary_quality_gate_pass'],s['declared_combined_gate_pass'],s['ordinary_token_NLL_regression']['cot_minus_B0']['ci95'][1],.03] for seed,s in r['seeds'].items()])
fig,axes=plt.subplots(1,2,figsize=(12.8,4.6),sharey=True)
labels=['Balanced CoT − B0','Balanced direct − B0','Balanced CoT − target-only CoT','Balanced direct − target-only direct']
names=['cot_minus_B0','direct_minus_B0','cot_minus_target_cot','direct_minus_target_direct']
for ax,(seed,s) in zip(axes,r['seeds'].items()):
 for y,name in enumerate(names):
  c=s['ordinary_token_NLL_regression'][name];v=c['difference_b_minus_a'];lo,hi=c['ci95']
  ax.errorbar(v,y,xerr=[[v-lo],[hi-v]],fmt='o',color='#2878a5',capsize=3)
 ax.axvline(0,color='#777777',lw=.8);ax.axvline(.03,color='#b15b26',ls=':',lw=1.2)
 ax.set_yticks(range(len(labels)),labels);ax.set_title('Seed '+seed);ax.set_xlabel('Ordinary NLL increase (nat/token)')
 ax.grid(axis='x',alpha=.16);ax.spines[['top','right']].set_visible(False)
axes[0].invert_yaxis();fig.suptitle('Source-quality retention: fresh ordinary 256 documents')
fig.text(.5,.02,'20,000 paired document bootstrap draws; 95% checkpoint-conditional intervals. Lower is better.\n'
 'The upper interval for Balanced CoT − B0 must be at most +0.03 (dotted line), separately in each seed.',ha='center',fontsize=8.8)
fig.tight_layout(rect=(0,.14,1,.93),w_pad=1.5)
for ext in ['png','svg']:fig.savefig(out/f'ordinary-quality-guard.{ext}',dpi=170,bbox_inches='tight')
plt.close(fig)
for p in out.iterdir():
 if p.suffix in ['.csv','.svg']:p.write_text('\n'.join(v.rstrip() for v in p.read_text().splitlines())+'\n')

"""Render the completed fixed study; no model selection or new endpoints."""
from pathlib import Path
import csv
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path('/home/ubuntu/dario')
REPORT = ROOT / 'validation/real_trace_cpt_v10_complete_20261019.json'
report = json.loads(REPORT.read_text())
assert report['status'] == 'COMPLETE_TWO_SEED_REAL_DOCUMENT_STUDY_ALL_FIXED_PANELS'
out = REPORT.with_suffix('')
assert out.is_dir()
seeds = list(report['seeds'])
assert seeds == ['2026101913', '2026101914']
colors = ['#20639b', '#d2691e']

contrasts = [
    ('Training: post raw − control raw', 'primary_AST', 'post_raw_minus_raw_raw', 'nats/span'),
    ('Inference: prior own − control raw', 'primary_AST', 'prior_own_minus_raw_raw', 'nats/span'),
    ('Same model: prior own − prior raw', 'primary_AST', 'prior_own_minus_prior_raw', 'nats/span'),
    ('Content: prior own − prior empty', 'primary_AST', 'prior_own_minus_prior_empty', 'nats/span'),
    ('Content: prior own − prior other', 'primary_AST', 'prior_own_minus_prior_other', 'nats/span'),
    ('Post ordinary-text regression', 'ordinary_token_NLL_regression', 'post', 'nats/token'),
    ('Prior ordinary-text regression', 'ordinary_token_NLL_regression', 'prior', 'nats/token'),
]
with (out / 'declared-contrasts.csv').open('w', newline='') as stream:
    writer = csv.writer(stream)
    writer.writerow(['contrast', 'training_seed', 'unit', 'treatment_minus_reference', 'ci95_low', 'ci95_high'])
    for label, group, key, unit in contrasts + [
        ('Equal mode (format diagnostic): prior own − control own', 'primary_AST', 'prior_own_minus_raw_own', 'nats/span')]:
        for seed in seeds:
            value = report['seeds'][seed][group][key]
            writer.writerow([label, seed, unit, value['difference_b_minus_a'], *value['ci95']])

fig, axes = plt.subplots(1, 3, figsize=(15, 5.8), gridspec_kw={'width_ratios': [1.1, 1.1, 1]})
panels = [
    (axes[0], contrasts[:3], 'Primary and same-model contrasts', 'Full 16-token LL gain (nats; higher is better)'),
    (axes[1], contrasts[3:5], 'Content controls', 'Full 16-token LL gain (nats; higher is better)'),
    (axes[2], contrasts[5:], 'Ordinary-text guardrail', 'Token NLL increase (nats; lower is better)'),
]
for ax, entries, title, xlabel in panels:
    for i, (label, group, key, _) in enumerate(entries):
        for j, seed in enumerate(seeds):
            value = report['seeds'][seed][group][key]
            mean = value['difference_b_minus_a']
            low, high = value['ci95']
            y = i + (j - .5) * .2
            ax.hlines(y, low, high, color=colors[j], linewidth=1.3)
            ax.plot([low, high], [y, y], '|', color=colors[j], markersize=7)
            ax.plot(mean, y, 'o', color=colors[j], label=f'Seed {seed}' if i == 0 else None)
    ax.axvline(0, color='#666666', linewidth=.9)
    ax.set_yticks(range(len(entries)), [x[0].split(': ', 1)[-1] for x in entries])
    ax.invert_yaxis()
    ax.set_title(title, fontsize=11, pad=28)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.grid(axis='x', alpha=.2)
    ax.spines[['top', 'right']].set_visible(False)
axes[0].axvline(.25, color='#666666', linestyle=':', linewidth=.8)
axes[2].axvline(.02, color='#666666', linestyle=':', linewidth=.8)
axes[2].axvline(.05, color='#999999', linestyle='--', linewidth=.8)
axes[0].text(0, 1.02, '0.25 threshold applies to first two rows', transform=axes[0].transAxes, fontsize=8)
axes[2].text(0, 1.02, 'Mean ≤ 0.02; CI upper ≤ 0.05 (post arm)', transform=axes[2].transAxes, fontsize=8)
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(.5, .92), ncol=2, fontsize=8)
fig.suptitle('Single-model real-document study: both training seeds, fixed terminal checkpoints', fontsize=13)
fig.text(.5, .025,
         'AST-novel arithmetic: 256 documents. Ordinary text: separate 256. 95% paired document bootstrap CIs.\n'
         'Likelihood endpoint; no free-answer accuracy or controlled end-to-end latency claim. Seeds are not pooled.',
         ha='center', fontsize=9)
fig.tight_layout(rect=(0, .12, 1, .93), w_pad=2)
for suffix in ('svg', 'png'):
    fig.savefig(out / f'declared-contrasts.{suffix}', dpi=180, bbox_inches='tight')
plt.close(fig)

with (out / 'absolute-primary-likelihood.csv').open('w', newline='') as stream:
    writer = csv.writer(stream)
    writer.writerow(['model', 'mode', 'documents', 'mean_span16_logprob', 'mean_token_NLL', 'generation_error_rows'])
    for model, panels in report['absolute_likelihood'].items():
        for mode, cell in panels['DEV_AST256'].items():
            writer.writerow([model, mode, cell['documents'], cell['mean_span16_logprob'],
                             cell['mean_token_NLL'], cell['generation_error_rows']])
print('Rendered declared contrasts and absolute primary likelihood, without pooling seeds.')

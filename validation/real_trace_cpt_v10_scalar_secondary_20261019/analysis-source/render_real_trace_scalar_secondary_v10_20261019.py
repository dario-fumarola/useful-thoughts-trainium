"""Plot all fixed scalar modes and retain every parser/error count in CSV."""
from pathlib import Path
import csv
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

R = Path('/home/ubuntu/dario')
report = json.loads((R/'validation/real_trace_cpt_v10_scalar_secondary_20261019.json').read_text())
assert report['status'] == 'COMPLETE_ALL_SEVEN_MODELS_BOTH_FIXED_SCALAR_PANELS'
out = R/'validation/real_trace_cpt_v10_scalar_secondary_20261019'
with (out/'all-modes-accuracy-and-failures.csv').open('w', newline='') as stream:
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow(['model', 'panel', 'mode', 'documents', 'correct', 'valid', 'unsupported',
                     'format_error', 'incomplete', 'execution_error', 'capped', 'generated_actions'])
    for model, panels in report['tables'].items():
        for panel, modes in panels.items():
            for mode, cell in modes.items():
                writer.writerow([model, panel, mode, cell['documents'], cell['correct'],
                                 *[cell['categories'].get(k, 0) for k in
                                   ('valid', 'unsupported', 'format_error', 'incomplete', 'execution_error')],
                                 cell['capped'], cell['generated_actions']])

fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.3), sharex=True, sharey=True)
for ax, seed in zip(axes, ('2026101913', '2026101914')):
    models = ['B0']+[f'{arm}-seed{seed}' for arm in ('real_raw', 'real_prior', 'real_post')]
    y = np.arange(4)
    cells = [report['tables'][m]['DEV_AST256'] for m in models]
    for mode, offset, color, label in [('raw', -.16, '#20639b', 'Raw continuation'),
                                       ('own', .16, '#d2691e', 'Generated own calculation')]:
        values = [100*c[mode]['correct']/256 for c in cells]
        ax.barh(y+offset, values, height=.27, color=color, label=label)
        for position, value, cell in zip(y+offset, values, cells):
            ax.text(min(99, value+.7), position, str(cell[mode]['correct']), va='center', fontsize=8)
    for mode, marker, color, label in [('empty', 'o', '#333333', 'Empty format'),
                                      ('other', 'x', '#777777', 'Unrelated calculation'),
                                      ('oracle', 'D', '#000000', 'Supplied trace (oracle)')]:
        values = [100*c[mode]['correct']/256 for c in cells]
        ax.scatter(values, y, marker=marker, s=25, color=color,
                   facecolors='none' if mode == 'empty' else color, label=label, zorder=4)
    ax.set_title(f'Training seed {seed}')
    ax.set_yticks(y, ['Unchanged B0 (shared)', 'Raw CPT', 'Prior-calculation CPT', 'Post-calculation CPT'])
    ax.set_xlabel('Complete leading scalar correct (%)')
    ax.grid(axis='x', alpha=.18)
    ax.set_xlim(0, 105)
    ax.spines[['top', 'right']].set_visible(False)
axes[0].invert_yaxis()
handles, labels = axes[0].get_legend_handles_labels()
lookup = dict(zip(labels, handles))
legend_order = ['Raw continuation','Generated own calculation','Empty format','Unrelated calculation','Supplied trace (oracle)']
fig.legend([lookup[k] for k in legend_order], legend_order, loc='upper center', bbox_to_anchor=(.5,.91), ncol=5, frameon=False, fontsize=8)
fig.suptitle('Secondary scalar continuation: AST-novel real-document panel, 256 items', fontsize=13)
fig.text(.5, .025, 'Numbers on bars are correct out of 256; every failure remains in the denominator. B0 is one shared reference.\n'
         'Raw uses 32 continuation actions; own uses up to 128 original note actions + 32. Supplied traces contain the result.\n'
         'Parser/error/cap counts are in the companion CSV. No general-QA, equal-compute or controlled latency claim.',
         ha='center', fontsize=8.7)
fig.tight_layout(rect=(0, .16, 1, .84), w_pad=2)
for suffix in ('png', 'svg'):
    fig.savefig(out/f'primary-scalar-accuracy.{suffix}', dpi=180, bbox_inches='tight')
plt.close(fig)
print('Rendered fixed scalar modes and complete failure-count CSV.')

# Keep generated text artifacts clean without changing data or SVG geometry.
for text_artifact in out.glob("*"):
    if text_artifact.suffix in (".csv", ".svg"):
        text_artifact.write_text("\n".join(line.rstrip() for line in text_artifact.read_text().splitlines()) + "\n")

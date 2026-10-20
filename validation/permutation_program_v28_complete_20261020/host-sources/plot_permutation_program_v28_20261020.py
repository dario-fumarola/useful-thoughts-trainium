"""Render completed fixed V28 scores; no scoring, model calls or new gates."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SEEDS = (2026102028, 2026102029)
PANELS = ('DEV_interpolation', 'DEV_composition', 'DEV_depth')
DISPLAY = {'DEV_interpolation': 'Fresh maps: interpolation',
           'DEV_composition': 'Held-out compositions (primary)',
           'DEV_depth': 'Longer programs'}
PATHS = [('direct', 'direct', 'D direct'), ('direct', 'empty', 'D empty'),
         ('pause', 'direct', 'P direct'), ('pause', 'pause', 'P pauses'),
         ('trace', 'direct', 'T direct'), ('trace', 'empty', 'T empty'),
         ('trace', 'trace', 'T trace')]
COLORS = ['#65758b', '#adb8c6', '#a07bc1', '#68409a', '#79b9b0', '#b6d6d0', '#167565']


def write_csv(path, rows):
    with path.open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save(fig, output, name):
    for ext in ('svg', 'png'):
        fig.savefig(output / (name + '.' + ext), dpi=160, bbox_inches='tight')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--readout', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    result = json.loads(args.readout.read_text())
    assert result['status'] == 'COMPLETE_FIXED_V28_ALL_TERMINAL_READOUT'
    cells = {(r['model'], r['panel'], r['mode']): r for r in result['absolute'] if not r['diagnostic']}
    for seed in SEEDS:
        for panel in PANELS:
            for arm, mode, label in PATHS:
                cell = cells[f'program-{arm}-seed{seed}-v1', panel, mode]
                assert cell['items'] == (18432 if panel == 'DEV_composition' else 2048)
    args.output.mkdir(parents=True, exist_ok=False)
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False,
                         'axes.spines.right': False, 'svg.fonttype': 'none'})
    fig, axes = plt.subplots(3, 2, figsize=(13, 11), sharey=True)
    for row, panel in enumerate(PANELS):
        for col, seed in enumerate(SEEDS):
            ax = axes[row, col]
            data = [cells[f'program-{arm}-seed{seed}-v1', panel, mode] for arm, mode, _ in PATHS]
            bars = ax.bar(range(len(data)), [100 * c['accuracy'] for c in data], color=COLORS)
            ax.bar_label(bars, labels=[f"{100*c['accuracy']:.1f}" for c in data], padding=3, fontsize=8)
            ax.set_xticks(range(len(data)), [v[2] for v in PATHS], rotation=30, ha='right')
            ax.set_ylim(0, 106)
            ax.set_yticks(range(0, 101, 20))
            ax.set_title(f"{DISPLAY[panel]} · order seed {seed}\nn={data[0]['items']:,}, paired families={data[0]['families']:,}")
            if col == 0:
                ax.set_ylabel('Complete-answer accuracy (%)')
    fig.suptitle('V28: terminal checkpoints, all original attempts retained', fontsize=14, y=1.01)
    fig.text(.5, -.01, 'D: direct training · P: trained provided pauses · T: full-trace training. Panels and seeds are separate.\nErrors, caps and unsupported answers score zero. B0 uses a smaller matched subset and is reported separately.',
             ha='center', fontsize=9)
    fig.tight_layout()
    save(fig, args.output, 'complete-answer-accuracy')

    labels = {'trace_minus_direct_model': 'T trace − D direct',
              'trace_minus_trained_pause': 'T trace − P pauses',
              'trace_minus_same_direct': 'T trace − T direct',
              'trained_direct_minus_direct_model': 'T direct − D direct',
              'trained_direct_minus_pause_direct': 'T direct − P direct'}
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    contrasts = []
    for ax, family in zip(axes, ('inference', 'training')):
        y = 0
        for seed, color in zip(SEEDS, ('#167565', '#7048a0')):
            for name, cell in result['primary_' + family][str(seed)].items():
                diff = cell['difference_b_minus_a'] * 100
                lo, hi = [v * 100 for v in cell['ci95']]
                ax.plot([lo, hi], [y, y], color=color, lw=2)
                ax.scatter([diff], [y], color=color, s=28)
                ax.text(0.99, y, f'{diff:+.1f} [{lo:+.1f}, {hi:+.1f}]',
                        ha='right', va='center', transform=ax.get_yaxis_transform(),
                        fontsize=8, bbox=dict(facecolor='white', edgecolor='none', alpha=.85))
                contrasts.append(dict(claim_family=family, seed=seed, contrast=name,
                    difference_pp=diff, lower95_pp=lo, upper95_pp=hi,
                    fixed=cell['fixed'], harmed=cell['harmed'],
                    meets_original_rule=cell['meets_practical_and_interval_rule']))
                y += 1
            y += 0.6
        ordered = [f"{labels[c['contrast']]}\nseed {c['seed']}" for c in contrasts if c['claim_family'] == family]
        positions = []
        offset = 0
        for seed in SEEDS:
            n = len(result['primary_' + family][str(seed)])
            positions.extend(range(offset, offset + n))
            offset += n
        # Match the explicit 0.6 separator used while drawing the second seed.
        n_first = len(result['primary_' + family][str(SEEDS[0])])
        positions = [p if i < n_first else p + .6 for i, p in enumerate(positions)]
        ax.set_yticks(positions, ordered, fontsize=9)
        ax.invert_yaxis()
        ax.axvline(0, color='#888888', lw=1)
        ax.axvline(3, color='#c48239', ls='--', lw=1)
        ax.set_xlabel('Accuracy difference (percentage points)')
        ax.set_title(f"{family.capitalize()} gate: {'PASS' if result['gates'][family] else 'FAIL'}")
        ax.margins(x=.25, y=.15)
    fig.suptitle('Primary held-composition contrasts · separate training and inference claims', fontsize=13)
    fig.text(.5, -.02, '95% paired family-bootstrap intervals, conditional on the checkpoints. Dashed line: original +3pp threshold.\nEach claim requires every original comparison in both seeds to meet its declared rule. A failed gate does not prove zero effect.',
             ha='center', fontsize=9)
    fig.tight_layout()
    save(fig, args.output, 'primary-contrasts')

    # These four paths are fixed by the primary comparison, rather than
    # selected after observing their scores. Never join depths across panels.
    depth_paths = [('direct', 'direct', 'D direct', '#65758b', 's', '-'),
                   ('pause', 'pause', 'P pauses', '#68409a', 'd', '-'),
                   ('trace', 'direct', 'T direct', '#1f8b9e', 'x', '--'),
                   ('trace', 'trace', 'T trace', '#167565', 'o', '-')]
    fig, axes = plt.subplots(3, 2, figsize=(12, 10), sharey=True)
    for row, panel in enumerate(PANELS):
        for col, seed in enumerate(SEEDS):
            ax = axes[row, col]
            for arm, mode, label, color, marker, style in depth_paths:
                model = f'program-{arm}-seed{seed}-v1'
                strata = result['strata'][f'{model}/{panel}/{mode}']['depth']
                depths = sorted(map(int, strata))
                assert sum(strata[str(d)]['items'] for d in depths) == cells[model, panel, mode]['items']
                ax.plot(depths, [100 * strata[str(d)]['accuracy'] for d in depths],
                        label=label, color=color, marker=marker, ls=style, lw=1.7)
            ax.set_xticks(depths, [f"{d}\n(n={strata[str(d)]['items']:,})" for d in depths], fontsize=8)
            ax.set_ylim(0, 102)
            ax.set_yticks(range(0, 101, 20))
            ax.set_title(f'{DISPLAY[panel]} · seed {seed}')
            ax.set_xlabel('Program steps (question counts)')
            if col == 0:
                ax.set_ylabel('Complete-answer accuracy (%)')
    handles, labels_for_depth = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels_for_depth, loc='lower center', ncol=4, bbox_to_anchor=(.5, -.01))
    fig.suptitle('Depth profiles: fresh maps, new compositions and longer programs separately', fontsize=13)
    fig.text(.5, -.04, 'Descriptive depth-specific points; no pooling across panels or seeds. All errors/caps/unsupported answers remain.\nPrimary uncertainty and decisions use the separately reported paired-family contrasts.', ha='center', fontsize=9)
    fig.tight_layout()
    save(fig, args.output, 'depth-profiles')
    write_csv(args.output / 'primary-contrasts.csv', contrasts)
    absolute = [{k: (json.dumps(v, sort_keys=True) if isinstance(v, dict) else v)
                 for k, v in row.items()} for row in result['absolute']]
    write_csv(args.output / 'absolute-answers-and-costs.csv', absolute)
    print(json.dumps(dict(status='RENDERED_COMPLETED_FIXED_READOUT', output=str(args.output),
                         new_scoring_rules=False, new_model_calls=0)))


if __name__ == '__main__':
    main()

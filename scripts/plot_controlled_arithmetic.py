"""Plot the complete B0 synthetic diagnostic; consumes saved public scores only."""
from pathlib import Path
import argparse
import hashlib
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path,
        default=ROOT/'validation/controlled_arithmetic_DEV256_B0_20261018.json')
    parser.add_argument('--output-dir', type=Path, default=ROOT/'figures')
    args = parser.parse_args()
    report = json.loads(args.input.read_text())
    assert report['design']['items'] == 256
    assert report['counts']['attempts'] == report['counts']['caps'] == 256
    assert report['counts']['scoring_errors'] == report['counts']['generation_errors'] == 0
    digest = hashlib.sha256(args.input.read_bytes()).hexdigest()
    plt.rcParams.update({'font.size': 10, 'svg.hashsalt': 'useful-thoughts-arithmetic-B0'})
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3), sharex=True)
    panels = [
        ('Supplied partial work\nFinal operation and answer omitted',
         ['partial_correct_minus_raw', 'partial_correct_minus_empty',
          'partial_correct_minus_partial_wrong'],
         ['vs raw', 'vs empty format', 'vs wrong partial']),
        ('Self-generated notes\nEvery note reached the 128-action cap',
         ['own_minus_raw', 'own_minus_empty', 'own_minus_other'],
         ['vs raw', 'vs empty format', 'vs unrelated note'])]
    styles = [('I3', 'Re-anchored reader', '#23689b', -.12),
              ('I6w0.375', 'Fixed logit mix (3/8)', '#b55f16', .12)]
    for axis, (title, contrasts, labels) in zip(axes, panels):
        axis.axvline(0, color='#555555', linewidth=.8, zorder=1)
        for path, label, color, shift in styles:
            for position, contrast in enumerate(contrasts):
                value = report['groups'][path]['fullspan_ll']['contrasts'][contrast]
                mean = value['difference_b_minus_a']; low, high = value['ci95']
                assert low <= mean <= high
                axis.errorbar(mean, position + shift,
                    xerr=[[mean-low], [high-mean]], fmt='o', markersize=5,
                    capsize=3, linewidth=1.6, color=color,
                    label=label if position == 0 else None, zorder=3)
        axis.set(title=title, yticks=range(3), yticklabels=labels,
                 xlabel='Full-span log-likelihood difference (nats)',
                 xlim=(-4.5, 8.6), ylim=(2.6, -.6))
        axis.grid(axis='x', color='#dddddd', linewidth=.5)
        for side in ('top', 'right', 'left'):
            axis.spines[side].set_visible(False)
        axis.tick_params(axis='y', length=0)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(.52, .88),
               ncol=2, frameon=False)
    fig.suptitle('Frozen B0: 256 fixed synthetic arithmetic expressions', y=.99, fontsize=13)
    fig.text(.5, .055,
        'Paired 95% document-bootstrap intervals. Generated-note benefit is concentrated in one grammar family.\n'
        'Secondary diagnostic; generated-answer accuracy and real-document gains are not established.',
        ha='center', va='bottom', fontsize=9)
    fig.subplots_adjust(top=.69, bottom=.25, left=.15, right=.98, wspace=.62)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.output_dir/'controlled_arithmetic_B0_20261018'
    fig.savefig(stem.with_suffix('.svg'), metadata={
        'Title': 'Controlled arithmetic B0 reader diagnostic',
        'Description': f'Public input SHA256 {digest}; all256 fixed items, full-span contrasts.',
        'Date': None})
    fig.savefig(stem.with_suffix('.png'), dpi=180, metadata={
        'Description': f'Public input SHA256 {digest}; all256 fixed items.'})
    plt.close(fig)
    print(json.dumps({'input_sha256': digest, 'svg': str(stem.with_suffix('.svg')),
                      'png': str(stem.with_suffix('.png')), 'contrasts_plotted': 12}))


if __name__ == '__main__':
    main()

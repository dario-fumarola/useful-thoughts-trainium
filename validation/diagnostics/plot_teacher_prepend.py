"""Plot the closed TRAIN16 external-teacher contrasts; no new statistics or inference.

Run from the repository root with its existing Python environment. The exact
public report SHA is fixed below. Error bars are its paired-document percentile
95% intervals (5000 draws, seed17), not intervals across training seeds.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT/'validation/teacher_prepend_train16_20261017.json'
REPORT_SHA256 = '2169dae01c5ec396960a50c5244b13eaaab8e61650e024d4dd86ef4cc480adf9'
OUTPUT = ROOT/'validation/figures/teacher_prepend_train16_20261017.svg'
BASELINES = ('raw', 'empty', 'filler', 'shuffled', 'other')
SERIES = (
    ('prefix_only', 'Prefix-only teacher · no future', '#1f77b4', 'o', -.13),
    ('hindsight', 'Hindsight teacher · visible future, TRAIN labels only', '#d97706', 's', .13),
)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    if sha(REPORT) != REPORT_SHA256:
        raise ValueError('Closed public report changed; inspect before plotting')
    report = json.loads(REPORT.read_text())
    assert report['schema'] == 'teacher-prepend-TRAIN16-result-v1'
    assert report['status'] == 'complete' and report['split'] == 'train'
    assert (report['documents'], report['notes'], report['score_rows'], report['failures']) == (16, 32, 176, 0)
    assert report['reader_unchanged_B0'] and report['all16_primary']
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'svg.fonttype': 'none', 'svg.hashsalt': 'teacher-train16-fixed-v1'})
    fig, ax = plt.subplots(figsize=(9.4, 5.2))
    fig.subplots_adjust(left=.19, right=.98, top=.69, bottom=.24)
    plotted = []
    maximum_position_error = 0.
    for condition, label, color, marker, offset in SERIES:
        cells = [report['paired_document_contrasts'][f'{condition}/own_minus_{b}/K16'] for b in BASELINES]
        for cell in cells:
            assert cell['documents'] == cell['items'] == 16
            assert cell['bootstrap_draws'] == 5000 and cell['bootstrap_seed'] == 17
            assert all(math.isfinite(v) for v in [cell['difference_b_minus_a'], *cell['ci95']])
        mean = np.asarray([cell['difference_b_minus_a'] for cell in cells])
        low = np.asarray([cell['ci95'][0] for cell in cells])
        high = np.asarray([cell['ci95'][1] for cell in cells])
        assert np.all(low <= mean) and np.all(mean <= high)
        y = np.arange(len(BASELINES))+offset
        artists = ax.errorbar(mean, y, xerr=[mean-low, high-mean], fmt=marker,
                              color=color, markersize=5.5, capsize=3, elinewidth=1.5,
                              label=label, zorder=3)
        assert np.array_equal(artists.lines[0].get_xdata(), mean)
        for i, segment in enumerate(artists.lines[2][0].get_segments()):
            error = float(np.max(np.abs(segment-np.array([[low[i], y[i]], [high[i], y[i]]]))))
            maximum_position_error = max(maximum_position_error, error)
            assert error < 1e-14
        plotted.append(dict(condition=condition, contrasts=[dict(baseline=b, mean=float(m), ci95=[float(l), float(h)])
                       for b, m, l, h in zip(BASELINES, mean, low, high)]))
    ax.axvline(0, color='#777777', linewidth=1, zorder=1)
    ax.set_yticks(np.arange(len(BASELINES)), [f'Own − {b}' for b in BASELINES])
    ax.invert_yaxis()
    ax.set_xlim(-1.65, 6.0)
    ax.set_xlabel('K16 continuation log-likelihood difference (nats; higher is better)', labelpad=10)
    ax.grid(axis='x', color='#e6e6e6', linewidth=.7, zorder=0)
    ax.spines[['top', 'right']].set_visible(False)
    fig.suptitle('External teacher notes · fixed TRAIN16', x=.19, y=.97, ha='left', fontsize=14, fontweight='bold')
    fig.text(.19, .905, 'Unchanged B0 prepend reader · all 16 documents retained', fontsize=10.5)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper left', bbox_to_anchor=(.18, .865), frameon=False, fontsize=10, handletextpad=.6)
    fig.text(.19, .105, 'Points: paired means. Bars: 95% document-bootstrap intervals (5000 draws; seed 17).', fontsize=9)
    fig.text(.19, .025, 'Visible-future gains concern TRAIN label generation, not deployed prefix-only utility.\nLikelihood gain ≠ mathematical correctness.', fontsize=9, color='#444444', linespacing=1.45)
    description = json.dumps(dict(report_sha256=REPORT_SHA256, source_report=str(REPORT.relative_to(ROOT)),
        all_documents=16, horizon=16, plotted_values=plotted,
        interpretation='Hindsight exposes future16 for TRAIN labels only. Likelihood gain is not mathematical correctness.'), sort_keys=True)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, metadata={'Title': 'External teacher TRAIN16 K16 paired content contrasts',
                                 'Description': description, 'Date': None})
    png = OUTPUT.with_suffix('.png')
    fig.savefig(png, dpi=180, metadata={'Description': description})
    plt.close(fig)
    print(json.dumps(dict(source_report_sha256=REPORT_SHA256, plot_code_sha256=sha(Path(__file__)),
                         svg_sha256=sha(OUTPUT), png_sha256=sha(png),
                         contrasts=10, maximum_errorbar_coordinate_error=maximum_position_error)))


if __name__ == '__main__':
    main()

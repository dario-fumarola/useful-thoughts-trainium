"""Exploratory position decomposition of already released original-token scores."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from weekend_thinking.artifacts import sha256_file
from weekend_thinking.stats import cluster_bootstrap

BINS = ((1, 16), (17, 32), (33, 64), (65, 128), (129, 256),
        (257, 384), (385, 511), (512, 527))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence', type=Path, required=True)
    parser.add_argument('--cohort', required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--control', default='B0')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--plot', action='store_true')
    args = parser.parse_args()
    score_file = args.evidence/'original-token-scores.npz'
    readout = json.loads((args.evidence/'readout.json').read_text())
    digest = sha256_file(score_file)
    if digest != readout['public_scores_sha256']:
        raise ValueError('The original score archive changed')
    with np.load(score_file, allow_pickle=False) as archive:
        prefix = args.cohort+'__' if args.cohort+'__B0' in archive else ''
        delta = (archive[prefix+args.model].astype(np.float64)
                 - archive[prefix+args.control].astype(np.float64))
        docs = archive[prefix+'document_ids'].tolist()
    if (delta.shape != (len(docs), 527) or len(set(docs)) != len(docs)
            or not np.isfinite(delta).all()):
        raise ValueError('Complete paired original windows and unique documents required')
    values = np.column_stack([delta[:, first-1:last].mean(1) for first, last in BINS])
    result = cluster_bootstrap(values, docs, unit='unique source document',
                               repeats=5000, seed=17)
    widths = np.array([last-first+1 for first, last in BINS])
    if (widths.sum() != 527 or not np.isclose(widths @ result['mean']/527,
                                            delta.mean(), rtol=0, atol=1e-13)):
        raise ValueError('The context decomposition must preserve the original aggregate')
    data = dict(status='EXPLORATORY_CONTEXT_POSITION_DECOMPOSITION',
        source_scores_sha256=digest, cohort=args.cohort, model=args.model,
        control=args.control,
        documents=len(docs), full_window_gain_nats_per_token=float(delta.mean()),
        bins=[dict(context_tokens_first=first, context_tokens_last=last,
            target_positions=last-first+1, gain_nats_per_token=float(result['mean'][i]),
            CI95=result['CI95'][i].tolist()) for i, (first, last) in enumerate(BINS)],
        statistics=result['metadata'], new_model_calls=0,
        scope='Exploratory decomposition of completed results, not a new gate or causal explanation. Context counts reset at each sampled original window; original absolute positions vary. Bin intervals are not simultaneous intervals.')
    args.output.write_text(json.dumps(data, indent=2)+'\n')
    if args.plot:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        plt.rcParams.update({'svg.hashsalt': 'original-context-position-gains', 'font.size': 10})
        centers = np.array([(first+last)/2 for first, last in BINS])
        means, ci = result['mean'], result['CI95']
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), constrained_layout=True)
        for ax, subset in zip(axes, (np.arange(8), np.arange(4, 8))):
            ax.axhline(0, color='0.5', lw=.8)
            ax.errorbar(centers[subset], means[subset],
                xerr=(widths[subset]-1)/2,
                yerr=np.array([means-ci[:, 0], ci[:, 1]-means])[:, subset],
                fmt='o', color='#236a8f', capsize=3)
            ax.set(xlabel='Available in-window context tokens',
                   ylabel='Gain (nats / target token)')
            ax.grid(axis='y', alpha=.2)
        axes[0].set_xscale('log', base=2)
        axes[0].set_title('All context bins')
        axes[1].set_title('Longer-context detail')
        fig.suptitle(f'{args.model} versus {args.control}\n'
                     f'{args.cohort}, {len(docs):,} source documents')
        fig.savefig(args.output.with_suffix('.png'), dpi=180)
        fig.savefig(args.output.with_suffix('.pdf'), metadata={'CreationDate': None})
        plt.close(fig)
    print(json.dumps(dict(output=str(args.output), documents=len(docs),
                         original_full_window_gain=float(delta.mean()))))


if __name__ == '__main__':
    main()

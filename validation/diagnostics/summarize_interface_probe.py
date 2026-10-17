"""Summarize a complete reader-interface pilot, paired by document."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import numpy as np


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def estimate(values):
    values=np.asarray(values,dtype=np.float64)
    indices=np.random.default_rng(17).integers(0,len(values),(5000,len(values)))
    lo,hi=np.percentile(values[indices].mean(axis=1),[2.5,97.5])
    return dict(items=len(values),mean_nats=float(values.mean()),
                median_nats=float(np.median(values)),positive_items=int((values>0).sum()),
                ci95_nats=[float(lo),float(hi)])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--figure',type=Path)
    args=parser.parse_args()
    protocol=json.loads((args.input/'protocol.json').read_text())
    summary=json.loads((args.input/'summary.json').read_text())
    assert summary['status']=='complete' and summary['failures']==0
    assert summary['scores_sha256']==digest(args.input/'scores.jsonl')
    assert summary['protocol_sha256']==digest(args.input/'protocol.json')
    rows=[json.loads(line) for line in (args.input/'scores.jsonl').read_text().splitlines()]
    ids=protocol['selection']['item_ids']
    interfaces=list(protocol['interfaces'])
    keys=[('raw',name) for name in ('raw','repeat32','repeat64')]
    keys.extend((name,mode) for name in interfaces for mode in protocol['modes'])
    expected={(key,interface,mode) for key in ids for interface,mode in keys}
    index={(r['item_id'],r['interface'],r['mode']):r for r in rows}
    assert len(rows)==len(index)==len(expected) and set(index)==expected
    assert all(r['status']=='ok' and len(r['token_log_likelihoods'])==16 and
               all(math.isfinite(v) for v in r['token_log_likelihoods']) for r in rows)
    output=dict(experiment_id=protocol['experiment_id'],status='complete',items=len(ids),
        rows=len(rows),failures=0,protocol_sha256=digest(args.input/'protocol.json'),
        scores_sha256=digest(args.input/'scores.jsonl'),
        B0_reference_max_token_error=summary['maximum_B0_reference_token_error'],
        design=protocol['purpose'],selection_rule=protocol['selection_rule'],
        candidates_per_document=1,horizons=[1,16],
        uncertainty='5000 paired document bootstrap draws, seed17, percentile95%; conditional on fixed B0/notes; exploratory adaptive development',
        limitations=[protocol['limitations'],'32 documents, not an independent confirmatory set',
                    'Repeated-prefix raw controls differ from same-format empty; report both',
                    'No training or deployed-policy improvement established'],groups={})
    for group in ['all',*sorted(set(protocol['strata'].values()))]:
        selected=[key for key in ids if group=='all' or protocol['strata'][key]==group]
        result={}
        for horizon in (1,16):
            def values(interface,mode):
                return np.array([math.fsum(index[(key,interface,mode)]['token_log_likelihoods'][:horizon])
                                 for key in selected],dtype=np.float64)
            for interface in interfaces:
                reanchor=protocol['interfaces'][interface][1]
                raw_repeat=f'repeat{reanchor}' if reanchor else 'raw'
                contrasts={'empty_minus_raw':values(interface,'empty')-values('raw','raw'),
                    'own_minus_raw':values(interface,'own')-values('raw','raw'),
                    'empty_minus_raw_repeat':values(interface,'empty')-values('raw',raw_repeat),
                    'own_minus_raw_repeat':values(interface,'own')-values('raw',raw_repeat)}
                contrasts.update({f'own_minus_{mode}':values(interface,'own')-values(interface,mode)
                                  for mode in ('empty','filler','shuffled','other')})
                result[f'{interface}/K{horizon}']=dict(
                    absolute_means={mode:float(values(interface,mode).mean()) for mode in protocol['modes']},
                    raw_mean=float(values('raw','raw').mean()),raw_repeat_mean=float(values('raw',raw_repeat).mean()),
                    contrasts={name:estimate(v) for name,v in contrasts.items()})
        output['groups'][group]=result
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(output,stream,indent=2,allow_nan=False);stream.write('\n')
    if args.figure:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,axes=plt.subplots(1,2,figsize=(10,4.2),layout='constrained')
        labels=[name.replace('_',' ') for name in interfaces]
        for ax,contrast,title in zip(axes,['empty_minus_raw','own_minus_empty'],
                                    ['Empty-format gain over raw prefix','Own thought gain over same-format empty']):
            estimates=[output['groups']['all'][f'{name}/K1']['contrasts'][contrast] for name in interfaces]
            means=np.array([r['mean_nats'] for r in estimates])
            lower=np.array([r['ci95_nats'][0] for r in estimates])
            upper=np.array([r['ci95_nats'][1] for r in estimates])
            ax.errorbar(means,np.arange(len(labels)),xerr=[means-lower,upper-means],fmt='o',capsize=3)
            ax.axvline(0,color='gray',lw=1)
            ax.set_yticks(np.arange(len(labels)),labels)
            ax.invert_yaxis();ax.set_xlabel('K1 log-likelihood difference (nats)');ax.set_title(title)
        fig.suptitle('B0 reader-interface pilot · fixed DEV32 · original one-sample notes\nPaired document intervals; exploratory, no training result',fontsize=11)
        args.figure.parent.mkdir(parents=True,exist_ok=True)
        fig.savefig(args.figure)
    for interface in interfaces:
        c=output['groups']['all'][f'{interface}/K1']['contrasts']
        print(interface,json.dumps({name:c[name] for name in ('empty_minus_raw','own_minus_empty','own_minus_other')}))


if __name__=='__main__':
    main()

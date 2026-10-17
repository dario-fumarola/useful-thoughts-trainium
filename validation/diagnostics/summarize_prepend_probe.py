"""Paired analysis of the complete fixed-note reader ordering experiment."""
import argparse
import json
import math
from pathlib import Path

from validation.diagnostics.summarize_interface_probe import digest, estimate


def summarize(directory):
    p=json.loads((directory/'protocol.json').read_text())
    run=json.loads((directory/'summary.json').read_text())
    assert run['status']=='complete' and run['failures']==0
    assert run['scores_sha256']==digest(directory/'scores.jsonl')
    assert run['protocol_sha256']==digest(directory/'protocol.json')
    rows=[json.loads(line) for line in (directory/'scores.jsonl').read_text().splitlines()]
    ids=p['selection']['item_ids'];modes=['empty','own','filler','shuffled','other']
    expected={(k,'raw',m) for k in ids for m in ['raw','repeat512']}
    expected|={(k,i,m) for k in ids for i in p['interfaces'] for m in modes}
    index={(r['item_id'],r['interface'],r['mode']):r for r in rows}
    assert len(rows)==len(index)==len(expected) and set(index)==expected
    assert all(r['status']=='ok' and len(r['token_log_likelihoods'])==16 and all(math.isfinite(v) for v in r['token_log_likelihoods']) for r in rows)
    report=dict(experiment_id=p['experiment_id'],status='complete',items=len(ids),rows=len(rows),failures=0,
        protocol_sha256=digest(directory/'protocol.json'),scores_sha256=digest(directory/'scores.jsonl'),
        original_reference_max_token_error=run['maximum_reference_error'],
        candidates_per_document=1,original_saved_note_cap=1024,compute=p['compute'],
        reader_contexts=dict(original='header+prefix+open+note+close',prepend_plain='note+two-newline separator+full prefix',
            prepend_labeled='Working notes: label+note+Passage: label+full prefix',reanchor512='header+prefix+open+note+close+full prefix'),
        uncertainty='5000 paired document bootstrap draws, seed17, percentile95%; fixed B0/notes, adaptive exploratory development',
        limitations=[p['limitation'],'Original saved G1 notes/cap1024; not the fresh G16/cap256 population.',
            'Reader ordering changes only; no new writer or learning.',
            'Whole-document and candidate selection uncertainty are not training-seed uncertainty.'],groups={})
    for group in ['all',*sorted(set(p['strata'].values()))]:
        keys=[k for k in ids if group=='all' or p['strata'][k]==group];cells={}
        for h in p['horizons']:
            values=lambda i,m:[math.fsum(index[k,i,m]['token_log_likelihoods'][:h]) for k in keys]
            for i in p['interfaces']:
                own,empty=values(i,'own'),values(i,'empty')
                baseline='repeat512' if i=='reanchor512' else 'raw'
                comparisons=dict(empty_minus_raw=(empty,values('raw','raw')),own_minus_raw=(own,values('raw','raw')),
                    empty_minus_matched_raw=(empty,values('raw',baseline)),own_minus_matched_raw=(own,values('raw',baseline)))
                comparisons.update({f'own_minus_{m}':(own,values(i,m)) for m in modes if m!='own'})
                cells[f'{i}/K{h}']=dict(absolute_means={m:sum(values(i,m))/len(keys) for m in modes},
                    contrasts={n:estimate([a-b for a,b in zip(x,y)]) for n,(x,y) in comparisons.items()})
        report['groups'][group]=cells
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    with a.output.open('x') as f:json.dump(summarize(a.input),f,indent=2,allow_nan=False);f.write('\n')

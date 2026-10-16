#!/usr/bin/env python3
from pathlib import Path
import argparse, json, sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from weekend_thinking.evaluation import downstream_contrasts
from weekend_thinking.artifacts import atomic_json, canonical_hash, sha256_file


CONDITIONS = {'c_direct': ('vanilla', 'direct'), 'c_thought': ('vanilla', 'one_thought'),
              'r_direct': ('thought', 'direct'), 'r_thought': ('thought', 'one_thought')}


def read_final_conditions(paths):
    rows, summaries = {}, {}
    for name, (arm, mode) in CONDITIONS.items():
        path = paths[name]
        summary = json.loads(path.with_suffix('.summary.json').read_text())
        condition, identity = summary.get('condition', {}), summary.get('model_identity', {})
        if (summary.get('qualification') is not False or summary.get('scores_sha256') != sha256_file(path)
                or condition.get('arm') != arm or condition.get('mode') != mode
                or any(condition.get(key) != identity.get(key) for key in ('arm', 'seed', 'spec_hash', 'step'))
                or not summary.get('protocol_sha256') or not summary.get('scoring_manifest_sha256')
                or not summary.get('inputs_sha256', {}).get('frozen_protocol')
                or summary.get('inputs_sha256', {}).get('scoring_manifest') != summary.get('scoring_manifest_sha256')):
            raise ValueError(f'Scored condition binding mismatch: {name}')
        scored = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        if not scored or len(scored) != summary.get('items'):
            raise ValueError(f'Scored item count mismatch: {name}')
        for row in scored:
            if (not str(row.get('item_id', '')).startswith('gsm8k:test:')
                    or row.get('condition') != condition or row.get('model_identity_sha256') != canonical_hash(identity)
                    or row.get('protocol_sha256') != summary['protocol_sha256']
                    or row.get('scoring_manifest_sha256') != summary['scoring_manifest_sha256']):
                raise ValueError(f'Scored row provenance mismatch: {name}')
        rows[name], summaries[name] = scored, summary
    reference = summaries['c_direct']
    for summary in summaries.values():
        for key in ('seed', 'spec_hash', 'step', 'order_hash', 'base_assets_sha256'):
            if not reference['model_identity'].get(key) or summary['model_identity'].get(key) != reference['model_identity'][key]:
                raise ValueError(f'Paired conditions differ in {key}')
        if (summary['scoring_manifest_sha256'] != reference['scoring_manifest_sha256'] or
                summary.get('inputs_sha256', {}).get('frozen_protocol') !=
                reference.get('inputs_sha256', {}).get('frozen_protocol')):
            raise ValueError('Paired conditions use different frozen evaluation or gold manifests')
    for mode in ('direct', 'thought'):
        if summaries[f'c_{mode}']['protocol_sha256'] != summaries[f'r_{mode}']['protocol_sha256']:
            raise ValueError('C/R inference protocols differ within a mode')
    for arm in ('c', 'r'):
        if summaries[f'{arm}_direct']['model_identity'] != summaries[f'{arm}_thought']['model_identity']:
            raise ValueError('Direct/thought conditions must use identical checkpoints within each arm')
    return rows, summaries


def main():
    p=argparse.ArgumentParser(description='All predeclared downstream contrasts, per training seed.')
    for flag in ('c-direct','c-thought','r-direct','r-thought'):
        p.add_argument('--'+flag,type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--repeats',type=int,default=5000)
    p.add_argument('--qualification',action='store_true',help='Only GSM8K TRAIN interface checks')
    a=p.parse_args()
    paths={name:getattr(a,name) for name in CONDITIONS}
    if a.qualification:
        rows={name:[json.loads(x) for x in path.read_text().splitlines() if x.strip()]
              for name,path in paths.items()}
        if any(not str(row.get('item_id','')).startswith('gsm8k:train:') for values in rows.values() for row in values):
            p.error('Qualification comparisons accept TRAIN items only')
        provenance=dict(qualification=True)
    else:
        rows,summaries=read_final_conditions(paths)
        provenance=dict(qualification=False,conditions={name:dict(
            model_identity=summaries[name]['model_identity'],
            protocol_sha256=summaries[name]['protocol_sha256'],
            scores_sha256=summaries[name]['scores_sha256'],
            summary_sha256=sha256_file(paths[name].with_suffix('.summary.json')))
            for name in CONDITIONS})
    result=downstream_contrasts(*[rows[name] for name in CONDITIONS],repeats=a.repeats)
    provenance['analysis_source_sha256']={name:sha256_file(Path(__file__).resolve().parents[1]/name)
        for name in ('scripts/summarize_downstream.py','weekend_thinking/evaluation.py','weekend_thinking/stats.py')}
    result['provenance']=provenance
    atomic_json(a.out,result);print(json.dumps(result,indent=2))
if __name__=='__main__':main()

#!/usr/bin/env python3
"""Score pre-existing outputs; never generates using gold answers."""
from pathlib import Path
import argparse, json, sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from weekend_thinking.evaluation import score_predictions
from weekend_thinking.artifacts import atomic_bytes, atomic_json, canonical_hash, sha256_file
from train.evaluation_integrity import require_numeric_scoring_manifest, validate_numeric_predictions


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--gold',type=Path,required=True)
    p.add_argument('--predictions',type=Path,action='append',required=True,
                   help='Original generation shard; repeat to cover every official item')
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--qualification',action='store_true',help='TRAIN-only prompt/interface checks')
    p.add_argument('--frozen-protocol',type=Path)
    p.add_argument('--scoring-manifest',type=Path)
    a=p.parse_args()
    read=lambda path:[json.loads(x) for x in path.read_text().splitlines() if x.strip()]
    if a.qualification:
        gold=read(a.gold)
        predictions=[row for path in a.predictions for row in read(path)]
        if (any(not str(row.get('item_id','')).startswith('gsm8k:train:') for row in gold+predictions)
                or any(row.get('split','train') != 'train' for row in gold+predictions)):
            p.error('Qualification scoring accepts GSM8K TRAIN items only')
        binding=dict(qualification=True)
    else:
        if a.scoring_manifest is None:
            p.error('Final scoring requires --scoring-manifest and --frozen-protocol')
        frozen,manifest=require_numeric_scoring_manifest(a.frozen_protocol,a.scoring_manifest,a.gold)
        predictions,runs=[],[]
        for path in a.predictions:
            projected,run=validate_numeric_predictions(frozen,manifest,path)
            predictions.extend(projected);runs.append(run)
        if len({canonical_hash(run['model_identity']) for run in runs}) != 1 or len({run['protocol_sha256'] for run in runs}) != 1:
            p.error('All scoring shards must use the same frozen model and protocol')
        gold=read(a.gold)  # First gold parsing in this process, strictly after all gates.
        binding=dict(qualification=False,scoring_manifest_sha256=sha256_file(a.scoring_manifest),
                     model_identity=runs[0]['model_identity'],protocol_sha256=runs[0]['protocol_sha256'],
                     condition=dict(arm=runs[0]['model_identity']['arm'],seed=runs[0]['model_identity']['seed'],
                                    mode=runs[0]['protocol']['mode'],
                                    spec_hash=frozen['training_spec']['spec_hash'],
                                    step=runs[0]['model_identity'].get('step',0)))
    rows,summary=score_predictions(gold,predictions)
    if not a.qualification:
        for row in rows:
            row.update(condition=binding['condition'],protocol_sha256=binding['protocol_sha256'],
                       model_identity_sha256=canonical_hash(binding['model_identity']),
                       scoring_manifest_sha256=binding['scoring_manifest_sha256'])
    summary.update(binding)
    atomic_bytes(a.out, ''.join(json.dumps(x)+'\n' for x in rows).encode())
    summary.update(scores_sha256=sha256_file(a.out),inputs_sha256=dict(gold=sha256_file(a.gold),
        predictions=[dict(file_sha256=sha256_file(path),
                          run_receipt_sha256=sha256_file(path.with_name('run.json')) if not a.qualification else None)
                     for path in a.predictions],
        frozen_protocol=sha256_file(a.frozen_protocol) if a.frozen_protocol else None,
        scoring_manifest=sha256_file(a.scoring_manifest) if a.scoring_manifest else None))
    atomic_json(a.out.with_suffix('.summary.json'), summary)
    print(json.dumps(summary, indent=2))
if __name__=='__main__':main()

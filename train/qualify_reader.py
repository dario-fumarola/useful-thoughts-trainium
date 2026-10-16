"""CPU diagnostic of prefix-derived arithmetic notes, using released weights.

These manually constructed notes are interface controls, not generated-policy
evidence or benchmark estimates. Correct notes contain partial products only.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time

import torch

from weekend_thinking.artifacts import atomic_json, sha256_file, utc_now
from weekend_thinking.backend_contract import Rollouts
from weekend_thinking.core import Format, action_mask
from weekend_thinking.pretrained import load_base
from train.neuron_backend import NeuronCausalBackend


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-dir', type=Path, required=True)
    parser.add_argument('--format', type=Path, required=True)
    parser.add_argument('--items', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.items.read_text().splitlines() if line.strip()]
    if not 1 <= len(rows) <= 64 or any(r.get('split') != 'diagnostic_not_test' for r in rows):
        parser.error('Only 1–64 explicitly non-test reader diagnostic items are allowed.')
    if len({row['item_id'] for row in rows}) != len(rows):
        parser.error('Diagnostic item IDs must be unique.')
    args.out.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[1]
    dependencies = ['train/qualify_reader.py', 'train/neuron_backend.py',
                    'weekend_thinking/pretrained.py', 'weekend_thinking/core.py',
                    'weekend_thinking/backend_contract.py', 'weekend_thinking/artifacts.py']
    hashes = {name: sha256_file(root / name) for name in dependencies}
    report = dict(scope='CPU_released_weight_reader_interface_only', status='running',
                  started_utc=utc_now(), compute='fp32', accelerator_execution=False,
                  items=len(rows), official_test_items_scored=0,
                  items_sha256=sha256_file(args.items), format_sha256=sha256_file(args.format),
                  model_manifest_sha256=sha256_file(args.model_dir / 'ASSET_MANIFEST.json'),
                  source_hashes=hashes,
                  limitations=['Manually constructed prefix-derived notes, not learned/generated thoughts.',
                               'No downstream generalization or training benefit is established.',
                               'Full-target and K=1 are distinct diagnostics; the first token can be whitespace.'])
    atomic_json(args.out / 'report.json', report)
    model, tokenizer = load_base(args.model_dir)
    backend = NeuronCausalBackend(model)
    fmt = Format.from_dict(json.loads(args.format.read_text())['format'])
    encode = lambda text: tokenizer.encode(text, add_special_tokens=False)
    filler_token = encode(' step')[0]
    started = time.monotonic()
    results = []
    with (args.out / 'scores.jsonl').open('x') as stream, torch.no_grad():
        for row in rows:
            result = dict(item_id=row['item_id'], split=row['split'], status='scored',
                          prefix=row['prefix'], future=row['future'])
            try:
                prefix, future = encode(row['prefix']), encode(row['future'])
                helpful, incorrect = encode(row['helpful_note']), encode(row['incorrect_note'])
                contents = [helpful, incorrect, [filler_token] * len(helpful), []]
                names = ['helpful_partial_products', 'incorrect_partial_products',
                         'length_matched_filler', 'empty_format']
                cap = max(map(len, contents)) + 1
                actions = torch.full((1, 4, cap), fmt.pad_id, dtype=torch.long)
                for group, content in enumerate(contents):
                    actions[0, group, :len(content) + 1] = torch.tensor(content + [fmt.stop_id])
                rollout = Rollouts(actions=actions,
                    action_mask=action_mask(actions, fmt.stop_id, fmt.additional_stop_ids),
                    behavior_logps=torch.zeros_like(actions, dtype=torch.float32), policy_version=0)
                p, f = torch.tensor([prefix]), torch.tensor([future])
                evidence = {}
                for horizon in sorted({1, len(future)}):
                    clean = float(backend.clean_score(p, f, horizon=horizon)[0])
                    values = backend.score(p, f, rollout, fmt=fmt, horizon=horizon)[0].tolist()
                    if not all(math.isfinite(value) for value in [clean] + values):
                        raise ValueError('Nonfinite reader evidence.')
                    evidence[str(horizon)] = dict(raw=clean, **dict(zip(names, values)))
                result.update(prefix_ids=prefix, future_ids=future,
                              first_target_text=tokenizer.decode(future[:1]),
                              note_ids=dict(zip(names, contents)),
                              note_texts={name: tokenizer.decode(ids) for name, ids in zip(names, contents)},
                              helpful_incorrect_token_lengths_equal=len(helpful) == len(incorrect),
                              log_evidence=evidence)
            except Exception as error:
                result.update(status='error', error=f'{type(error).__name__}: {error}')
            results.append(result)
            stream.write(json.dumps(result, allow_nan=False) + '\n')
            stream.flush()
            print(json.dumps({'item_id': row['item_id'], 'status': result['status']}), flush=True)
    if any(sha256_file(root / name) != hashes[name] for name in hashes):
        raise RuntimeError('A diagnostic dependency changed during execution.')
    good = [row for row in results if row['status'] == 'scored']
    contrasts = {}
    for label, horizon in [('K1', lambda r: '1'), ('full_target', lambda r: str(len(r['future_ids'])))]:
        means = {}
        for a, b in [('helpful_partial_products', 'incorrect_partial_products'),
                     ('helpful_partial_products', 'length_matched_filler'),
                     ('helpful_partial_products', 'raw')]:
            differences = [r['log_evidence'][horizon(r)][a] - r['log_evidence'][horizon(r)][b] for r in good]
            means[a + '_minus_' + b] = sum(differences) / len(differences) if differences else None
        contrasts[label] = means
    report.update(status='completed' if len(good) == len(rows) else 'completed_with_errors',
                  successful_items=len(good), errors=len(rows) - len(good),
                  mean_paired_logp_differences=contrasts,
                  scoring_seconds=time.monotonic() - started,
                  scores_sha256=sha256_file(args.out / 'scores.jsonl'), finished_utc=utc_now())
    atomic_json(args.out / 'report.json', report, overwrite=True)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()

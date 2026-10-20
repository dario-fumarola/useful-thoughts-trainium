"""Check every TRAIN pack with the actual tokenizer; no model computation."""
from collections import Counter
import argparse
import hashlib
import json
from pathlib import Path

from transformers import AutoTokenizer

from train.permutation_program_data import question_from_dict
from train.permutation_program_packing import (EOS, PAUSES_PER_STEP,
    inference_context, pause_id, training_pack)


def audit(data, model_dir):
    data = Path(data)
    manifest = json.loads((data/'manifest.json').read_text())
    train = data/'TRAIN-labels.jsonl'
    assert hashlib.sha256(train.read_bytes()).hexdigest() == manifest['files'][train.name]
    rows = [json.loads(line) for line in train.read_text().splitlines()]
    tok = AutoTokenizer.from_pretrained(model_dir, local_files_only=True, trust_remote_code=False)
    pid = pause_id(tok)
    totals = {v: Counter() for v in ('direct', 'empty', 'trace', 'pause')}
    pairs = Counter()
    for i, row in enumerate(rows):
        q = question_from_dict(row['question']); expected_answer = None
        pairs[row['family_id']] += 1
        for view in totals:
            pack = training_pack(row, tok, view)
            answer = pack['labels'][pack['answer_mask']].tolist()
            if expected_answer is None:
                expected_answer = answer
            assert answer == expected_answer and answer[-1] == EOS
            active = pack['answer_mask'] | pack['work_mask']
            assert not (pack['answer_mask'] & pack['work_mask']).any()
            positions = pack['target_positions'][active].tolist()
            labels = pack['labels'][active].tolist()
            assert labels == pack['input_ids'][0, positions].tolist()
            paused = (pack['input_ids'][0] == pid).nonzero()[:, 0].tolist()
            assert len(paused) == (PAUSES_PER_STEP*len(q.program) if view == 'pause' else 0)
            assert set(paused).isdisjoint(positions)
            assert pack['work_mask'].sum().item() == (6*len(q.program) if view == 'trace' else 0)
            # Every inference context is a prefix of its matching training
            # sequence. Trace mode starts before state generation; other modes
            # start after the same Final header, with no supplied answer token.
            ids = inference_context(q, tok, view)
            assert ids == pack['input_ids'][0, :len(ids)].tolist()
            assert (pack['labels'][~active] == -100).all()
            totals[view].update(packs=1, answer_labels=len(answer), work_labels=int(pack['work_mask'].sum()),
                                supplied_pause_positions=len(paused), target_labels=len(labels))
        if (i+1) % 4096 == 0:
            print('All native TRAIN views checked:', i+1, flush=True)
    assert len(pairs) == manifest['inventories']['TRAIN']['families']
    assert set(pairs.values()) == {2}
    return dict(status='PASS_ALL_NATIVE_TRAIN_PACKS_MATCHED_ANSWER_MASKS_AND_PROVIDED_PAUSES',
        train_rows=len(rows), train_families=len(pairs), native_packs=4*len(rows),
        per_view={k: dict(v) for k, v in totals.items()}, width=256, target_slots=64,
        provided_pause_id=pid, provided_pause_token='<|fim_pad|>', pause_positions_per_step=PAUSES_PER_STEP,
        producer_sha256=hashlib.sha256((Path(__file__).resolve().parents[1]/'train/permutation_program_packing.py').read_bytes()).hexdigest(),
        checker_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        manifest_sha256=hashlib.sha256((data/'manifest.json').read_bytes()).hexdigest(),
        limits=['CPU/tokenizer proof, not an accelerator or learning result.',
                'The pretrained FIM-pad embedding is reused and must be trained through downstream answer gradients.',
                'Pause count is prefix length only; no pause-token prediction loss. Equal positions are not equal latency.',
                'No future-dependent pause budget, loss dilution or dropped target rows.'])


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--model-dir', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    assert not a.output.exists()
    result = audit(a.data, a.model_dir)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))

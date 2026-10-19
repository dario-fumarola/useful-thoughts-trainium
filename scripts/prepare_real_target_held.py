"""Freeze fresh real-document developmental boundaries without model outcomes.

Select384 distinct ordered exact-value expressions from unused source TRAIN
documents, excluding the actual TRAIN5120 and both old arithmetic DEV panels.
Never open official TEST or an unlisted raw corpus line.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import json
from pathlib import Path

from transformers import AutoTokenizer
from train.calculation_boundaries import maximal_lhs, parse_numeric
from train.expression_identity import expression_key
from train.real_trace_packing import prefix_calculation
from train.real_target_packing import ARMS, training_pack
from weekend_thinking.artifacts import canonical_hash


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root, output = args.repo_root.resolve(), args.output.resolve()
    data = root / 'local/nvme-study/data/real-prefix-partial-labels-v9'
    train_path = data / 'TRAIN.jsonl'
    assert sha(train_path) == '437c64bc9ce4b451f4d5da07fccca9e108ee9add75b3c955bc54d2a2eca6b915'
    train = rows(train_path)
    previous_paths = [data / 'DEV.jsonl', root / 'local/nvme-study/data/real-trace-cpt-v10-held-panels-v2/DEV_novel_expression256.jsonl']
    previous = [v for path in previous_paths for v in rows(path)]
    excluded_documents = {v['document_id'] for v in train + previous}
    excluded_expressions = {expression_key(v['normalized_expression']) for v in train + previous}
    inventory_path = data / 'all-candidate-inventory.jsonl'
    inventory = rows(inventory_path)
    unused = [v for v in inventory if v['split'] == 'train' and v['document_id'] not in excluded_documents
              and v['partial_actions'] <= 64]
    eligible = [v for v in unused if expression_key(v['expression']) not in excluded_expressions]
    salt = 'real-target-sft-v15-fresh-real-boundaries-384-v1'
    eligible.sort(key=lambda v: canonical_hash(dict(salt=salt, document_id=v['document_id'])))
    unique, seen = [], set()
    for v in eligible:
        key = expression_key(v['expression'])
        if key not in seen:
            unique.append(v)
            seen.add(key)
    assert len(unused) == 687 and len(eligible) == 505 and len(unique) == 499
    selected = {v['item_id']: v for v in unique[:384]}
    assert len(selected) == 384
    manifest_path = root / 'local/nvme-study/data/finemath-raw-compound-prefix-v9/manifest.json'
    source_manifest = json.loads(manifest_path.read_text())
    source = Path(source_manifest['accepted_file'])
    assert sha(source) == source_manifest['accepted_sha256']
    tok = AutoTokenizer.from_pretrained(root / 'local/assets/qwen3-1.7b-base', local_files_only=True, trust_remote_code=False)
    materialized = {}
    for number, line in enumerate(source.read_text().splitlines(), 1):
        row = json.loads(line)
        if row['item_id'] not in selected:
            continue
        assert row['split'] == 'train' and row['document_id'] not in excluded_documents
        text = tok.decode(row['prefix_ids'], skip_special_tokens=False, clean_up_tokenization_spaces=False)
        expr = parse_numeric(maximal_lhs(text)).expression
        assert expression_key(expr) == expression_key(selected[row['item_id']]['expression'])
        full, full_ids = prefix_calculation(row['prefix_ids'], tok)
        partial = '\n'.join(full.splitlines()[:-1]) + '\n'
        from train.symbolic_note_labels import postorder_steps
        assert postorder_steps(expr)[0] == Fraction(row['numeric_value'])
        value = dict(item_id=row['item_id'], document_id=row['document_id'], split='followup_dev',
            prefix_ids=row['prefix_ids'], prefix_text=text, continuation16_ids=row['natural_future16_ids'],
            numeric_surface_ids=row['numeric_surface_ids'], numeric_surface_text=row['numeric_surface_text'],
            closing_boundary_ids=row['closing_boundary_ids'], numeric_value=row['numeric_value'],
            normalized_expression=expr, partial_note_text=partial, partial_note_ids=tok.encode(partial, add_special_tokens=False),
            full_calculation_text=full, full_calculation_ids=full_ids, stop_id=tok.eos_token_id,
            source_accepted_line=number, source_accepted_line_sha256=hashlib.sha256((line+'\n').encode()).hexdigest(),
            source_provenance={k:row[k] for k in ['raw_line_sha256','raw_text_sha256','raw_source_line',
                'source_url','prefix_source_char_end','source_chunk_char_start','source_chunk_char_end',
                'absolute_source_cut','raw_rhs_validation']})
        assert row['item_id'] not in materialized
        materialized[row['item_id']] = value
    cohort = [materialized[key] for key in selected]
    assert len(cohort) == len({v['document_id'] for v in cohort}) == 384
    assert len({expression_key(v['normalized_expression']) for v in cohort}) == 384
    label_counts = {}
    for arm in ARMS:
        counts = Counter()
        for row in train:
            pack = training_pack(row, tok, arm)
            counts['documents'] += 1
            counts['active_labels'] += int(pack['future_mask'].sum())
            assert not pack['note_close_mask'].any()
        label_counts[arm] = dict(counts)
    output.mkdir(parents=True, exist_ok=False)
    held_path = output / 'DEV_fresh384.jsonl'
    held_path.write_text(''.join(json.dumps(v, separators=(',', ':'))+'\n' for v in cohort))
    question_path = output / 'DEV_fresh384-questions.jsonl'
    question_path.write_text(''.join(json.dumps({k:v[k] for k in ['item_id','document_id','prefix_ids']}, separators=(',', ':'))+'\n' for v in cohort))
    result = dict(status='FROZEN_NEW_DOCUMENT_AND_EXACT_AST_DISJOINT_DEVELOPMENT_NO_MODEL_RUN',
        schema='real-target-sft-v15-held-v1', utc=datetime.now(timezone.utc).isoformat(),
        source_script_sha256=sha(__file__), source_manifest_sha256=sha(manifest_path),
        inventory_sha256=sha(inventory_path), training_sha256=sha(train_path),
        previous_panels={str(path):sha(path) for path in previous_paths},
        unused_train_documents=687, eligible_novel_documents=505, unique_novel_expressions=499,
        selected=384, selection_salt=salt, per_training_pass=label_counts,
        trace_operations=dict(Counter(len(v['full_calculation_text'].splitlines()) for v in cohort)),
        files={p.name:dict(bytes=p.stat().st_size, sha256=sha(p)) for p in [held_path,question_path]},
        limits=['New prospective development from unused source TRAIN documents,not official TEST.',
                'Disjoint from actual TRAIN5120 and both previous arithmetic panels by document and ordered exact-value AST.',
                'Operator patterns and equal answers may recur;no algebraic-equivalence or pretraining-novelty claim.',
                'No model output,likelihood,answer correctness or difficulty score selects these384.',
                'Source curator verifies observed RHS;supervised traces are rederived from the prefix alone.',
                'Earlier investigations may have inventoried source documents;no claim of universally unseen source text.'])
    (output/'manifest.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()

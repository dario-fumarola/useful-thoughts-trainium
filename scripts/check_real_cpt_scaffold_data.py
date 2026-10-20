"""Independent native-source readback for V29 CPU data, without model calls."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from transformers import AutoTokenizer

from train import real_cpt_scaffold as scaffold
from train.expression_identity import expression_key
from weekend_thinking.artifacts import atomic_json, sha256_file, utc_now


def rows(path):
    with Path(path).open() as f: return [json.loads(line) for line in f]


def native_selection(path, wanted):
    """Parse only a metadata projection until a non-TEST ID is selected.

    A deliberately malformed reserved payload can pass this reader: its token
    list is never parsed. The original file's full SHA is checked by the caller.
    """
    result, counts = {}, Counter()
    with Path(path).open('rb') as f:
        for line in f:
            header, marker, _ = line.partition(b',"prefix":')
            if not marker: raise ValueError('Original native header changed')
            metadata = json.loads(header + b'}')
            if set(metadata) != {'id', 'document_id', 'split'}:
                raise ValueError('Metadata projection includes unknown fields')
            split = metadata['split']
            if split not in ('train', 'dev', 'test'): raise ValueError('Unknown source scope')
            counts['headers:' + split] += 1
            if split == 'test': continue
            if metadata['id'] not in wanted: continue
            value = json.loads(line)
            if value['id'] in result: raise ValueError('Repeated selected native record')
            result[value['id']] = value
            counts['payloads:' + split] += 1
    if set(result) != set(wanted): raise ValueError('Selected native records are missing')
    return result, dict(counts)


def pack_digest(pack):
    h = hashlib.sha256()
    for name in ('input_ids', 'target_positions', 'labels'):
        h.update(name.encode()); h.update(pack[name].numpy().tobytes())
    for name, mask in sorted(pack['component_masks'].items()):
        h.update(name.encode()); h.update(mask.numpy().tobytes())
    return h.digest()


def check(root, directory, output):
    manifest = json.loads((directory/'manifest.json').read_text())
    if manifest['status'] != 'CPU_DATA_AND_PACKING_COMPLETE_NOT_TRAINING_RELEASE':
        raise ValueError('Closed CPU data manifest required')
    for path, expected in manifest['inputs'].items():
        if sha256_file(path) != expected['sha256']: raise ValueError(f'Changed input: {path}')
    cohorts = {}
    for name, expected in manifest['files'].items():
        path = directory/name
        if sha256_file(path) != expected['sha256'] or path.stat().st_size != expected['bytes']:
            raise ValueError('Frozen output file changed')
        cohorts[name] = rows(path)
        if len(cohorts[name]) != expected['rows']: raise ValueError('Incorrect frozen row count')
    train, arithmetic, ordinary = (cohorts[n] for n in (
        'TRAIN.jsonl', 'DEV_calculation.jsonl', 'DEV_ordinary.jsonl'))
    if Counter(r['kind'] for r in train) != {'calculation':5120, 'ordinary':15360}:
        raise ValueError('Incorrect proposed source mixture')
    train_documents = {r['document_id'] for r in train}
    held_documents = {r['document_id'] for r in arithmetic + ordinary}
    if len(train_documents) != len(train) or len(held_documents) != len(arithmetic)+len(ordinary) or train_documents & held_documents:
        raise ValueError('Duplicated or overlapping source documents')
    expressions = {expression_key(r['normalized_expression']) for r in train if r['kind']=='calculation'}
    if any(expression_key(r['normalized_expression']) in expressions for r in arithmetic):
        raise ValueError('Held ordered expression appears in training')
    projected = [{k:r[k] for k in ('item_id','document_id','prefix_ids')} for r in arithmetic]
    if projected != cohorts['DEV_calculation-questions.jsonl']:
        raise ValueError('Writer projection changed or includes future labels')
    original_train = {r['item_id']:r for r in rows(root/'local/nvme-study/data/real-prefix-partial-labels-v9/TRAIN.jsonl')}
    original_fresh = {}
    for path in manifest['inputs']:
        if Path(path).name == 'accepted.jsonl':
            original_fresh.update({r['item_id']:r for r in rows(path)})
    for r in [r for r in train if r['kind']=='calculation'] + arithmetic:
        original = (original_train if r['split']=='train' else original_fresh)[r['item_id']]
        future = original.get('continuation16_ids', original.get('natural_future16_ids'))
        if r['prefix_ids'] != original['prefix_ids'] or r['continuation16_ids'] != future:
            raise ValueError('A calculation source token was changed')
    wanted = {r['item_id']:r for r in train + ordinary if r['kind']=='ordinary'}
    originals, scan = native_selection(root/'local/data/finemath/production-500000.audit-v2.records.jsonl', wanted)
    for key, r in wanted.items():
        original = originals[key]
        if (r['prefix_ids'] != original['prefix'] or r['continuation16_ids'] != original['future']
                or r['document_id'] != original['document_id'] or r['split'] != original['split']):
            raise ValueError('An ordinary source token or scope was changed')

    tok = AutoTokenizer.from_pretrained(root/'local/assets/qwen3-1.7b-base', local_files_only=True, trust_remote_code=False)
    clean_hash = hashlib.sha256()
    totals = {arm:Counter() for arm in scaffold.ARMS}
    opening, answer = (tok.encode(s, add_special_tokens=False) for s in (scaffold.OPEN, scaffold.ANSWER))
    for r in train:
        common = None
        for arm in scaffold.ARMS:
            view = scaffold.training_views(r, tok, arm)
            digest = pack_digest(view.clean)
            if common is not None and digest != common: raise ValueError('Clean-view mismatch')
            common = digest
            c = totals[arm]
            c['documents'] += 1; c['source_allocated_positions'] += scaffold.WIDTH
            c['source_targets'] += int(view.clean['component_masks']['source'].sum())
            if view.auxiliary is None: continue
            aux = view.auxiliary
            c['auxiliary_examples'] += 1; c['auxiliary_allocated_positions'] += scaffold.WIDTH
            for name, mask in aux['component_masks'].items(): c[name+'_targets'] += int(mask.sum())
            future_mask = aux['component_masks']['continuation']
            if aux['labels'][future_mask].tolist() != r['continuation16_ids']:
                raise ValueError('Auxiliary continuation target mismatch')
            context = list(r['prefix_ids'])
            if arm != 'raw': context += opening + (r['full_calculation_ids'] if arm=='computation' else []) + answer
            ids = context + r['continuation16_ids']
            if aux['input_ids'][0,:len(ids)].tolist() != ids:
                raise ValueError('Auxiliary context is not the declared native concatenation')
        clean_hash.update(common)
    if {k:dict(v) for k,v in totals.items()} != manifest['per_pass']:
        raise ValueError('Current source packing differs from the frozen CPU audit')
    for r in arithmetic:
        scaffold.validate_source(r, tok, training=False)
        text, ids = scaffold.verified_work(r['prefix_ids'], tok)
        if text != r['full_calculation_text'] or ids != r['full_calculation_ids']:
            raise ValueError('Held teacher is not prefix-computable')
    result = dict(status='PASS_NATIVE_SOURCE_AND_ALL_ARM_PACK_READBACK', utc=utc_now(),
        schema=scaffold.SCHEMA, manifest_sha256=sha256_file(directory/'manifest.json'),
        source_files={str(Path(__file__).resolve()):sha256_file(__file__),
                      'train/real_cpt_scaffold.py':sha256_file(scaffold.__file__)},
        native_calculation_rows=5120+len(arithmetic), native_ordinary_rows=len(wanted),
        held_writer_projections=len(projected), reserved_TEST_payloads_parsed=0, source_scan=scan,
        clean_all_arm_equal_documents=len(train), clean_pack_stream_sha256=clean_hash.hexdigest(),
        current_per_pass={k:dict(v) for k,v in totals.items()},
        proposed_loss_weights={'source':1.,'auxiliary_continuation':scaffold.CONTINUATION_WEIGHT,'work':scaffold.WORK_WEIGHT},
        scope='CPU data only; no released-model forward, hardware qualification or scientific outcome')
    atomic_json(output, result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo-root', type=Path, required=True)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    check(args.repo_root.resolve(), args.data.resolve(), args.output.resolve())

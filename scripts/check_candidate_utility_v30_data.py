"""Independent all-pool raw/native/exact-value audit before V30 model calls."""
from __future__ import annotations

import argparse
import ast
from fractions import Fraction
import hashlib
import json
from pathlib import Path
from urllib.parse import unquote, urlsplit

from weekend_thinking.artifacts import atomic_json, sha256_file


def read(path):
    return json.loads(Path(path).read_text())


def lines(path):
    return [json.loads(line) for line in Path(path).open()]


def page(url):
    value = urlsplit(url)
    return (value.hostname or '').casefold().removeprefix('www.') + unquote(value.path).casefold().rstrip('/')


def value(text):
    """Separate rational AST evaluator; no eval or shared arithmetic interpreter."""
    text = text.replace('−', '-').replace('×', '*').replace('÷', '/').replace('^', '**').strip()
    tree = ast.parse(text, mode='eval')
    assert len(text) <= 512 and sum(1 for _ in ast.walk(tree)) <= 128
    def visit(node):
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            return Fraction(ast.get_source_segment(text, node))
        if isinstance(node, ast.UnaryOp):
            assert isinstance(node.op, (ast.UAdd, ast.USub))
            child = visit(node.operand)
            return child if isinstance(node.op, ast.UAdd) else -child
        assert isinstance(node, ast.BinOp)
        left, right = visit(node.left), visit(node.right)
        if isinstance(node.op, ast.Add): return left + right
        if isinstance(node.op, ast.Sub): return left - right
        if isinstance(node.op, ast.Mult): return left * right
        if isinstance(node.op, ast.Div): return left / right
        assert isinstance(node.op, ast.Pow) and right.denominator == 1 and abs(right) <= 12
        return left ** int(right)
    return visit(tree.body)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('data', 'source', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    complete, plan = read(args.data / 'complete.json'), read(args.data / 'plan.json')
    assert complete['status'] == 'COMPLETE_CPU_FRESH_TRAIN_POOL_NOT_MODEL_RELEASE'
    assert sha256_file(args.data / 'plan.json') == complete['plan_sha256']
    for name, info in complete['files'].items():
        assert sha256_file(args.data / name) == info['sha256']
    prior_pages, prior_content, prior_expressions = set(), set(), set()
    from train.expression_identity import expression_key
    for name, digest in plan['prior_metadata_files'].items():
        path = Path(name); assert sha256_file(path) == digest
        if not name.endswith('.jsonl'): continue
        if path.name.endswith('.documents.jsonl') or 'membership' in path.name:
            for row in lines(path):
                assert not {'text', 'prefix', 'future', 'answer'} & row.keys()
                prior_pages.add(page(row['url']))
                prior_content.update(row[k] for k in ('document_id', 'content_sha256') if row.get(k))
        else:
            for row in lines(path):
                if 'normalized_expression' in row:
                    prior_expressions.add(expression_key(row['normalized_expression']))
    assert len(prior_pages) == plan['old_pages'] and len(prior_content) == plan['old_content_keys']
    assert len(prior_expressions) == plan['old_exact_expressions']
    pool = lines(args.data / 'eligible-new-TRAIN.jsonl')
    assert len(pool) == complete['eligible_documents'] and len(pool) >= 264
    assert len({r['item_id'] for r in pool}) == len({r['document_id'] for r in pool}) == len(pool)
    accepted = {row['item_id']: row for row in lines(args.data / 'accepted.jsonl')}
    needed = {accepted[row['item_id']]['raw_source_line'] for row in pool}
    membership = {row['source_line']: row for row in lines(args.data / 'TRAIN-membership.jsonl') if row['source_line'] in needed}
    assert set(membership) == needed
    raw = {}
    with (args.data / 'TRAIN-raw.jsonl').open('rb') as stream:
        for number, line in enumerate(stream, 1):
            if number in needed:
                raw[number] = (json.loads(line), hashlib.sha256(line).hexdigest())
    assert set(raw) == needed
    wanted = {membership[n]['parquet_row']: n for n in needed}
    assert len(wanted) == len(needed)
    source = read(args.source / 'download-complete.json')
    assert source['sha256'] == plan['source']['sha256']
    assert source['revision'] == plan['source']['revision'] == 'e92b25a616738fe95dc186b64dfb19f9c8525594'
    assert source['filename'] == plan['source']['filename'] == 'finemath-4plus/train-00061-of-00064.parquet'
    assert source['split'] == 'train'
    parquet = Path(source['path'])
    assert sha256_file(parquet) == source['sha256']
    import pyarrow as pa
    import pyarrow.parquet as pq
    position = 0; checked_source = set()
    for batch in pq.ParquetFile(parquet).iter_batches(batch_size=1024, columns=['url', 'text']):
        selected = sorted(i - position for i in wanted if position <= i < position + len(batch))
        if selected:
            payload = batch.take(pa.array(selected, type=pa.int64())).to_pylist()
            for offset, actual in zip(selected, payload):
                line = wanted[position + offset]; saved, _ = raw[line]
                assert actual['url'] == saved['url'] and actual['text'] == saved['text']
                assert saved['source']['parquet_row'] == position + offset
                assert saved['source']['revision'] == source['revision'] and saved['source']['filename'] == source['filename']
                assert position + offset < 65536
                checked_source.add(line)
        position += len(batch)
        if position > max(wanted): break
    assert checked_source == needed
    from transformers import AutoTokenizer
    from train.calculation_boundaries import maximal_lhs
    from train.real_cpt_scaffold import verified_work
    from train.real_cpt_readout import numeric_masks
    from train.scalar_completion import parse_initial_scalar_v2
    # Tokenizer location is bound by its actual files in the frozen input plan.
    tokenizer_paths = {Path(name).parent for name in plan['prior_metadata_files'] if Path(name).name == 'tokenizer.json'}
    assert len(tokenizer_paths) == 1
    tokenizer_path = tokenizer_paths.pop()
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    seen_pages, seen_content, seen_expressions = set(), set(), set()
    equation_count = 0
    for row in pool:
        original = accepted[row['item_id']]; number = original['raw_source_line']
        saved, raw_sha = raw[number]; meta = membership[number]; text = saved['text']
        assert row['split'] == meta['split'] == original['split'] == 'train'
        assert row['document_id'] == meta['document_id'] == original['document_id']
        assert row['source_file_sha256'] == complete['files']['accepted.jsonl']['sha256']
        assert raw_sha == original['raw_line_sha256']
        content = hashlib.sha256(' '.join(text.split()).encode()).hexdigest()
        p = page(saved['url']); key = expression_key(row['normalized_expression'])
        assert content == meta['content_sha256'] == row['document_id']
        assert p not in prior_pages and content not in prior_content and key not in prior_expressions
        assert p not in seen_pages and content not in seen_content and key not in seen_expressions
        seen_pages.add(p); seen_content.add(content); seen_expressions.add(key)
        past, future = row['prefix_ids'], row['continuation16_ids']
        assert len(past) == 512 and len(future) == 16 and future == original['natural_future16_ids']
        tokens = tokenizer(text[:65536], add_special_tokens=False, truncation=True, max_length=8208)['input_ids']
        assert tokens[original['source_chunk_token_start']:original['source_chunk_token_end']] == past + future
        start, end, cut = original['source_chunk_char_start'], original['source_chunk_char_end'], original['prefix_source_char_end']
        prefix = tokenizer.decode(past, skip_special_tokens=False, clean_up_tokenization_spaces=False)
        assert prefix == text[start:cut] and tokenizer.decode(past + future, skip_special_tokens=False,
            clean_up_tokenization_spaces=False) == text[start:end]
        root = value(maximal_lhs(prefix)); assert root == Fraction(row['numeric_value'])
        proof = parse_initial_scalar_v2(text[cut:cut+8192], 'length_cap')
        assert proof['status'] == 'valid' and Fraction(proof['value']) == root
        label, ids = verified_work(past, tokenizer)
        assert label == row['full_calculation_text'] and ids == row['full_calculation_ids'] and len(ids) < 128
        equations = label.strip().splitlines(); assert 2 <= len(equations) <= 6
        for equation in equations:
            left, right = equation.rsplit(' = ', 1)
            assert value(left) == value(right)
            equation_count += 1
        assert value(equations[-1].rsplit(' = ', 1)[1]) == root
        masks = numeric_masks(row, tokenizer)
        assert all(len(mask) == 16 for mask in masks.values()) and any(masks['numeral'])
    for name, digest in plan['curator_source_files'].items():
        assert sha256_file(Path(__file__).resolve().parents[1] / name) == digest
    args.output.mkdir(parents=True, exist_ok=False)
    atomic_json(args.output / 'independent-data-check.json', dict(
        status='PASS_ALL_FRESH_POOL_RAW_PARQUET_NATIVE_PREFIX_LABEL_VALUE_AND_EXCLUSION_CHECKS',
        documents=len(pool), independently_true_equations=equation_count, actual_parquet_rows_checked=len(checked_source),
        pool_sha256=sha256_file(args.data / 'eligible-new-TRAIN.jsonl'), complete_sha256=sha256_file(args.data / 'complete.json'),
        source_sha256=sha256_file(__file__),
        limits='Within-repository independent source/value reconstruction,not model execution,external replication,pretraining novelty or a diagnostic-model release.'))
    print('PASS_ALL_FRESH_POOL_DOCUMENTS', len(pool), 'TRUE_EQUATIONS', equation_count, flush=True)


if __name__ == '__main__':
    main()

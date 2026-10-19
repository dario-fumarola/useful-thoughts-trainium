"""Post-hoc token accounting of completed observations; no new model forwards.

Run from repository root with the existing Python environment. Boundaries are
frozen curator annotations, not a new free-answer parser or evaluation gate.
"""
from pathlib import Path
from datetime import datetime, timezone
import gzip
import hashlib
import json
import numpy as np
from tokenizers import Tokenizer

R = Path.cwd()
REPORT = R / 'validation/real_trace_cpt_v10_complete_20261019.json'
sha = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
assert sha(REPORT) == 'eb5f6b07aa4b39da6f708a7941fc2bc3bbaf1580b5bc3942979730acab2e1bf4'
report = json.loads(REPORT.read_text())
tokenizer_path = R / 'local/assets/qwen3-1.7b-base/tokenizer.json'
tok = Tokenizer.from_file(str(tokenizer_path))
out = R / 'validation/real_trace_v10_token_partition_20261019'
out.mkdir(exist_ok=False)
parts = ['leading_whitespace', 'numeral', 'curator_boundary_with_gap', 'remaining_tail']
all_rows, metadata, results = [], {}, {}
for panel, path, digest in [
    ('DEV_AST256', 'local/nvme-study/data/real-trace-cpt-v10-held-panels-v2/DEV_novel_expression256.jsonl',
     '41d4aa6e6d217cdd57296953c669a95005ae1eb17b29ba5dc8565627b0c78fba'),
    ('DEV_legacy256', 'local/nvme-study/data/real-prefix-partial-labels-v9/DEV.jsonl',
     '8ba676b1cb7cb1f46405d259911207e2d4467ffed922ef07ab2bf9e28738ce79')]:
    assert sha(R / path) == digest
    data = [json.loads(line) for line in (R / path).read_text().splitlines()]
    assert len(data) == len({v['document_id'] for v in data}) == 256
    data.sort(key=lambda v: v['item_id'])
    bounds = {}
    for v in data:
        ids = v['continuation16_ids']
        text = tok.decode(ids, skip_special_tokens=False)
        stripped = text.lstrip(' \t\r\n')
        literal = v['numeric_surface_text']
        assert stripped.startswith(literal)
        left, right = len(text) - len(stripped), len(text) - len(stripped) + len(literal)
        ends = [0] + [len(tok.decode(ids[:i], skip_special_tokens=False)) for i in range(1, 17)]
        start = next(i for i in range(16) if ends[i+1] > left)
        end = next(i for i in range(1, 17) if ends[i] >= right)
        close = v['closing_boundary_ids']
        matches = [i for i in range(end, 16)
                   if not tok.decode(ids[end:i], skip_special_tokens=False).strip()
                   and ids[i:i+len(close)] == close]
        assert matches and len(matches) == 1
        boundary_end = matches[0] + len(close)
        assert 0 <= start < end <= boundary_end <= 16
        bounds[v['item_id']] = [0, start, end, boundary_end, 16]
        metadata.setdefault(panel, []).append(dict(item_id=v['item_id'], document_id=v['document_id'],
            bounds=bounds[v['item_id']], target16=ids, literal=literal,
            closing_boundary_ids=close))

    cells = {}
    for model in report['absolute_likelihood']:
        name = f'validation/real_trace_cpt_v10_complete_20261019/{model}-{panel}-all1280-logps.jsonl.gz'
        payload = (R / name).read_bytes()
        assert hashlib.sha256(payload).hexdigest() == report['payloads'][name]['sha256']
        values = [json.loads(line) for line in gzip.decompress(payload).splitlines()]
        assert len(values) == 1280
        cells[model] = {(v['item_id'], v['mode']): v['logps16'] for v in values}
        assert len(cells[model]) == 1280
    rng = np.random.default_rng(2026101915)
    indices = rng.integers(0, 256, size=(20000, 256))
    results[panel] = {}
    for seed in ['2026101913', '2026101914']:
        raw, prior, post = [f'real_{arm}-seed{seed}' for arm in ['raw', 'prior', 'post']]
        results[panel][seed] = {}
        contrasts = [
            ('post_raw_minus_raw_raw', raw, 'raw', post, 'raw'),
            ('prior_own_minus_raw_raw', raw, 'raw', prior, 'own'),
            ('prior_own_minus_prior_raw', prior, 'raw', prior, 'own'),
            ('prior_own_minus_prior_empty', prior, 'empty', prior, 'own'),
            ('prior_own_minus_B0raw', 'B0', 'raw', prior, 'own')]
        for name, a, am, b, bm in contrasts:
            differences = []
            for v in data:
                key = v['item_id']
                aa, bb = cells[a][key, am], cells[b][key, bm]
                assert len(aa) == len(bb) == 16
                delta = [y-x for x, y in zip(aa, bb)]
                edges = bounds[key]
                components = [sum(delta[edges[i]:edges[i+1]]) for i in range(4)]
                assert abs(sum(components) - sum(delta)) < 1e-10
                differences.append(components)
                all_rows.append(dict(panel=panel, seed=seed, contrast=name, item_id=key,
                                     document_id=v['document_id'], parts=dict(zip(parts, components)),
                                     total16=sum(delta)))
            arr = np.asarray(differences, dtype=np.float64)
            totals = arr.sum(1)
            original = report['seeds'][seed]['primary_AST' if panel == 'DEV_AST256' else 'legacy_secondary']
            if name in original:
                assert abs(totals.mean() - original[name]['difference_b_minus_a']) < 1e-10
            columns = {}
            for i, part in enumerate(parts):
                ci = np.quantile(arr[:, i][indices].mean(1), [.025, .975]).tolist()
                columns[part] = dict(mean_nats_per_document=float(arr[:, i].mean()), ci95=ci)
            columns['total16_mean'] = float(totals.mean())
            results[panel][seed][name] = columns

payload_path = out / 'all-document-component-differences.jsonl.gz'
payload_path.write_bytes(gzip.compress((''.join(json.dumps(v, separators=(',', ':'))+'\n' for v in all_rows)).encode(), mtime=0))
meta_path = out / 'fixed-token-partitions.json'
meta_path.write_text(json.dumps(metadata, indent=2))
result = dict(status='POSTHOC_ADDITIVE_TOKEN_ACCOUNTING_ALL_FIXED_DOCUMENTS', utc=datetime.now(timezone.utc).isoformat(),
              source_report_sha256=sha(REPORT), tokenizer_sha256=sha(tokenizer_path),
              source_sha256=sha(__file__), observations_reused=len(all_rows),
              results=results, parts=parts, bootstrap=dict(draws=20000, seed=2026101915, unit='document'),
              limits=[
                  'Post-hoc descriptive partition; neither primary gate nor scalar parser is changed.',
                  'All256 documents remain in every contrast; each seed and panel separate. CIs are unadjusted and checkpoint-conditional.',
                  'Numeral includes any whitespace merged into its native token; fully separate preceding whitespace is its own component.',
                  'Boundary means the frozen curator closing token(s), plus intervening whitespace; it can be a word or mathematical relation, not necessarily sentence punctuation.',
                  'All annotated numeral/curator-boundary tokens fit in16 here; this does not certify initial-scalar-v2 support or full semantic completion.',
                  'Component intervals need not add to total intervals. Components add exactly per document and their means sum to the unchanged original endpoint.',
                  'No model forward, additional generation, model selection, or new independent observation.'],
              files={str(p.relative_to(R)):dict(sha256=sha(p), bytes=p.stat().st_size) for p in [payload_path, meta_path]})
out.with_suffix('.json').write_text(json.dumps(result, indent=2))
print(json.dumps(dict(status=result['status'], primary=results['DEV_AST256']), indent=2))

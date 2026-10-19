"""All fixed native/replay answer contrasts; developmental panel, no success gate."""
from pathlib import Path
from collections import Counter
from dataclasses import asdict
import gzip
import json
import re
import numpy as np
from train.document_qa_answers import parse_answer, score
from train.document_evidence_plan_answers import parse_evidence, annotated_support_overlap
from weekend_thinking.artifacts import atomic_json, sha256_file, utc_now

R = Path('/home/ubuntu/dario')
C = R/'local/diagnostics/document-evidence-weight-v25-ablation-v1'
D = R/'local/nvme-cpt/data/document-evidence-plan-v23-TRAIN2048-held255-v2'
P = R/'validation/document_evidence_plan_v23_pilot_complete_20261019'
read = lambda p: json.loads(p.read_text())
rows = lambda p: [json.loads(v) for v in p.read_text().splitlines()]
gold = {v['item_id']: v for v in rows(D/'held255-gold-only.jsonl')}
assert len(gold) == 255
questions = {v['item_id']: v for v in rows(D/'held255-questions-only.jsonl')}
assert set(questions) == set(gold)
manifest = read(D/'manifest.json')
assert sha256_file(D/'held255-gold-only.jsonl') == manifest['files']['held_gold']['sha256']
tables, scored, inputs = {}, {}, {}

for kind, count in [('native', 510), ('answer-replay', 765)]:
    output = R/f'local/nvme-cpt/evaluation/document-evidence-weight-v25-held255-{kind}-v1'
    control = C/('HELD-native' if kind == 'native' else 'HELD-replay')
    assert read(control/'GEN/closure.json')['actual_attempts'] == count
    p = read(output/'protocol.json')
    inputs[kind+'_protocol_sha256'] = sha256_file(output/'protocol.json')
    attempts = []
    for shard in range(4):
        directory = output/'answers'/f'shard-{shard}'
        report = read(directory/'run.json')
        assert report['status'] == 'complete_all_attempts'
        assert report['attempts_sha256'] == sha256_file(directory/'attempts.jsonl')
        assert read(directory/'native.json')['passed']
        assert read(directory/'cache-prefill.json')['max_logp_error'] <= .001
        attempts.extend(rows(directory/'attempts.jsonl'))
    assert len(attempts) == count
    for mode in p['modes']:
        values = [v for v in attempts if v['mode'] == mode]
        assert len(values) == len({v['item_id'] for v in values}) == 255
        assert {v['item_id'] for v in values} == set(gold)
        key = kind+':'+mode
        result = {}
        for v in values:
            g = gold[v['item_id']]
            answer = parse_answer(v['raw_text'], v['parser_mode'],
                terminated=v['stop_type'].startswith('token:'), capped=v['stop_type'] == 'length_cap')
            metrics = score(answer, g['answer'], g['answer_aliases'])
            if v['status'] != 'ok':
                metrics = {name: 0. for name in metrics}
            extra = {}
            if kind == 'native':
                field = re.search(r'(?im)^[ \t]*Final answer:[ \t]*', v['raw_text'])
                prefix = v['raw_text'][:field.start()] if field else v['raw_text']
                extra['generated_prefix_ID_count'] = len(re.findall(r'\[\d+\]', prefix))
                if mode == 'selection':
                    selection = parse_evidence(v['raw_text'], len(questions[v['item_id']]['contexts']))
                    extra['selection'] = asdict(selection)
                    extra['annotation_overlap'] = annotated_support_overlap(selection, g['support_indices'])
            else:
                first = v['raw_text'].strip().split('\n', 1)[0]
                extra['generated_evidence_header'] = bool(re.search(r'(?im)^\s*Evidence:', v['raw_text']))
                extra['generated_final_header'] = bool(re.search(r'(?im)^\s*Final answer:', v['raw_text']))
                extra['leading_bracket_ID_line'] = bool(re.fullmatch(r'(?:\[\d+\]\s*)+', first))
            result[v['item_id']] = dict(native=v, answer=asdict(answer), score=metrics, **extra)
        scored[key] = result
        tables[key] = dict(items=255, canonical_correct=sum(v['score']['canonical_em'] for v in result.values()),
            mean_canonical_F1=sum(v['score']['canonical_f1'] for v in result.values())/255,
            alias_correct=sum(v['score']['attributed_alias_em'] for v in result.values()),
            parsed_status=dict(Counter(v['answer']['status'] for v in result.values())),
            unsupported=sum(v['answer']['status'] != 'complete' for v in result.values()),
            execution_errors=sum(v['native']['status'] != 'ok' for v in result.values()),
            capped=sum(v['native']['stop_type'] == 'length_cap' for v in result.values()),
            generated_actions=sum(v['native']['n_actions'] for v in result.values()),
            generated_prefix_ID_rows=sum(v.get('generated_prefix_ID_count', 0) > 0 for v in result.values()),
            repeated_evidence_headers=sum(v.get('generated_evidence_header', False) for v in result.values()),
            repeated_final_headers=sum(v.get('generated_final_header', False) for v in result.values()),
            leading_bracket_ID_lines=sum(v.get('leading_bracket_ID_line', False) for v in result.values()))

for name, file, expected in [
    ('old-matched-direct', 'evidence_direct-seed2026101923-direct-native-and-scored.jsonl.gz', 127),
    ('old-selection-native', 'evidence_select-seed2026101923-selection-native-and-scored.jsonl.gz', 129)]:
    records = [json.loads(line) for line in gzip.decompress((P/file).read_bytes()).splitlines()]
    assert len(records) == 255 and {v['native']['item_id'] for v in records} == set(gold)
    assert sum(v['score']['canonical_em'] for v in records) == expected
    scored[name] = {v['native']['item_id']: v for v in records}
    inputs[name+'_sha256'] = sha256_file(P/file)

comparisons = [
    ('native:selection', 'old-matched-direct'),
    ('native:selection', 'old-selection-native'),
    ('native:selection', 'native:direct'),
    ('answer-replay:selected_id_replay', 'answer-replay:empty'),
    ('answer-replay:oracle', 'answer-replay:selected_id_replay'),
    ('native:selection', 'answer-replay:selected_id_replay'),
    ('native:direct', 'answer-replay:empty'),
]
contrasts = []
for index, (treatment, control) in enumerate(comparisons):
    keys = sorted(gold)
    delta = np.array([scored[treatment][k]['score']['canonical_em']-scored[control][k]['score']['canonical_em'] for k in keys])
    rng = np.random.default_rng(2026101925+100+index)
    draw = delta[rng.integers(0, 255, (20000, 255))].mean(1)
    low, high = np.quantile(draw, [.025, .975])
    contrasts.append(dict(treatment=treatment, control=control, paired_gain=float(delta.mean()),
        CI95=[float(low), float(high)], corrected=int((delta > 0).sum()), regressed=int((delta < 0).sum())))
summary = dict(status='COMPLETE_DEVELOPMENTAL_WEIGHT_V25_NATIVE_AND_CLOSED_FIELD_ANSWERS', utc=utc_now(),
    tables=tables, contrasts=contrasts, inputs=inputs, source_sha256=sha256_file(__file__),
    limitations=['Held255 was inspected before the coefficient choice; this is developmental, not untouched confirmation.',
        'Native and canonical-pointer replay are different inference interfaces.',
        'Any direct path generating IDs is operationally contaminated and must be qualified as such.',
        'Oracle annotations can be insufficient or ambiguous. No semantic-gold or calibration claim.',
        'All original V23 scores/gates remain unchanged; unadjusted paired question intervals condition on checkpoints.'])
atomic_json(C/'held-native-and-replay-result.json', summary)
with (C/'held-native-and-replay-scored.json').open('x') as stream:
    json.dump(scored, stream)
print(json.dumps(summary, indent=2))

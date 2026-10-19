"""Post-hoc error partition from existing public outputs; no model execution."""
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import gzip
import hashlib
import json

ROOT = Path(__file__).resolve().parents[2]
original = json.loads((ROOT/'validation/variable_depth_writer_v11_complete_20261019.json').read_text())
directory = ROOT/'validation/variable_depth_writer_v11_complete_20261019'
def load(pattern):
    files = list(directory.glob(pattern))
    assert len(files) == 1
    path = files[0]
    assert hashlib.sha256(path.read_bytes()).hexdigest() == original['payloads'][str(path.relative_to(ROOT))]['sha256']
    return [json.loads(line) for line in gzip.decompress(path.read_bytes()).splitlines()]

panels = {}
for panel in ('DEV_composition', 'DEV_depth2', 'DEV_depth6'):
    teacher = {v['item_id']: v for v in load(f'{panel}-reader-all*-answers.jsonl.gz') if v['mode'] == 'teacher'}
    panels[panel] = {}
    for arm in ('fixed', 'variable'):
        own = {v['item_id']: v for v in load(f'{panel}-{arm}-student-answers-all*-answers.jsonl.gz') if v['mode'] == 'own'}
        assert set(teacher) == set(own)
        counts, table = Counter(), Counter()
        for key, value in own.items():
            reference = teacher[key]
            same = value['prompt_ids'] == reference['prompt_ids']
            correct, supplied_correct = bool(value['score']), bool(reference['score'])
            table[same, correct, supplied_correct] += 1
            counts['items'] += 1
            counts['correct'] += correct
            counts['teacher_correct'] += supplied_correct
            counts['exact_teacher_input'] += same
            counts['same_input_different_actions'] += same and value['actions'] != reference['actions']
            if not correct:
                if same:
                    counts['wrong_despite_exact_teacher_input'] += 1
                elif supplied_correct:
                    counts['wrong_with_changed_input_and_correct_teacher_reference'] += 1
                else:
                    counts['wrong_with_changed_input_and_wrong_teacher_reference'] += 1
        assert sum(v for k, v in counts.items() if k.startswith('wrong_')) == counts['items']-counts['correct']
        assert counts['correct'] == original['panels'][panel]['answers'][arm+'-student-answers']['own']['correct']
        panels[panel][arm] = dict(counts=counts, contingency=[dict(
            same_teacher_input=a, own_correct=b, teacher_correct=c, items=n)
            for (a, b, c), n in sorted(table.items())])

report = dict(status='POSTHOC_NATIVE_INPUT_ERROR_DECOMPOSITION_NO_NEW_GENERATION',
              utc=datetime.now(timezone.utc).isoformat(), panels=panels,
              audit_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              original_report_sha256=hashlib.sha256((ROOT/'validation/variable_depth_writer_v11_complete_20261019.json').read_bytes()).hexdigest(),
              limits=['All original observations reused; no new evaluation, uncertainty estimate or success gate.',
                      'Exact equality compares original native reader prompt IDs, not decoded-text similarity.',
                      'Changed input plus a correct teacher response identifies a communication gap; it does not alone prove the generated calculation was wrong.',
                      'Teacher-conditioned performance references this frozen reader/representation, not every possible computation or a pure model-capacity effect.'])
(ROOT/'validation/variable_depth_writer_v11_error_decomposition_20261019.json').write_text(json.dumps(report, indent=2))
print(json.dumps(panels, indent=2))

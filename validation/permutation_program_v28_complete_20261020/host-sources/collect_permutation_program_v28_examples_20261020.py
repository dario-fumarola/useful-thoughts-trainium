"""Collect the sixteen prospectively fixed cases after the complete checked readout."""
from pathlib import Path
from collections import Counter
import argparse
import json

from tokenizers import Tokenizer
from train.permutation_program_answers import score_answer, execution_diagnostic
from train.permutation_program_data import question_from_dict
from weekend_thinking.artifacts import sha256_file, atomic_json

R = Path('/home/ubuntu/dario')
FIXED = R / 'local/diagnostics/permutation-program-v28-evaluation-CPU-v1/fixed-report-example-identities.json'
FIXED_SHA = '92cf5ebf8f98e55141d2c91ddba4e98628a1ba3fc4281a1c89c2d6cfa941b301'
read = lambda path: json.loads(Path(path).read_text())


def rows(path):
    with Path(path).open() as stream:
        for line in stream:
            yield json.loads(line)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('index', 'readout', 'independent-check', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    result = read(args.readout)
    check = read(args.independent_check)
    assert result['status'] == 'COMPLETE_FIXED_V28_ALL_TERMINAL_READOUT'
    assert check['status'] == 'PASS_INDEPENDENT_NATIVE_ACTIONS_NODES_WITNESSES_FAMILY_BOOTSTRAPS_AND_GATES'
    assert check['readout_sha256'] == sha256_file(args.readout)
    assert check['index_sha256'] == sha256_file(args.index)
    assert sha256_file(FIXED) == FIXED_SHA
    fixed = read(FIXED)
    assert fixed['status'] == 'FIXED_EXAMPLE_IDENTITIES_BEFORE_ANY_HELD_MODEL_OUTCOME'
    ids = fixed['ordered_item_ids']
    assert len(ids) == len(set(ids)) == 16
    cases = {item: dict(item_id=item, conditions={}) for item in ids}
    sources = {}
    main_counts = Counter()
    paths = read(args.index)
    assert len(paths) == len(set(paths)) == 23
    tokenizer = None
    for path in paths:
        root = Path(path)
        protocol = read(root / 'protocol.json')
        if protocol['panel'] != 'DEV_composition':
            continue
        if tokenizer is None:
            tokenizer = Tokenizer.from_file(str(Path(protocol['model_dir']) / 'tokenizer.json'))
        model = protocol['model']['id']
        source_key = model + ('/partial-controls' if protocol['diagnostic'] else '/main')
        assert result['input_protocols'][f"{model}/DEV_composition/{protocol['diagnostic']}"] == sha256_file(root / 'protocol.json')
        sources[source_key] = dict(protocol_sha256=sha256_file(root / 'protocol.json'),
                                  checkpoint=protocol['model'])
        gold = {v['item_id']: v for v in rows(root / 'gold.jsonl') if v['item_id'] in cases}
        if not protocol['diagnostic']:
            assert set(gold) == set(ids)
        prompts = {}
        for mode in protocol['modes']:
            source = root / f'prompts-{mode}.jsonl'
            assert sha256_file(source) == protocol['files'][source.name]
            prompts[mode] = {v['item_id']: v['prompt_ids'] for v in rows(source) if v['item_id'] in cases}
        for shard in range(4):
            folder = root / 'answers' / f'shard-{shard}'
            run = read(folder / 'run.json')
            assert run['status'] == 'complete_all_attempts'
            assert sha256_file(folder / 'attempts.jsonl') == run['attempts_sha256']
            for value in rows(folder / 'attempts.jsonl'):
                item = value['item_id']
                if item not in cases:
                    continue
                case = cases[item]
                g = gold[item]
                q = g['question']
                node = q['start']
                for operation in q['program']:
                    node = q[operation][node]
                assert g['answer'] == 'ABCDEFGHIJKLMNOP'[node]
                if 'question' in case:
                    assert case['question'] == q and case['expected'] == g['answer']
                else:
                    case.update(family_id=g['family_id'], question=q, expected=g['answer'])
                mode = value['mode']
                key = model + '/' + mode
                assert key not in case['conditions']
                entry = dict(input_text=tokenizer.decode(prompts[mode][item], skip_special_tokens=False),
                             official_score=score_answer(value, g['answer'], mode),
                             original_attempt=value)
                if mode == 'trace':
                    entry['execution_diagnostic'] = execution_diagnostic(value['text'], question_from_dict(q))
                case['conditions'][key] = entry
                if not protocol['diagnostic']:
                    main_counts[item] += 1
    assert set(main_counts) == set(ids) and set(main_counts.values()) == {18}
    atomic_json(args.output, dict(status='COMPLETE_FIXED_EIGHT_FAMILIES_BOTH_VARIANTS_ALL_MAIN_CONDITIONS',
        selected_before_outcomes=fixed, cases=[cases[item] for item in ids], sources=sources,
        readout_sha256=sha256_file(args.readout), independent_check_sha256=sha256_file(args.independent_check),
        collector_sha256=sha256_file(__file__), model_calls=0,
        limits=['All sixteen hash-selected cases are retained, regardless of success or failure.',
                'Original attempts and frozen official scores are copied; no new generation or scoring rules.',
                'Partial interventions appear only when a fixed case belongs to their separately frozen cohort.',
                'Examples illustrate the complete numerical readout; they do not replace it.']))
    print(json.dumps(dict(status='COLLECTED_ALL_FIXED_EXAMPLES', cases=len(cases), main_attempts=sum(main_counts.values()))))


if __name__ == '__main__':
    main()

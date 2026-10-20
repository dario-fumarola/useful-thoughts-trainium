"""Independent execution/split audit and tokenizer sizing; no model forward."""
from collections import Counter, defaultdict
import argparse
import hashlib
import json
from pathlib import Path
import re

from transformers import AutoTokenizer


def audit(data, model_dir):
    data = Path(data)
    read = lambda p: json.loads(p.read_text())
    rows = lambda p: [json.loads(v) for v in p.read_text().splitlines()]
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    manifest = read(data/'manifest.json')
    assert manifest['status'] == 'CPU_DATA_ONLY_NO_TRAINING_OR_EVALUATION_RELEASE'
    assert all(sha(data/name) == digest for name, digest in manifest['files'].items())
    tok = AutoTokenizer.from_pretrained(model_dir, local_files_only=True, trust_remote_code=False)
    nodes = 'ABCDEFGHIJKLMNOP'
    graph_splits = {}; item_ids = set(); family_splits = {}; program_sets = {}
    token_lengths = defaultdict(list); trace_lengths = defaultdict(set)
    summaries = {}; examples = []
    for split in ('TRAIN', 'DEV_interpolation', 'DEV_composition', 'DEV_depth'):
        labels = rows(data/f'{split}-labels.jsonl')
        questions = rows(data/f'{split}-questions.jsonl')
        assert len(labels) == len(questions) == manifest['inventories'][split]['questions']
        families = defaultdict(list); program_sets[split] = set()
        input_texts = []; direct_prefixes = []; work_prefixes = []; full_prefixes = []
        for row, policy in zip(labels, questions):
            assert set(policy) == {'item_id', 'family_id', 'question'}
            assert all(policy[k] == row[k] for k in policy)
            q = policy['question']
            assert set(q) == {'r', 's', 'start', 'program'}
            assert row['item_id'] not in item_ids; item_ids.add(row['item_id'])
            fid = row['family_id']; families[fid].append(row)
            assert family_splits.setdefault(fid, split) == split
            maps = {op: dict(zip(nodes, (nodes[i] for i in q[op]))) for op in ('r', 's')}
            assert all(sorted(q[op]) == list(range(16)) for op in maps)
            state = nodes[q['start']]; expected_states = [state]; expected_lines = []
            for op in q['program']:
                next_state = maps[op][state]
                expected_lines.append(f'{op}: {state} -> {next_state}\n')
                state = next_state; expected_states.append(state)
            assert row['answer'] == state
            assert row['states'] == [nodes.index(s) for s in expected_states]
            assert len(set(expected_states)) == len(expected_states)
            assert row['full_trace'] == ''.join(expected_lines)
            assert row['partial_trace'] == ''.join(expected_lines[:-1])
            assert state not in row['partial_trace']
            gid = hashlib.sha256(json.dumps([q['r'], q['s']], separators=(',', ':')).encode()).hexdigest()
            assert gid == row['graph_id'] and gid not in graph_splits
            graph_splits[gid] = split
            motif = any(m in q['program'] for m in ('rsrs', 'srsr'))
            assert motif == (split == 'DEV_composition')
            depth = len(q['program']); program_sets[split].add(q['program'])
            assert 7 <= depth <= 10 if split == 'DEV_depth' else 1 <= depth <= 6
            # Reconstruct text without importing the producer's prompt/executor.
            text = 'Transition table:\nNode r s\n' + ''.join(
                f'{n} {maps["r"][n]} {maps["s"][n]}\n' for n in nodes)
            text += f'Start: {nodes[q["start"]]}\nProgram: ' + ' '.join(q['program']) + '\n'
            direct = text + 'Final answer:\n'; work = text + 'Work:\n'
            full = work + row['full_trace'] + 'Final answer:\n'
            input_texts.append(text); direct_prefixes.append(direct)
            work_prefixes.append(work); full_prefixes.append(full)
        assert len(families) == manifest['inventories'][split]['families']
        for paired in families.values():
            assert len(paired) == 2
            a, b = sorted(paired, key=lambda v: v['variant'] != 'original')
            assert {a['variant'], b['variant']} == {'original', 'counterfactual'}
            qa, qb = a['question'], b['question']
            assert qa['program'] == qb['program'] and qa['start'] == qb['start']
            assert a['states'][:-1] == b['states'][:-1] and a['answer'] != b['answer']
            differences = {(op, i) for op in ('r', 's') for i in range(16) if qa[op][i] != qb[op][i]}
            used = set(zip(qa['program'], a['states']))
            assert len(differences) == 2 and len(differences & used) == 1
        # Batched tokenizer only; labels never go through a model.
        for start in range(0, len(labels), 128):
            batch = labels[start:start+128]
            kwargs = dict(add_special_tokens=False, truncation=False, return_attention_mask=False)
            direct = tok(direct_prefixes[start:start+128], **kwargs)['input_ids']
            work = tok(work_prefixes[start:start+128], **kwargs)['input_ids']
            full = tok(full_prefixes[start:start+128], **kwargs)['input_ids']
            completed = tok([t + v['answer'] for t, v in zip(full_prefixes[start:start+128], batch)], **kwargs)['input_ids']
            for row, dp, wp, fp, answer in zip(batch, direct, work, full, completed):
                assert answer[:len(fp)] == fp and answer[len(fp):] == tok.encode(row['answer'], add_special_tokens=False)
                assert len(answer[len(fp):]) == 1
                # Stable Work header boundary; all scratch states are targets.
                assert fp[:len(wp)] == wp
                depth = len(row['question']['program'])
                token_lengths['direct_prompt'].append(len(dp))
                token_lengths['work_prompt'].append(len(wp))
                token_lengths['full_training_including_EOS'].append(len(answer)+1)
                trace_ids = tok.encode(row['full_trace'], add_special_tokens=False)
                trace_lengths[depth].add(len(trace_ids))
                assert len(trace_ids) <= 128
        summaries[split] = dict(questions=len(labels), independent_families=len(families),
            depths=dict(Counter(len(v['question']['program']) for v in labels)))
        if split == 'TRAIN':
            # Fixed hash selection, not output-conditioned examples.
            examples = sorted(labels, key=lambda v: hashlib.sha256(
                ('permutation-v28-fixed-TRAIN-examples:'+v['item_id']).encode()).hexdigest())[:4]
    assert program_sets['TRAIN'].isdisjoint(program_sets['DEV_composition'])
    assert program_sets['TRAIN'].isdisjoint(program_sets['DEV_depth'])
    quantiles = {}
    for k, values in token_lengths.items():
        values.sort()
        quantiles[k] = {str(p): values[round(p*(len(values)-1))] for p in (0, .5, .95, 1)}
    return dict(status='PASS_INDEPENDENT_CPU_EXECUTION_COUNTERFACTUAL_SPLIT_AND_TOKEN_BUDGETS',
        inventories=summaries, unique_graphs=len(graph_splits), independent_families=len(family_splits),
        token_lengths=quantiles, full_trace_tokens_by_depth={k: sorted(v) for k, v in sorted(trace_lengths.items())},
        existing_FIM_pad_id=tok.get_vocab().get('<|fim_pad|>'),
        pause_status='Candidate trained constant-token input only; no pause implementation or hardware result.',
        manifest_sha256=sha(data/'manifest.json'), checker_sha256=sha(Path(__file__)),
        fixed_TRAIN_examples=examples,
        limits=['No model forward, training, hardware qualification or learned capability result.',
                'New full maps, not novel elementary edges or an unknown pretraining algorithm.',
                'Observed paths are simple and lengths1..10; cycles/arbitrary execution are outside this laboratory.',
                'Paired variants share a family; do not use them as independent bootstrap observations.',
                'Trained pauses and generated states have different execution graphs/latency even at equal positions.'])


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
    print(json.dumps({k: v for k, v in result.items() if k != 'fixed_TRAIN_examples'}, indent=2))

"""Verified variable-depth computation, not a natural-language pretraining corpus.

The policy receives only two maps, an initial node and an operation sequence.
Labels are exact executions of those inputs. Paired counterfactuals share a
family ID and must never be counted as independent evaluation observations.
"""
from collections import Counter
from dataclasses import asdict, dataclass
from functools import lru_cache
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import random


SCHEMA = 'permutation-program-v28-cpu-v1'
NODES = tuple('ABCDEFGHIJKLMNOP')
HELD_MOTIFS = ('rsrs', 'srsr')
SPLITS = ('TRAIN', 'DEV_interpolation', 'DEV_composition', 'DEV_depth')


@dataclass(frozen=True, slots=True)
class Question:
    r: tuple[int, ...]
    s: tuple[int, ...]
    start: int
    program: str

    def __post_init__(self):
        expected = list(range(len(NODES)))
        if (type(self.r) is not tuple or type(self.s) is not tuple
                or sorted(self.r) != expected or sorted(self.s) != expected
                or any(type(v) is not int for v in self.r + self.s)
                or type(self.start) is not int or not 0 <= self.start < len(NODES)
                or type(self.program) is not str or not self.program
                or set(self.program) - {'r', 's'}):
            raise ValueError('Two complete permutations, a valid start and r/s program required')


def question_from_dict(value):
    if type(value) is not dict or set(value) != {'r', 's', 'start', 'program'}:
        raise ValueError('Only the explicit available-input projection is accepted')
    return Question(tuple(value['r']), tuple(value['s']), value['start'], value['program'])


def execute(question):
    if type(question) is not Question:
        raise TypeError('Execution accepts a Question, not a labelled training row')
    path = [question.start]
    for op in question.program:
        path.append(getattr(question, op)[path[-1]])
    return tuple(path)


def render(question):
    if type(question) is not Question:
        raise TypeError('Prompt accepts only a Question')
    lines = ['Transition table:', 'Node r s']
    lines.extend(f'{node} {NODES[question.r[i]]} {NODES[question.s[i]]}'
                 for i, node in enumerate(NODES))
    lines.extend([f'Start: {NODES[question.start]}',
                  'Program: ' + ' '.join(question.program)])
    return '\n'.join(lines) + '\n'


def trace(question, *, include_final=True):
    path = execute(question)
    end = len(question.program) if include_final else len(question.program) - 1
    return ''.join(f'{op}: {NODES[a]} -> {NODES[b]}\n'
                   for op, a, b in zip(question.program[:end], path, path[1:]))


def graph_id(question):
    return hashlib.sha256(json.dumps([question.r, question.s],
                                     separators=(',', ':')).encode()).hexdigest()


@lru_cache(maxsize=None)
def programs(depth, held_motifs=False):
    if type(depth) is not int or not 1 <= depth <= 10:
        raise ValueError('Declared depths are one through ten')
    values = [''.join(v) for v in itertools.product('rs', repeat=depth)]
    return tuple(v for v in values
                 if any(m in v for m in HELD_MOTIFS) == held_motifs)


def make_question(rng, depth, *, held_motifs=False):
    choices = programs(depth, held_motifs)
    if not choices:
        raise ValueError('No programs satisfy the declared composition condition')
    program = rng.choice(choices)
    # Simple observed paths make a final-edge counterfactual exact. This is an
    # explicit restricted task distribution; cycles are not covered by this test.
    for _ in range(4096):
        r, s = list(range(len(NODES))), list(range(len(NODES)))
        rng.shuffle(r); rng.shuffle(s)
        question = Question(tuple(r), tuple(s), rng.randrange(len(NODES)), program)
        path = execute(question)
        if len(set(path)) == len(path):
            return question
    raise RuntimeError('Bounded simple-path sampling failed; do not change the split silently')


def counterfactual(question, rng):
    """Swap two map entries, exactly one used on the original execution path.

    Earlier states remain identical; the last state changes. Both complete maps
    remain permutations. Neither version exposes the paired example to the policy.
    """
    path = execute(question)
    if len(set(path)) != len(path):
        raise ValueError('The prospective counterfactual requires a simple observed path')
    op = question.program[-1]
    table = list(getattr(question, op))
    alternatives = [v for v in range(len(NODES)) if v not in path]
    if not alternatives:
        raise ValueError('No fresh terminal node remains')
    other = table.index(rng.choice(alternatives))
    last = path[-2]
    table[last], table[other] = table[other], table[last]
    changed = Question(tuple(table) if op == 'r' else question.r,
                       tuple(table) if op == 's' else question.s,
                       question.start, question.program)
    new_path = execute(changed)
    assert new_path[:-1] == path[:-1] and new_path[-1] != path[-1]
    assert len(set(new_path)) == len(new_path)
    return changed


def family(split, index, seed):
    if split not in SPLITS or type(index) is not int or index < 0:
        raise ValueError('Declared split and nonnegative family index required')
    key = f'{SCHEMA}:{seed}:{split}:{index}'
    family_id = hashlib.sha256(key.encode()).hexdigest()
    rng = random.Random(int(family_id, 16))
    if split == 'DEV_depth':
        depth = 7 + index % 4
    elif split == 'DEV_composition':
        depth = 4 + index % 3
    else:
        depth = 1 + index % 6
    original = make_question(rng, depth, held_motifs=split == 'DEV_composition')
    changed = counterfactual(original, rng)
    result = []
    for variant, question in [('original', original), ('counterfactual', changed)]:
        path = execute(question)
        result.append(dict(schema=SCHEMA, split=split, family_id=family_id,
            variant=variant, item_id=family_id + ':' + variant,
            question=asdict(question), graph_id=graph_id(question),
            answer=NODES[path[-1]], full_trace=trace(question),
            partial_trace=trace(question, include_final=False), states=list(path)))
    return result


def generate(*, train_families=16384, dev_families=1024, seed=2026102028):
    if any(type(n) is not int or n <= 0 for n in (train_families, dev_families)):
        raise ValueError('Positive explicit family counts required')
    result = {}; seen_graphs = set(); seen_inputs = set()
    for split in SPLITS:
        values = []
        for i in range(train_families if split == 'TRAIN' else dev_families):
            for row in family(split, i, seed):
                question = question_from_dict(row['question'])
                assert row['graph_id'] not in seen_graphs
                seen_graphs.add(row['graph_id'])
                text = render(question)
                assert text not in seen_inputs
                seen_inputs.add(text)
                values.append(row)
        result[split] = values
    return result


def write_dataset(output, **kwargs):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    data = generate(**kwargs)
    files = {}; inventories = {}
    for split, values in data.items():
        # Policy inputs are separate from offline supervision, even for a
        # fully deterministic, prefix-computable labelling procedure.
        questions = [dict(item_id=v['item_id'], family_id=v['family_id'],
                          question=v['question']) for v in values]
        for suffix, records in [('questions', questions), ('labels', values)]:
            target = output / f'{split}-{suffix}.jsonl'
            target.write_text(''.join(json.dumps(v, separators=(',', ':')) + '\n'
                                      for v in records))
            files[target.name] = hashlib.sha256(target.read_bytes()).hexdigest()
        inventories[split] = dict(questions=len(values), families=len(values)//2,
            depths=dict(Counter(len(v['question']['program']) for v in values)),
            answers=dict(Counter(v['answer'] for v in values)))
    manifest = dict(schema=SCHEMA, status='CPU_DATA_ONLY_NO_TRAINING_OR_EVALUATION_RELEASE',
        arguments=kwargs, files=files, inventories=inventories,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        limits=['Synthetic mechanism laboratory, not general-language pretraining.',
                'Simple observed paths only; no claim of cycle or arbitrary program execution.',
                'Paired variants share a family; bootstrap families rather than variants.',
                'Full trace contains the final state; supplied-oracle controls must omit the final operation.',
                'No model/tokenizer or hardware qualification has run.'])
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--train-families', type=int, default=16384)
    parser.add_argument('--dev-families', type=int, default=1024)
    parser.add_argument('--seed', type=int, default=2026102028)
    args = vars(parser.parse_args())
    print(json.dumps(write_dataset(**args), indent=2))

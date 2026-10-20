"""Execution, counterfactual, information-boundary and held-program invariants."""
from dataclasses import asdict
import json
import random

import pytest

from train.permutation_program_data import (HELD_MOTIFS, NODES, Question,
    counterfactual, execute, family, generate, graph_id, programs,
    question_from_dict, render, trace, write_dataset)


def independent_path(q):
    # Compose explicit dictionaries, independent of the generator's indexing loop.
    maps = {name: dict(zip(NODES, (NODES[i] for i in getattr(q, name))))
            for name in ('r', 's')}
    state = NODES[q.start]; result = [state]
    for operation in q.program:
        state = maps[operation][state]; result.append(state)
    return result


def test_known_execution_and_partial_exclusion():
    q = Question(tuple((i+1) % 16 for i in range(16)),
                 tuple((i+3) % 16 for i in range(16)), 0, 'rsr')
    assert independent_path(q) == ['A', 'B', 'E', 'F']
    assert execute(q) == (0, 1, 4, 5)
    assert trace(q) == 'r: A -> B\ns: B -> E\nr: E -> F\n'
    assert trace(q, include_final=False) == 'r: A -> B\ns: B -> E\n'
    one = Question(q.r, q.s, q.start, 'r')
    assert trace(one, include_final=False) == ''


@pytest.mark.parametrize('split', ['TRAIN', 'DEV_interpolation', 'DEV_composition', 'DEV_depth'])
def test_paired_intervention_changes_only_the_last_executed_transition(split):
    for i in range(48):
        a, b = family(split, i, 2718)
        q, changed = (question_from_dict(v['question']) for v in (a, b))
        original, new = independent_path(q), independent_path(changed)
        assert original[:-1] == new[:-1] and original[-1] != new[-1]
        assert a['answer'] == original[-1] and b['answer'] == new[-1]
        assert a['family_id'] == b['family_id'] and a['item_id'] != b['item_id']
        assert q.start == changed.start and q.program == changed.program
        differences = [(op, j) for op in ('r', 's') for j in range(16)
                       if getattr(q, op)[j] != getattr(changed, op)[j]]
        assert len(differences) == 2
        used = {(op, j) for op, j in zip(q.program, execute(q))}
        assert len(set(differences) & used) == 1
        assert all(sorted(getattr(changed, op)) == list(range(16)) for op in ('r', 's'))


def test_available_input_boundary_and_roundtrip():
    row = family('TRAIN', 7, 101)[0]
    q = question_from_dict(row['question']); before = render(q)
    row.update(answer='POISON', full_trace='future', partial_trace='future', states=[])
    assert render(question_from_dict(row['question'])) == before
    assert question_from_dict(json.loads(json.dumps(asdict(q)))) == q
    with pytest.raises(ValueError):
        question_from_dict(row)
    with pytest.raises(TypeError):
        render(row)


def test_split_programs_graphs_and_lengths_are_separate():
    data = generate(train_families=48, dev_families=24, seed=311)
    train_programs = {v['question']['program'] for v in data['TRAIN']}
    graphs = [v['graph_id'] for rows in data.values() for v in rows]
    assert len(graphs) == len(set(graphs))
    for split, rows in data.items():
        for v in rows:
            q = question_from_dict(v['question']); path = independent_path(q)
            assert len(set(path)) == len(path)
            assert v['answer'] == path[-1]
            assert v['graph_id'] == graph_id(q)
            if split == 'DEV_composition':
                assert any(m in q.program for m in HELD_MOTIFS)
                assert q.program not in train_programs and 4 <= len(q.program) <= 6
            else:
                assert not any(m in q.program for m in HELD_MOTIFS)
                assert (7 <= len(q.program) <= 10) if split == 'DEV_depth' else (1 <= len(q.program) <= 6)


def test_deterministic_data_and_separate_question_files(tmp_path):
    assert family('TRAIN', 3, 17) == family('TRAIN', 3, 17)
    assert family('TRAIN', 3, 17) != family('TRAIN', 3, 18)
    manifest = write_dataset(tmp_path/'one', train_families=12, dev_families=6, seed=3)
    other = write_dataset(tmp_path/'two', train_families=12, dev_families=6, seed=3)
    assert manifest == other
    for p in (tmp_path/'one').glob('*questions.jsonl'):
        for line in p.read_text().splitlines():
            row = json.loads(line)
            assert set(row) == {'item_id', 'family_id', 'question'}
            assert set(row['question']) == {'r', 's', 'start', 'program'}
    with pytest.raises(FileExistsError):
        write_dataset(tmp_path/'one', train_families=1, dev_families=1)


def test_invalid_inputs_fail_without_sampling_or_coercing_labels():
    identity = tuple(range(16))
    for args in [(identity, identity, True, 'r'),
                 (identity, identity, 16, 'r'),
                 (identity, identity, 0, ''),
                 (identity, identity, 0, 'future'),
                 (identity[:-1], identity, 0, 'r')]:
        with pytest.raises(ValueError):
            Question(*args)
    with pytest.raises(ValueError):
        counterfactual(Question(identity, identity, 0, 'rr'), random.Random(2))
    with pytest.raises(ValueError):
        programs(0)

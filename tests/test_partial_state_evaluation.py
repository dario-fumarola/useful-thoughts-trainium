import pytest

from train.partial_state_data import row
from train.partial_state_evaluation import donor_permutation, prompt_tokens, question_projection, reader_tokens, score_partial_state, teacher_controls


class Tokenizer:
    def encode(self, text, **kwargs):
        return [ord(c) for c in text]

    def decode(self, ids, **kwargs):
        return ''.join(chr(i) for i in ids)


TREE = ('*', ('+', 37, 18), ('-', 92, 57))


def test_generation_projection_rejects_supervised_row_and_never_contains_gold():
    r = row(TREE, 'TRAIN'); q = question_projection(r); tok = Tokenizer()
    assert set(q) == {'item_id', 'panel', 'expression_tree'}
    assert '1925' not in tok.decode(prompt_tokens(q, tok, cot=True))
    with pytest.raises(ValueError):
        prompt_tokens(r, tok)


@pytest.mark.parametrize('lengths', [[1,1], [1,2], [1,2,2], [1,1,2,3,3], [1,1,2,2]])
def test_donors_are_an_intact_derangement_including_singleton_bins(lengths):
    keys = [str(i) for i in range(len(lengths))]
    d = donor_permutation(lengths, keys)
    assert sorted(d) == list(range(len(keys))) and all(i != j for i,j in enumerate(d))
    assert sorted(lengths[j] for j in d) == sorted(lengths)


def test_controls_omit_root_change_one_partial_value_and_preserve_other_note():
    tok = Tokenizer(); qs = [question_projection(row(t, 'TRAIN')) for t in [TREE, ('*', ('+', 12, 24), ('-', 87, 39))]]
    controls = teacher_controls(qs, tok)
    a, b = controls
    teacher = tok.decode(a['notes']['teacher']); corrupt = tok.decode(a['notes']['corrupt'])
    assert teacher == '37 + 18 = 55\n92 - 57 = 35\n'
    assert corrupt == '37 + 18 = 55\n92 - 57 = 36\n'
    assert a['notes']['other_teacher'] == b['notes']['teacher']
    assert '1925' not in teacher and '55 * 35' not in teacher
    assert tok.decode(reader_tokens(qs[0], a['notes']['teacher'], tok)).endswith(teacher + 'Answer:\n')


def test_student_native_content_is_preserved_with_only_answer_separator():
    tok = Tokenizer(); q = question_projection(row(TREE, 'TRAIN')); note = tok.encode('unfinished 55')
    ids = reader_tokens(q, note, tok); prefix = prompt_tokens(q,tok,cot=True)
    assert ids[len(prefix):len(prefix)+len(note)] == note
    assert tok.decode(ids).endswith('unfinished 55\nAnswer:\n')
    with pytest.raises(ValueError):
        reader_tokens(q, note+[151643],tok)


def test_exact_partial_state_is_distinct_from_arithmetic_truth_and_natural_stop():
    text = '37 + 18 = 55\n92 - 57 = 35\n'
    good = score_partial_state(text,TREE,stop_type='token:151643')
    assert good['canonical_and_natural_stop'] and not good['root_operation_transmitted']
    cap = score_partial_state(text,TREE,stop_type='length_cap')
    assert cap['canonical_nonroot_exact'] and not cap['canonical_and_natural_stop']
    root = score_partial_state(text+'55 * 35 = 1925\n',TREE,stop_type='token:151643')
    assert root['root_operation_transmitted'] and not root['canonical_nonroot_exact']
    other = score_partial_state('2 + 2 = 4\n',TREE,stop_type='token:151643')
    assert other['all_stated_equations_correct'] and not other['canonical_nonroot_exact']
    assert not score_partial_state('Compute carefully',TREE,stop_type='token:151643')['equations_supported']

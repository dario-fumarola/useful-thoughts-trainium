import pytest
from train.permutation_program_answers import parse_answer, score_answer, execution_diagnostic
from train.permutation_program_data import Question


def test_exact_symbol_in_designated_complete_field_only():
    assert parse_answer('A', 'direct', terminated=True).answer == 'A'
    assert parse_answer('P\n', 'pause').answer == 'P'
    assert parse_answer('\n a \n', 'direct').answer == 'A'
    assert parse_answer('Final answer: A', 'direct', terminated=True).answer == 'A'
    assert parse_answer('Final answer: A\n', 'trace').answer == 'A'
    assert parse_answer('r: A -> B\nFinal answer:\nC', 'trace', terminated=True).answer == 'C'
    assert parse_answer('A', 'direct', capped=True).status == 'unsupported'
    assert parse_answer('r: A -> B\n', 'trace', terminated=True).status == 'unsupported'
    for text in ['A or B\n', 'Answer: A\n', 'Z\n', 'A\nB\n']:
        assert parse_answer(text, 'direct', terminated=True).status == 'unsupported'
    assert parse_answer('Final answer:\nA\nFinal answer:\nB', 'trace', terminated=True).status == 'unsupported'


def test_caps_errors_and_unsupported_stay_in_denominator():
    for stop, status, expected in [('token:151643','ok',1), ('qa_answer_line','ok',1),
                                    ('length_cap','ok',0), ('token:151643','error',0)]:
        value=score_answer(dict(text='A\n',stop_type=stop,status=status),'A','direct')
        assert value['score']==expected
    assert score_answer(dict(text='B',stop_type='token:151643'),'A','direct')['score']==0
    with pytest.raises(ValueError):
        score_answer(dict(text='A',stop_type='some_new_stop'),'A','direct')


def test_trace_checks_distinguish_local_correctness_from_executing_requested_program():
    r=tuple(list(range(1,16))+[0]);s=tuple(reversed(range(16)))
    q=Question(r,s,0,'rs')
    good='r: A -> B\ns: B -> O\nFinal answer:\nO'
    assert execution_diagnostic(good,q)['complete_correct_execution']
    short=execution_diagnostic('r: A -> B\nFinal answer:\nO',q)
    assert short['correct_execution_prefix_steps']==1 and not short['complete_correct_execution']
    wrong=execution_diagnostic('s: A -> P\nr: P -> A\nFinal answer:\nA',q)
    assert wrong['locally_correct_transitions']==2 and wrong['correct_execution_prefix_steps']==0
    assert not wrong['complete_correct_execution']
    malformed=execution_diagnostic('I think O.\nFinal answer:\nO',q)
    assert not malformed['syntactically_valid']

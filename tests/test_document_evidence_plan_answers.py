import pytest

from train.document_evidence_plan_answers import parse_evidence, annotated_support_overlap
from train.document_qa_answers import parse_answer


def test_valid_pointer_order_is_not_required_to_match_teacher_order():
    text='Evidence: [3] [1]\nFinal answer:\nParis\n'
    parsed=parse_evidence(text, 4)
    assert parsed.status=='complete' and parsed.indices==(3,1)
    assert parse_answer(text,'prior').answer=='Paris'
    assert annotated_support_overlap(parsed,[1,3])['exact_match']==1
    assert annotated_support_overlap(parsed,[2,4])['recall']==0


def test_empty_control_has_complete_answer_but_no_generated_selection():
    text='Evidence:\nFinal answer:\nParis\n'
    assert parse_evidence(text,4).status=='empty'
    assert parse_answer(text,'prior').status=='complete'


@pytest.mark.parametrize('field', ['Evidence: [0]','Evidence: [5]',
    'Evidence: [1] [1]','Evidence: 1, 2','Evidence: [1]\nUnsupported claim.',
    'An unsupported claim.\nEvidence: [1]','Evidence: [1]\nEvidence: [2]'])
def test_invalid_selections_never_discard_otherwise_complete_answers(field):
    text=field+'\nFinal answer:\nParis\n'
    assert parse_evidence(text,4).status=='unsupported'
    assert parse_answer(text,'prior').answer=='Paris'


def test_parser_uses_no_gold_or_canonical_order_and_does_not_credit_missing_final():
    assert parse_evidence('Evidence: [1]\n',4).status=='unsupported'
    assert parse_evidence('Final answer:\nParis\n',4).status=='unsupported'
    assert parse_evidence('Evidence: [2]\nFinal answer:\nWrong\n',4).status=='complete'
    assert parse_evidence('Evidence: [2]\nFinal answer:\nA\nFinal answer:\nB\n',4).status=='unsupported'

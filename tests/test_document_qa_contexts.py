import copy
import pytest

from train.document_qa_contexts import context,question_only,omit_evidence,trace_from_output


def question():
    return dict(item_id='example',question='Where did the inventor live?',contexts=[
        dict(title='Inventor',text='Ada invented it. She lived in Lyon.'),
        dict(title='Other',text='Ben lived in Bern.')])


@pytest.mark.parametrize('mode',['direct','prior','empty','reference_cot'])
def test_all_original_passages_present_and_unprojected_gold_rejected(mode):
    q=question();text=context(q,mode)
    assert 'Ada invented it. She lived in Lyon.' in text and 'Ben lived in Bern.' in text
    q['answer']='Lyon'
    with pytest.raises(ValueError):context(q,mode)


def test_question_and_evidence_controls_are_explicit_unchanged_index_interventions():
    q=question();original=copy.deepcopy(q);ablated=omit_evidence(q,0)
    assert q==original and ablated['contexts'][1]==q['contexts'][1]
    assert ablated['contexts'][0]['title']==q['contexts'][0]['title']
    assert 'Ada' not in context(ablated,'prior')
    only=question_only(q);assert only['question']==q['question'] and 'Lyon' not in context(only,'prior')


def test_generated_trace_is_not_selected_by_the_answer():
    trace,has_boundary=trace_from_output('I considered Lyon.\nFinal answer: Bern\n')
    assert trace=='I considered Lyon.' and has_boundary
    assert trace_from_output('unfinished reasoning')[0]=='unfinished reasoning'

import copy

import pytest

from train.document_evidence_data import question_projection,render,supplied_support


class InputOnly(dict):
    def __getitem__(self,key):
        if key not in {'id','composed_question_text','contexts','wikipedia_title','paragraph_text'}:
            raise AssertionError('Generation read an offline label: '+key)
        return super().__getitem__(key)


def row():
    return dict(id='document-question',composed_question_text='Which city hosted the event?',
        question_text='[[CQS]] 0 [[CQE]] hidden composition template',
        answer_text='Lyon',decomposed_instances=['gold hidden step'],
        contexts=[dict(wikipedia_title='Event',paragraph_text='The event was held in Lyon.',
                       is_supporting=True,retrieval_score=1.,primary=True),
                  dict(wikipedia_title='Other event',paragraph_text='A different event took place in Bern.',
                       is_supporting=False,retrieval_score=0.,primary=False)])


def test_generation_projection_never_reads_gold_or_retrieval_fields():
    value=row();value['contexts']=[InputOnly(c) for c in value['contexts']]
    q=question_projection(InputOnly(value));text=render(q)
    assert 'Lyon' in text and 'Bern' in text  # BOTH supplied documents remain
    assert 'gold hidden step' not in text and 'is_supporting' not in text
    changed=copy.deepcopy(value);changed.update(answer_text='poison',decomposed_instances=[])
    for c in changed['contexts']:c.update(is_supporting=False,retrieval_score=999.,primary=False)
    assert question_projection(changed)==q and render(question_projection(changed))==text


def test_offline_support_copies_only_the_visible_paragraphs():
    value=row();q=question_projection(value);support=supplied_support(value,q)
    assert support=='[1] Event\nThe event was held in Lyon.'
    value['answer_text']='An incorrect hidden answer'
    assert supplied_support(value,q)==support
    with pytest.raises(ValueError):
        supplied_support(value,{**q,'question':'A different question'})


def test_render_rejects_unprojected_supervised_row():
    with pytest.raises(ValueError):render(row())
    q=question_projection(row());q['contexts'][0]['is_supporting']=True
    with pytest.raises(ValueError):render(q)


def test_raw_composition_program_is_not_a_natural_question_fallback():
    value=row();value['composed_question_text']='[[CQS]] 0 [[CQE]] A program'
    with pytest.raises(ValueError):question_projection(value)
    del value['composed_question_text']
    with pytest.raises(KeyError):question_projection(value)

import copy

import pytest

from train.document_evidence_plan_evaluate import prompt_text, generate
from train.document_evidence_plan_packing import FINAL


def question():
    return dict(item_id='question',question='Where is the inventor from?',
        contexts=[dict(title='Object',text='Ada invented the object.'),
                  dict(title='Ada',text='Ada is from Lyon.')])


class GoldGuard(dict):
    def __getitem__(self, key):
        if key != 'support_indices':
            raise AssertionError('Answer annotation read by prompt construction')
        return [1,2]


@pytest.mark.parametrize('mode',['direct','selection','empty','oracle'])
def test_gold_answer_never_enters_even_the_pointer_oracle(mode):
    q=question()
    text=prompt_text(q,mode,GoldGuard())
    assert 'POISON' not in text
    if mode=='oracle':
        assert text.endswith('Evidence: [1] [2]\n')
    elif mode=='empty':
        assert text.endswith('Evidence:\n')
    assert not text.endswith(FINAL)


def test_ordinary_modes_ignore_annotations_and_reject_unprojected_questions():
    q=question()
    for mode in ('direct','selection','empty'):
        assert prompt_text(q,mode)==prompt_text(q,mode,dict(answer='POISON',support_indices=[99]))
        injected=copy.deepcopy(q);injected['answer']='POISON'
        with pytest.raises(ValueError):
            prompt_text(injected,mode)


def test_unreleased_held_protocol_cannot_start_the_generation_engine(tmp_path,monkeypatch):
    from train import document_evidence_plan_evaluate as module
    monkeypatch.setattr(module.engine,'read',lambda path:dict(purpose='FIXED_HELD_EVIDENCE_PLAN_V23',release_file=None))
    called=[]
    monkeypatch.setattr(module.engine,'generate',lambda *args,**kwargs:called.append(True))
    with pytest.raises(ValueError):
        generate(tmp_path,0,'2099-01-01T00:00:00+00:00')
    assert not called

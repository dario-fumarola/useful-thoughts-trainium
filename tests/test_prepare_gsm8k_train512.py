"""Question selection must stay answer-free, disjoint and prefix preserving."""
import pytest
from scripts import prepare_gsm8k_train512 as p


def parent(questions):
    excluded=['gsm8k:train:00000']
    return dict(excluded_ids=excluded,salt='fixed',ordered_item_ids=[],
        excluded_question_normal_forms_sha256=p.parent_hash([p.normalized(questions[0])]))


def test_extension_preserves_old_order_and_skips_normalized_duplicates():
    questions=['Demo','demo','A question','a  QUESTION','B question','C question','D question']
    selection=parent(questions)
    old,_=p.select_questions(questions,selection,count=2)
    selection['ordered_item_ids']=[r['item_id'] for r in old]
    extended,_=p.select_questions(questions,selection,count=4)
    assert extended[:2]==old
    assert all(p.normalized(r['question'])!='demo' for r in extended)
    assert len({p.normalized(r['question']) for r in extended})==4


def test_changed_old_membership_or_shortfall_is_never_replaced():
    questions=['Demo','A','B'];selection=parent(questions)
    with pytest.raises(ValueError,match='quota'):p.select_questions(questions,selection,count=3)
    selection['ordered_item_ids']=['gsm8k:train:00000']
    with pytest.raises(ValueError,match='existing ordered'):p.select_questions(questions,selection,count=2)

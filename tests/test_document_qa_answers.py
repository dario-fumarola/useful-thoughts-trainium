import pytest

from train.document_qa_answers import parse_answer, score, normalize_answer, token_f1


@pytest.mark.parametrize('text,mode,answer',[
    ('Lyon\nUnscored later prose','direct','Lyon'),
    ('\n Lyon \n','direct','Lyon'),
    ('Step 1: Ada\nFinal answer:\nLyon\n','prior','Lyon'),
    ('Work\nfinal answer: Lyon\n','prior','Lyon'),
    ('Lyon','direct','Lyon'),
    ('Final answer:\nLyon','prior','Lyon'),
])
def test_complete_field_requires_boundary_and_is_independent_of_gold(text,mode,answer):
    p=parse_answer(text,mode,terminated=True)
    assert p.status=='complete' and p.answer==answer
    assert score(p,'Lyon')['canonical_em']==1
    assert score(p,'Bern')['canonical_em']==0


@pytest.mark.parametrize('text,mode,reason',[
    ('Lyo','direct','unterminated_cap'),
    ('Final answer:\nLyo','prior','unterminated_cap'),
    ('Step 1: Lyon is mentioned.','prior','missing_final_field'),
    ('Final answer: Bern\nFinal answer: Lyon\n','prior','multiple_final_fields'),
    ('','direct','empty_answer'),
])
def test_caps_missing_or_multiple_fields_are_not_rescued(text,mode,reason):
    p=parse_answer(text,mode,capped=True)
    assert p.status=='unsupported' and p.reason==reason
    assert all(v==0 for v in score(p,'Lyon',['Lyo']).values())


def test_first_designated_answer_wins_not_any_correct_word_in_output():
    p=parse_answer('Bern\nActually Lyon\n','direct',terminated=True)
    assert p.answer=='Bern' and score(p,'Lyon')['canonical_em']==0
    p=parse_answer('Lyon appeared in evidence\nFinal answer: Bern\n','prior',terminated=True)
    assert p.answer=='Bern' and score(p,'Lyon')['canonical_em']==0


def test_canonical_and_alias_scores_remain_distinct():
    p=parse_answer('NYC','direct',terminated=True);r=score(p,'New York City',['NYC'])
    assert r==dict(canonical_em=0,canonical_f1=0.,attributed_alias_em=1,attributed_alias_f1=1.)
    assert normalize_answer(' The, United States! ')=='united states'
    assert token_f1('New York','New York City')==.8


def test_partial_generation_is_pending_not_scored():
    p=parse_answer('Some reasoning\nFinal ans','prior')
    assert p.status=='pending' and all(v==0 for v in score(p,'Lyon').values())

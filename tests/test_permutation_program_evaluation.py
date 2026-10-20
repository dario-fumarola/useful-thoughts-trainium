from dataclasses import asdict
import pytest
from train.permutation_program_data import Question
from train.permutation_program_evaluation import available_item, partial_states, corrupt_partial, controls, prompt_ids


def item(family, variant, start=0, program='rs'):
    q=Question(tuple(list(range(1,16))+[0]),tuple(reversed(range(16))),start,program)
    return dict(item_id=f'{family}:{variant}',family_id=family,question=asdict(q))


def test_available_inputs_reject_labelled_rows_and_root_excluded_states():
    row=item('a','original');q=available_item(row)
    assert partial_states(q)=='r: A -> B\n'
    assert corrupt_partial(q)=='r: A -> C\n'
    assert partial_states(available_item(item('a','short',program='r'))) == ''
    for field in ['answer','states','full_trace']:
        with pytest.raises(ValueError):available_item(dict(row,**{field:'secret'}))


def test_donors_are_intact_deranged_families_with_same_depth():
    rows=[item(f,v,start=i) for i,f in enumerate('abc') for v in ('original','counterfactual')]
    result=controls(rows);lookup={row['item_id']:row for row in rows}
    for row in rows:
        value=result[row['item_id']]
        assert value['donor_family_id'] != row['family_id']
        donor=lookup[value['donor_item_id']]
        assert value['donor']==partial_states(available_item(donor))
    assert len({v['donor_item_id'] for v in result.values()})==len(rows)
    with pytest.raises(ValueError):controls(rows[:2])


def test_supplied_partial_context_has_the_actual_final_answer_header():
    class Tokenizer:
        def encode(self,text,add_special_tokens=False):return list(text.encode())
    row=item('a','original');q=available_item(row);note=partial_states(q)
    ids=prompt_ids(row,Tokenizer(),'partial',dict(partial=note))
    text=bytes(ids).decode()
    assert text.endswith('Work:\nr: A -> B\nFinal answer:\n')
    assert 'O' not in text.split('Work:\n')[1]

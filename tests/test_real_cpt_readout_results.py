"""Declared contrasts, both-order conjunction, and no dropped paired documents."""
from copy import deepcopy

import pytest

from train.real_cpt_readout_results import decisions, finite_logps, paired


def passing_order():
    gain=dict(difference_b_minus_a=.3,ci95=[.01,.6])
    return {**{k:deepcopy(gain) for k in ('raw_control','format_control','base','own_raw','own_empty','own_raw_control')},
        'ordinary_base':dict(difference_b_minus_a=.01,ci95=[-.01,.03]),
        'inference_execution_complete':True}


def test_recipe_absolute_retention_and_inference_remain_distinct_and_require_both_orders():
    data={str(s):passing_order() for s in (2026102029,2026102030)}
    assert decisions(data)['overall_CPT_criterion']
    data['2026102030']['base']['ci95'][0]=0
    out=decisions(data)
    assert out['training_recipe_pass_both_orders']
    assert not out['absolute_base_pass_both_orders']
    assert not out['overall_CPT_criterion']
    assert out['inference_gate_pass_both_orders']
    data['2026102029']['inference_execution_complete']=False
    assert not decisions(data)['inference_gate_pass_both_orders']
    with pytest.raises(ValueError):decisions({'2026102029':passing_order()})


def test_threshold_means_and_retention_upper_bound_are_both_used():
    data={str(s):passing_order() for s in (2026102029,2026102030)}
    data['2026102029']['raw_control']['difference_b_minus_a']=.249999
    data['2026102030']['ordinary_base']['ci95'][1]=.0300001
    result=decisions(data)
    assert not result['training_recipe_pass_both_orders']
    assert not result['retention_target_met_both_orders']
    assert not result['overall_CPT_criterion']


def test_paired_scores_reconstruct_hand_computed_mean_and_reject_missing_items():
    a=[dict(item_id=str(i),document_id='doc-'+str(i),score=0.) for i in range(4)]
    b=[dict(r,score=float(i)) for i,r in enumerate(a)]
    c=paired(a,b,name='known',metric='test')
    assert c['difference_b_minus_a']==1.5
    assert c['documents']==c['items']==4
    assert c['bootstrap_draws']==10000 and c['bootstrap_seed']==2026102029
    with pytest.raises(ValueError):paired(a,b[:-1],name='missing',metric='test')


def test_full_target_alignment_rejects_truncation_or_corrupt_summaries():
    row=dict(target_ids=[1,2,3],tokens=3,token_logps=[-.1,-.2,-.3],full_logp=-.6,K1_logp=-.1)
    assert finite_logps(row,[1,2,3])==row['token_logps']
    with pytest.raises(ValueError):finite_logps(row,[1,2])
    row['full_logp']=-.59
    with pytest.raises(ValueError):finite_logps(row,[1,2,3])

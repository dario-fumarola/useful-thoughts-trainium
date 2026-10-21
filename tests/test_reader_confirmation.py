import numpy as np

from train.reader_confirmation import ORDERS, BASELINES, contrast_columns, verdicts


def reports(calc=.4, natural=-.03):
    output={}
    for cohort,value in [('CONFIRM_calculation',calc),('CONFIRM_natural',natural)]:
        output[cohort]={f'{order}:minus-{baseline}':dict(mean=value,CI95=[value-.01,value+.01])
                        for order in ORDERS for baseline in BASELINES}
    return output


def test_fresh_calculation_confirmation_cannot_be_relabeled_a_natural_text_win():
    result=verdicts(reports())
    assert all(v['calculation_base'] and v['calculation_all_controls'] for v in result.values())
    assert all(not v['natural_gain'] and not v['natural_all_controls'] for v in result.values())
    assert all(v['natural_old_retention_guard'] for v in result.values())


def test_stronger_unassisted_controls_and_every_order_remain_in_the_verdict():
    r=reports(natural=.05)
    r['CONFIRM_calculation']['2026102146:minus-T_raw:position']['CI95']=[-.01,.05]
    result=verdicts(r)
    assert result['2026102141']['calculation_all_controls']
    assert not result['2026102146']['calculation_all_controls']
    assert all(v['natural_gain'] for v in result.values())
    assert set(result)=={str(i) for i in ORDERS}


def test_contrasts_pair_each_order_with_its_frozen_unassisted_head():
    totals={'B0':np.zeros(3)}
    for role in ('T_note','T_raw'):
        for mode in ('global','position'):totals[role+':'+mode]=np.ones(3)*.1
        for i,order in enumerate(ORDERS):totals[f'{role}-hard-order{order}']=np.ones(3)*(i+(role=='T_note'))
    columns=contrast_columns(totals)
    assert len(columns)==40
    assert all(np.array_equal(columns[f'{order}:minus-matched_raw'],np.ones(3)) for order in ORDERS)

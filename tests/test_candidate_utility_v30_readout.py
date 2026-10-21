from scripts.report_candidate_utility_v30 import decide


def statistic(mean, low):
    return dict(difference_b_minus_a=mean,ci95=[low,mean+1.])


def test_oracle_best_of_eight_cannot_rescue_negative_correctness_alignment():
    contrasts = {k:statistic(2.,1.) for k in
        ('best8_minus_raw','best8_minus_B0_raw','best8_minus_greedy','oracle_minus_corrupt')}
    gates = decide(contrasts,helpful_documents=100,mixed_documents=100,alignment=statistic(-1.,-2.))
    assert gates['opportunity'] and not gates['alignment'] and not all(gates.values())


def test_improving_degraded_reader_does_not_rescue_negative_base_contrast():
    contrasts = {k:statistic(2.,1.) for k in
        ('best8_minus_raw','best8_minus_B0_raw','best8_minus_greedy','oracle_minus_corrupt')}
    contrasts['best8_minus_B0_raw'] = statistic(-.1,-.5)
    gates = decide(contrasts,helpful_documents=100,mixed_documents=100,alignment=statistic(1.,.5))
    assert not gates['opportunity'] and not all(gates.values())


def test_too_few_mixed_documents_is_unidentifiable_not_a_pass():
    contrasts = {k:statistic(2.,1.) for k in
        ('best8_minus_raw','best8_minus_B0_raw','best8_minus_greedy','oracle_minus_corrupt')}
    gates = decide(contrasts,helpful_documents=100,mixed_documents=5,alignment=statistic(1.,.5))
    assert not gates['alignment']

import copy
import math

import pytest

from scripts.greedy_token_opportunity import analyze


def complementary_pair():
    # Each model is right about a different token. Both distributions are valid;
    # identical remaining positions contribute nothing to the comparison.
    common = dict(document_id="one-source", omitted_source_arrays={
        "target_ids": {"length": 16, "canonical_json_sha256": "same-target"}})
    return {
        ("item", "B0_raw"): dict(common, token_logps=[math.log(.9), math.log(.1)]
                                + [math.log(.5)] * 14),
        ("item", "greedy"): dict(common, token_logps=[math.log(.1), math.log(.9)]
                                + [math.log(.5)] * 14),
    }


def test_token_oracle_opportunity_is_not_an_observed_whole_span_gain():
    scores = complementary_pair()
    rows, summary = analyze(scores, expected_documents=1)
    assert summary["mean_greedy_minus_B0_nats_per16"] == pytest.approx(0)
    assert summary["mean_upper_nats_per16"] == pytest.approx(math.log(9))
    # Every fixed probability blend is bounded by the tokenwise oracle. A
    # positive ceiling alone does not identify a prospective selection policy.
    for g in (0, .01, .2, .5, .9, 1):
        actual = (math.log((1-g)*.9+g*.1) - math.log(.9)
                  + math.log((1-g)*.1+g*.9) - math.log(.1))
        assert actual <= rows[0]["unrestricted_hindsight_gain_nats"] + 1e-12


def test_matching_item_names_cannot_hide_different_scored_futures():
    scores = copy.deepcopy(complementary_pair())
    scores[("item", "greedy")]["omitted_source_arrays"] = {
        "target_ids": {"length": 16, "canonical_json_sha256": "other-target"}}
    with pytest.raises(ValueError, match="target/document mismatch"):
        analyze(scores, expected_documents=1)

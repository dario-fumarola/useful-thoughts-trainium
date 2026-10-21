import numpy as np
import pytest

from weekend_thinking.stats import cluster_bootstrap, paired_cluster_bootstrap


def test_unequal_clusters_keep_all_items_and_match_explicit_resampling():
    # Mean of cluster means would be 5; the declared item-weighted mean is 2.5.
    values = np.array([[0., 2.], [0., 3.], [0., 4.], [10., -1.]])
    result = cluster_bootstrap(values, ['large']*3+['small'], unit='source family',
                               repeats=200, seed=37, batch_size=17)
    rng = np.random.default_rng(37)
    expected = []
    for _ in range(200):
        selected = rng.integers(0, 2, 2)
        items = np.concatenate([np.arange(3) if cluster == 0 else np.array([3])
                                for cluster in selected])
        expected.append(values[items].mean(0))
    np.testing.assert_array_equal(result['mean'], values.mean(0))
    np.testing.assert_array_equal(result['draws'], expected)
    np.testing.assert_array_equal(result['CI95'], np.quantile(expected, [.025, .975], axis=0).T)
    assert result['metadata']['independent_clusters'] == 2
    assert result['metadata']['items'] == 4
    assert result['metadata']['resampling_unit'] == 'source family'


def test_unique_document_draws_preserve_original_generator_stream():
    values = np.random.default_rng(1).normal(size=(23, 4))
    old_rng = np.random.default_rng(71)
    expected = np.array([values[old_rng.integers(0, 23, 23)].mean(0)
                         for _ in range(200)])
    new_rng = np.random.default_rng(71)
    result = cluster_bootstrap(values, list(range(23)), unit='source document',
                               repeats=200, rng=new_rng, batch_size=13)
    np.testing.assert_array_equal(result['draws'], expected)
    assert new_rng.integers(0, 1000000) == old_rng.integers(0, 1000000)
    assert result['metadata']['random_source'] == 'supplied Generator'


def test_related_tokens_are_not_independent_observations():
    # Repeating 100 tokens from either of two documents must not narrow the CI
    # as if we had collected 200 independent documents.
    values = np.repeat([[-1.], [1.]], 100, axis=0)
    clustered = cluster_bootstrap(values, ['a']*100+['b']*100,
                                  unit='source document', repeats=200, seed=9)
    independent = cluster_bootstrap(values, list(range(200)),
                                    unit='token (deliberately wrong)', repeats=200, seed=9)
    assert clustered['CI95'][0].tolist() == [-1., 1.]
    assert np.ptp(independent['CI95'][0]) < .5


def test_row_pairing_wrapper_retains_legacy_cluster_weights_and_strict_pairing():
    a = [dict(item_id=str(i), document_id='a' if i < 3 else 'b', score=0.) for i in range(4)]
    b = [dict(row, score=10. if row['document_id']=='b' else 0.) for row in a]
    result = paired_cluster_bootstrap(a, list(reversed(b)), repeats=200, seed=37,
                                      unit='map/program family')
    assert result['difference_b_minus_a'] == 2.5
    assert result['ci95'] == [0., 10.]
    assert result['resampling_unit'] == 'map/program family'
    assert result['documents'] == 2
    with pytest.raises(ValueError):
        paired_cluster_bootstrap(a, b[:-1], repeats=200)
    with pytest.raises(ValueError):
        paired_cluster_bootstrap(a, b+[b[0]], repeats=200)


@pytest.mark.parametrize('values,ids,unit', [
    (np.ones((3, 1)), ['a', 'b'], 'document'),
    (np.ones((3, 1)), ['a']*3, 'document'),
    (np.ones((3, 1)), ['a', None, 'b'], 'document'),
    (np.ones((3, 1)), ['a', '', 'b'], 'document'),
    (np.ones((3, 1)), ['a', 'b', 'c'], ''),
    (np.ones(3), ['a', 'b', 'c'], 'document'),
    (np.array([[0.], [np.nan]]), ['a', 'b'], 'document'),
])
def test_bad_or_unnamed_resampling_units_fail(values, ids, unit):
    with pytest.raises(ValueError):
        cluster_bootstrap(values, ids, unit=unit, repeats=200)


def test_generator_and_seed_are_not_silently_overridden():
    with pytest.raises(ValueError):
        cluster_bootstrap(np.ones((2, 1)), ['a', 'b'], unit='document',
                          rng=np.random.default_rng(1), seed=9)


def test_internal_selector_wrapper_requires_complete_source_ids():
    from train.selector_optimization import bootstrap
    values = np.tile([.3, -.1], (512, 1))
    result = bootstrap(values, 2026102147, document_ids=[f'doc-{i}' for i in range(512)])
    assert result[0]['mean'] == pytest.approx(.3)
    assert result[0]['resampling']['independent_clusters'] == 512
    assert result[0]['resampling']['bootstrap_seed'] == 2026102147
    with pytest.raises(ValueError):
        bootstrap(values, 2026102147, document_ids=['same']*512)

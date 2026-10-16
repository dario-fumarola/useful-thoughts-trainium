"""Resume and recovery policy checks; these do not qualify device execution."""
import pytest

from weekend_thinking.runtime_policy import (checkpoint_due, require_runtime_policy,
                                             runtime_policy, validate_site_runtime)


def test_policy_excludes_secrets_and_partition_endpoints():
    policy = runtime_policy({'PJRT_DEVICE': 'NEURON', 'XLA_COMPILATION_CACHE_SIZE': '2',
        'NEURON_RT_VISIBLE_CORES': '0-31', 'NEURON_RT_ROOT_COMM_ID': '127.0.0.1:64194',
        'AWS_SECRET_ACCESS_KEY': 'must-not-be-recorded'})
    assert policy['environment']['XLA_COMPILATION_CACHE_SIZE'] == '2'
    assert policy['environment']['NEURON_NUM_RECENT_MODELS_TO_KEEP'] is None
    assert 'must-not-be-recorded' not in str(policy)
    assert 'NEURON_RT_VISIBLE_CORES' not in policy['environment']
    assert 'NEURON_RT_ROOT_COMM_ID' not in policy['environment']


def test_policy_rejects_changed_or_missing_cache_setting():
    policy = runtime_policy({'XLA_COMPILATION_CACHE_SIZE': '2'})
    for changed in (None, runtime_policy({'XLA_COMPILATION_CACHE_SIZE': '1'}), runtime_policy({})):
        with pytest.raises(ValueError, match='Runtime policy'):
            require_runtime_policy(policy, changed)


def test_slow_training_saves_before_first_quarter():
    assert not checkpoint_due(step=1, total_steps=2048, elapsed_seconds=43199,
                              interval_seconds=43200)
    assert checkpoint_due(step=2, total_steps=2048, elapsed_seconds=43200,
                          interval_seconds=43200)
    assert checkpoint_due(step=512, total_steps=2048, elapsed_seconds=1,
                          interval_seconds=43200)
    assert checkpoint_due(step=2048, total_steps=2048, elapsed_seconds=1,
                          interval_seconds=43200)
    for invalid in (0, 86400, 12.0):
        with pytest.raises(ValueError, match='Checkpoint interval'):
            checkpoint_due(step=1, total_steps=2048, elapsed_seconds=1,
                           interval_seconds=invalid)


def test_freeze_requires_complete_per_arm_runtime_and_finite_checkpoint_interval():
    policy = runtime_policy({'XLA_COMPILATION_CACHE_SIZE': '2'})
    site = dict(runtime_policy_by_arm=dict(vanilla=policy, thought=policy),
                checkpoint_interval_seconds=43200)
    validate_site_runtime(site)
    for invalid in ({}, dict(site, checkpoint_interval_seconds=None),
                    dict(site, runtime_policy_by_arm={'thought': policy})):
        with pytest.raises(ValueError):
            validate_site_runtime(invalid)

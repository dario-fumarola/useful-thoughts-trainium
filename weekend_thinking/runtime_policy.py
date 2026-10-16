"""Record performance-affecting runtime settings without collecting credentials.

Device ranges and communication endpoints belong to each launch's topology,
not this policy: paired arms can occupy different partitions. The policy is
shared across ranks and must survive an exact-resume comparison unchanged.
"""
from __future__ import annotations

import os


POLICY_ENVIRONMENT = (
    'PJRT_DEVICE', 'XLA_COMPILATION_CACHE_SIZE',
    'NEURON_NUM_RECENT_MODELS_TO_KEEP', 'NEURON_CC_FLAGS',
    'XLA_USE_BF16', 'XLA_DOWNCAST_BF16',
    'OMP_NUM_THREADS', 'MKL_NUM_THREADS',
)
MAX_CHECKPOINT_INTERVAL_SECONDS = 12 * 3600


def runtime_policy(environ=None):
    environment = os.environ if environ is None else environ
    # An absent setting stays null: do not mistake an assumed default for the
    # installed runtime's behavior (the legacy Neuron setting can be ignored).
    return dict(schema=1, environment={key: environment.get(key)
                                      for key in POLICY_ENVIRONMENT})


def require_runtime_policy(expected, actual):
    if expected != actual:
        raise ValueError('Runtime policy differs from the frozen or resumed run')


def validate_site_runtime(site):
    policies = site.get('runtime_policy_by_arm')
    if not isinstance(policies, dict) or set(policies) != {'vanilla', 'thought'}:
        raise ValueError('Resolve runtime_policy_by_arm for vanilla and thought before freezing')
    for policy in policies.values():
        if (not isinstance(policy, dict) or set(policy) != {'schema', 'environment'}
                or policy['schema'] != 1 or not isinstance(policy['environment'], dict)
                or set(policy['environment']) != set(POLICY_ENVIRONMENT)
                or any(value is not None and not isinstance(value, str)
                       for value in policy['environment'].values())):
            raise ValueError('Resolve each runtime policy with the complete allowlisted environment')
    # Use the same validation as the checkpoint decision, before reading assets.
    checkpoint_due(step=0, total_steps=1, elapsed_seconds=0,
                   interval_seconds=site.get('checkpoint_interval_seconds'))


def checkpoint_due(*, step, total_steps, elapsed_seconds, interval_seconds,
                   resume_probe=False):
    if type(interval_seconds) is not int or not 1 <= interval_seconds <= MAX_CHECKPOINT_INTERVAL_SECONDS:
        raise ValueError('Checkpoint interval must be 1..43200 seconds')
    milestones = {max(1, total_steps // 4), max(1, total_steps // 2), total_steps}
    if resume_probe:
        # A bounded stochastic qualification needs an uninterrupted full-state
        # counterpart immediately after its early checkpoint, not just tapes.
        milestones.add(min(total_steps, max(1, total_steps // 4) + 1))
    return (step in milestones
            or elapsed_seconds >= interval_seconds)

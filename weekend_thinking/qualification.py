"""Budget arithmetic from MEASURED complete optimizer-step times.

No throughput claims are baked in. Measurements must include synchronous
rollouts, reader scoring, backward, reduction and optimizer work.
"""
from __future__ import annotations
import math
import statistics


def project(step_seconds: list[float], *, training_hours: float,
            unique_prefixes_per_step: int, prefix_tokens: int,
            max_thought_tokens: int, group_size: int,
            logical_cores: int, thought_horizon: int = 1) -> dict:
    if len(step_seconds) < 10 or any(not math.isfinite(x) or x <= 0 for x in step_seconds):
        raise ValueError("Need >=10 finite positive steady-state full-step timings.")
    if training_hours <= 0 or min(unique_prefixes_per_step, prefix_tokens,
                                   max_thought_tokens, group_size, logical_cores, thought_horizon) < 1:
        raise ValueError("Invalid plan dimensions.")
    med = statistics.median(step_seconds)
    # Engineering reserve for checkpointing, variability and evaluation hooks.
    usable_seconds = training_hours * 3600 * 0.85
    steps = int(usable_seconds // med)
    prefixes = steps * unique_prefixes_per_step
    return {"median_full_step_seconds": med, "engineering_reserve_fraction": 0.15,
            "projected_steps": steps, "prefix_visits": prefixes,
            "source_token_visits": prefixes * (prefix_tokens + thought_horizon),
            "generated_token_upper_bound": prefixes * group_size * max_thought_tokens,
            "logical_core_hours": logical_cores * training_hours,
            "warning": "Token visits are not necessarily unique corpus tokens. This is a planning estimate, not a completion guarantee."}

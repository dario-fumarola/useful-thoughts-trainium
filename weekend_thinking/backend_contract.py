"""Typed interface for the not-yet-qualified production adapter."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Protocol
import torch
from .core import Format


@dataclass
class Rollouts:
    actions: torch.Tensor          # [B,G,L], including sampled first terminator
    action_mask: torch.Tensor      # bool [B,G,L], no post-stop padding
    behavior_logps: torch.Tensor   # [B,G,L], transformed sampling distribution
    policy_version: int            # optimizer step before sampling
    decode_iterations: int | None = None  # Actual padded decode work, not semantic tokens


class CausalBackend(Protocol):
    def sample(self, prefix_ids: torch.Tensor, *, fmt: Format, group_size: int,
               max_actions: int, temperature: float, policy_version: int) -> Rollouts:
        """Prefix only. No gold continuation or task answer in this interface."""
        ...

    def score(self, prefix_ids: torch.Tensor, future_ids: torch.Tensor,
              rollout: Rollouts, *, fmt: Format, horizon: int) -> torch.Tensor:
        """Stopped-gradient log evidence [B,G]; true future is restricted here."""
        ...

    def replay(self, prefix_ids: torch.Tensor, rollout: Rollouts, *,
               fmt: Format, temperature: float) -> torch.Tensor:
        """Differentiable transformed action logps [B,G,L], same policy version."""
        ...

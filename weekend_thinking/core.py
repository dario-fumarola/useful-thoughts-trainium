"""Auditable policy-gradient and causal-layout primitives.

The thought policy only receives prefix tokens. The actual continuation is
used later for teacher-forced reward scoring, never for selecting a deployed
thought. No best-of-N inference, hidden answer selection, or reader gradients.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence
import numpy as np
import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class Record:
    id: str
    document_id: str
    split: str
    prefix: tuple[int, ...]
    future: tuple[int, ...]
    domain: str = "unknown"

    def __post_init__(self):
        if not self.id or not self.document_id or not self.prefix or not self.future:
            raise ValueError("Records require IDs, a nonempty prefix, and a nonempty future.")
        if self.split not in {"train", "dev", "test"}:
            raise ValueError("split must be train, dev, or test")
        if any(type(t) is not int or t < 0 for t in self.prefix + self.future):
            raise ValueError("Token IDs must be nonnegative Python integers.")


@dataclass(frozen=True)
class Format:
    open_ids: tuple[int, ...]
    close_ids: tuple[int, ...]
    stop_id: int
    pad_id: int
    header_ids: tuple[int, ...] = ()
    additional_stop_ids: tuple[int, ...] = ()

    def __post_init__(self):
        if not self.open_ids or not self.close_ids:
            raise ValueError("Use explicit, nonempty writer/reader delimiters.")
        if any(type(t) is not int or t < 0 for t in self.header_ids + self.open_ids + self.close_ids + self.additional_stop_ids + (self.stop_id, self.pad_id)):
            raise ValueError("Format token IDs must be nonnegative integers.")


    @classmethod
    def from_dict(cls, values: dict) -> "Format":
        """Round-trip JSON lists into the immutable token-tuple contract."""
        data = dict(values)
        for key in ("open_ids", "close_ids", "header_ids", "additional_stop_ids"):
            data[key] = tuple(data.get(key, ()))
        return cls(**data)


def writer_inputs(prefixes: Sequence[Sequence[int]], fmt: Format) -> list[list[int]]:
    """This API intentionally accepts no Record/future/answer argument."""
    return [list(fmt.header_ids) + list(prefix) + list(fmt.open_ids) for prefix in prefixes]


def content_from_actions(actions: Sequence[int], stop_id: int,
                         additional_stop_ids: Sequence[int] = ()) -> list[int]:
    out = []
    stops = {stop_id, *additional_stop_ids}
    for token in actions:
        if int(token) in stops:
            break
        out.append(int(token))
    return out


def action_mask(actions: torch.Tensor, stop_id: int,
                additional_stop_ids: Sequence[int] = ()) -> torch.Tensor:
    """Include the first sampled stop action; exclude all subsequent padding.

    At a hard length cap there is NO invented stop action in the policy loss.
    """
    is_stop = actions.eq(stop_id)
    for value in additional_stop_ids:
        is_stop = is_stop | actions.eq(value)
    stop = is_stop.to(torch.int64)
    previous_stops = stop.cumsum(dim=-1) - stop
    return previous_stops.eq(0)


def policy_log_probs(logits: torch.Tensor, temperature: float,
                     banned_ids: Sequence[int] = ()) -> torch.Tensor:
    """The SAME transformed distribution must be used in sampling and loss.

    No top-k/top-p/frequency penalties or unrecorded logit floors. A fixed
    vocabulary mask is supported and applied on both paths.
    """
    if temperature <= 0:
        raise ValueError("Training temperature must be strictly positive.")
    scores = logits.float() / temperature
    if banned_ids:
        if any(type(i) is not int or not 0 <= i < scores.shape[-1] for i in banned_ids):
            raise ValueError("Banned token IDs must lie inside the vocabulary.")
        ids = torch.as_tensor(tuple(banned_ids), device=scores.device, dtype=torch.long)
        if len(set(banned_ids)) >= scores.shape[-1]:
            raise ValueError("Cannot ban the entire vocabulary.")
        scores = scores.index_fill(-1, ids, -1e9)
    return F.log_softmax(scores, dim=-1)


def sample_actions(logits: torch.Tensor, temperature: float,
                   banned_ids: Sequence[int] = ()) -> tuple[torch.Tensor, torch.Tensor]:
    """Device-side Gumbel-max draw, returning behavior log probability."""
    lp = policy_log_probs(logits, temperature, banned_ids)
    u = torch.rand_like(lp).clamp_min(torch.finfo(lp.dtype).tiny)
    token = (lp - torch.log(-torch.log(u))).argmax(dim=-1)
    return token, lp.gather(-1, token.unsqueeze(-1)).squeeze(-1)


def loo_advantages(rewards: torch.Tensor) -> torch.Tensor:
    """Leave-one-out baseline, shape [..., G]. No sample-std normalization.

    A_i = r_i - mean_{j != i}(r_j). Any common per-context no-thought
    baseline cancels exactly. Rewards/advantages do not backpropagate.
    """
    if rewards.ndim < 1 or rewards.shape[-1] < 2:
        raise ValueError("At least two independent samples per context are required.")
    r = rewards.detach()
    return r - (r.sum(dim=-1, keepdim=True) - r) / (r.shape[-1] - 1)


def policy_loss(current_lp: torch.Tensor, actions_mask: torch.Tensor,
                advantages: torch.Tensor, *, old_lp: torch.Tensor | None = None,
                clip_epsilon: float = 0.2, normalization: str = "sequence") -> torch.Tensor:
    """Signed on-policy REINFORCE, or optional tokenwise PPO surrogate.

    Shapes: current_lp/mask [B,G,L], advantages [B,G]. The default averages
    each completion over its sampled actions, as in the RLP equation. The
    optional fixed_cap normalization is a separate, explicitly named ablation.
    For one fresh update per rollout, REINFORCE needs no importance ratio.
    """
    if current_lp.ndim != 3 or actions_mask.shape != current_lp.shape:
        raise ValueError("Expected matching [batch, group, length] logps and masks.")
    if advantages.shape != current_lp.shape[:-1]:
        raise ValueError("Advantage shape must be [batch, group].")
    a = advantages.detach().unsqueeze(-1)
    if old_lp is None:
        objective = current_lp * a
    else:
        if old_lp.shape != current_lp.shape or not 0 < clip_epsilon < 1:
            raise ValueError("Invalid old logps or PPO clip.")
        ratio = torch.exp(current_lp - old_lp.detach())
        objective = torch.minimum(ratio * a,
                                  ratio.clamp(1-clip_epsilon, 1+clip_epsilon) * a)
    mask = actions_mask.to(objective.dtype)
    if normalization == "sequence":
        return -((objective * mask).sum(-1) / mask.sum(-1).clamp_min(1)).mean()
    if normalization == "fixed_cap":
        return -(objective * mask).sum() / current_lp.numel()
    raise ValueError("normalization must be sequence or fixed_cap")


@dataclass
class SequenceBatch:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    labels: torch.Tensor
    loss_mask: torch.Tensor


def causal_batch(sequences: Sequence[Sequence[int]], target_starts: Sequence[int],
                 *, pad_id: int, total_length: int | None = None) -> SequenceBatch:
    """Shift labels ONCE. Position j predicts token j+1.

    target_starts gives the first supervised TOKEN index (not logit index).
    The final real input token and all padding have no next-token target.
    Static padding changes neither the context nor the supervised targets.
    """
    if not sequences or len(sequences) != len(target_starts):
        raise ValueError("Provide one target start per nonempty sequence batch.")
    width = total_length or max(map(len, sequences))
    x = torch.full((len(sequences), width), pad_id, dtype=torch.long)
    attention = torch.zeros_like(x)
    labels = torch.zeros_like(x)
    mask = torch.zeros(x.shape, dtype=torch.float32)
    for i, (seq, start) in enumerate(zip(sequences, target_starts)):
        n = len(seq)
        if n < 2 or n > width or not 1 <= start < n:
            raise ValueError(f"Invalid sequence/target range: length={n}, start={start}, width={width}")
        x[i, :n] = torch.tensor(seq, dtype=torch.long)
        attention[i, :n] = 1
        labels[i, :n-1] = x[i, 1:n]
        mask[i, start-1:n-1] = 1
    return SequenceBatch(x, attention, labels, mask)


def scoring_batch(records: Sequence[Record], notes: Sequence[Sequence[int] | None],
                  fmt: Format, horizon: int, *, max_length: int | None = None) -> SequenceBatch:
    """Teacher force true future ONLY in the reader. None means raw silence.

    An empty list is an empty-note FORMAT control, which is not raw silence.
    """
    if len(records) != len(notes) or horizon < 1:
        raise ValueError("One note per record and a positive horizon required.")
    sequences, starts = [], []
    for row, note in zip(records, notes):
        if horizon > len(row.future):
            raise ValueError("Requested horizon exceeds the stored continuation.")
        seq = list(row.prefix)
        if note is not None:
            seq = list(fmt.header_ids) + seq
            seq += list(fmt.open_ids) + list(note) + list(fmt.close_ids)
        starts.append(len(seq))
        sequences.append(seq + list(row.future[:horizon]))
    return causal_batch(sequences, starts, pad_id=fmt.pad_id, total_length=max_length)


def target_log_likelihood(logits: torch.Tensor, labels: torch.Tensor,
                          loss_mask: torch.Tensor) -> torch.Tensor:
    """Per-sequence SUM log likelihood; denominator tracked separately."""
    lp = logits.float().log_softmax(-1).gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    return (lp * loss_mask).sum(-1)


def matched_controls(notes: Sequence[Sequence[int]], *, filler_ids: Sequence[int],
                     seed: int) -> dict[str, list[list[int]]]:
    """No targets are accepted. Match each control to its recipient's length.

    Other-note controls use deterministic cyclic donors and cyclic repeat or
    trim for exact length. Report this construction; it is not a perfect
    linguistic control. At least two notes required. Empty donors use filler.
    """
    if len(notes) < 2 or not filler_ids:
        raise ValueError("Need >=2 notes and a nonempty filler token sequence.")
    rng = np.random.default_rng(seed)
    def fit(values, length):
        values = list(values) or list(filler_ids)
        return (values * ((length + len(values) - 1)//len(values)))[:length]
    filler, shuffled, other = [], [], []
    for i, note in enumerate(notes):
        filler.append(fit(filler_ids, len(note)))
        shuffled.append(rng.permutation(list(note)).tolist())
        other.append(fit(notes[(i+1) % len(notes)], len(note)))
    return {"filler": filler, "shuffled": shuffled, "other": other}

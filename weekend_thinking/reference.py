"""Small eager CPU reference. This is NOT the production Neuron execution path.

The reference deliberately recomputes full prefixes, so it is useful for
checking a cached accelerator implementation, not for timing large models.
It trains one fresh policy update per independently generated group.
"""
from __future__ import annotations
from dataclasses import dataclass
import torch
from torch import nn
from .core import (Format, Record, action_mask, causal_batch, content_from_actions,
                   loo_advantages, policy_log_probs, policy_loss, sample_actions,
                   scoring_batch, target_log_likelihood, writer_inputs)


class TinyCausalLM(nn.Module):
    """Toy recurrent causal LM used only for executable plumbing tests."""
    def __init__(self, vocab_size: int = 31, hidden_size: int = 24):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.rnn = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.head = nn.Linear(hidden_size, vocab_size)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        hidden, _ = self.rnn(self.embed(input_ids))
        return self.head(hidden)


@dataclass(frozen=True)
class UpdateConfig:
    group_size: int = 4
    max_thought_tokens: int = 8
    horizon: int = 1
    temperature: float = 0.7
    banned_ids: tuple[int, ...] = ()
    grad_clip: float = 1.0

    def __post_init__(self):
        if self.group_size < 2 or self.max_thought_tokens < 1 or self.horizon < 1:
            raise ValueError("Invalid rollout configuration.")


@torch.no_grad()
def generate(model: nn.Module, prefixes: list[list[int]], fmt: Format,
             cfg: UpdateConfig) -> tuple[torch.Tensor, torch.Tensor]:
    """Accepts prefixes only. No continuation/label/evaluator is accessible.

    Output [B,G,L] contains the actually sampled first stop. Later pad actions
    are masked. Length-capped samples do NOT acquire an artificial stop action.
    """
    model.eval()
    device = next(model.parameters()).device
    all_actions, all_lp = [], []
    for prefix in writer_inputs(prefixes, fmt):
        seq = torch.tensor([prefix] * cfg.group_size, device=device)
        alive = torch.ones(cfg.group_size, dtype=torch.bool, device=device)
        actions, lps = [], []
        for _ in range(cfg.max_thought_tokens):
            logits = model(seq)[:, -1]
            token, lp = sample_actions(logits, cfg.temperature, cfg.banned_ids)
            token = torch.where(alive, token, fmt.pad_id)
            lp = torch.where(alive, lp, 0.0)
            actions.append(token); lps.append(lp)
            for stop_id in (fmt.stop_id, *fmt.additional_stop_ids):
                alive = alive & token.ne(stop_id)
            seq = torch.cat([seq, token[:, None]], dim=1)
        all_actions.append(torch.stack(actions, -1))
        all_lp.append(torch.stack(lps, -1))
    return torch.stack(all_actions), torch.stack(all_lp)


def replay_log_probs(model: nn.Module, prefixes: list[list[int]],
                     actions: torch.Tensor, fmt: Format, cfg: UpdateConfig) -> torch.Tensor:
    """Teacher force the sampled actions with exactly the sampling transform."""
    device = next(model.parameters()).device
    results = []
    for b, prefix in enumerate(writer_inputs(prefixes, fmt)):
        prompt = torch.tensor([prefix] * cfg.group_size, device=device)
        seq = torch.cat([prompt, actions[b]], dim=1)
        logits = model(seq)[:, len(prefix)-1:len(prefix)-1+cfg.max_thought_tokens]
        lp = policy_log_probs(logits, cfg.temperature, cfg.banned_ids)
        results.append(lp.gather(-1, actions[b].unsqueeze(-1)).squeeze(-1))
    return torch.stack(results)


@torch.no_grad()
def score(model: nn.Module, records: list[Record], notes: list[list[int] | None],
          fmt: Format, horizon: int) -> torch.Tensor:
    model.eval()
    batch = scoring_batch(records, notes, fmt, horizon)
    device = next(model.parameters()).device
    logits = model(batch.input_ids.to(device))
    return target_log_likelihood(logits, batch.labels.to(device), batch.loss_mask.to(device))


def thought_update(model: nn.Module, optimizer: torch.optim.Optimizer,
                   records: list[Record], fmt: Format, cfg: UpdateConfig) -> dict:
    """One stopped-reward, signed mean-objective update; no NTP auxiliary loss.

    Keep deterministic model execution for exact behavior/replay agreement.
    .eval() does NOT disable gradients. Dropout is deliberately disabled.
    Accelerator adapters must retain this property or account for randomness.
    """
    prefixes = [list(r.prefix) for r in records]
    optimizer.zero_grad(set_to_none=True)
    actions, behavior_lp = generate(model, prefixes, fmt, cfg)
    notes = [content_from_actions(actions[b, g].tolist(), fmt.stop_id, fmt.additional_stop_ids)
             for b in range(len(records)) for g in range(cfg.group_size)]
    repeated = [r for r in records for _ in range(cfg.group_size)]
    evidence = score(model, repeated, notes, fmt, cfg.horizon).reshape(len(records), cfg.group_size)
    clean = score(model, records, [None] * len(records), fmt, cfg.horizon)
    gains = evidence - clean[:, None]
    advantages = loo_advantages(gains)
    # Common no-thought baselines cancel in the group-relative update. Still
    # report absolute same-model clean and thought evidence at every evaluation.
    replay_lp = replay_log_probs(model, prefixes, actions, fmt, cfg)
    mask = action_mask(actions, fmt.stop_id, fmt.additional_stop_ids)
    max_error = (replay_lp.detach() - behavior_lp).abs()[mask].max().item()
    if max_error > 1e-5:
        raise RuntimeError(f"Behavior/replay mismatch in CPU reference: {max_error}")
    loss = policy_loss(replay_lp, mask, advantages)
    if not torch.isfinite(loss):
        raise RuntimeError("Nonfinite policy loss.")
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip,
                                         error_if_nonfinite=True)
    optimizer.step()
    is_stop = actions.eq(fmt.stop_id)
    for stop_id in fmt.additional_stop_ids:
        is_stop = is_stop | actions.eq(stop_id)
    return {
        "loss": float(loss.detach()), "gradient_norm_before_clip": float(norm),
        "mean_local_gain_nats": float(gains.mean()),
        "mean_clean_log_evidence": float(clean.mean()),
        "mean_thought_log_evidence": float(evidence.mean()),
        "mean_sampled_actions": float(mask.float().sum(-1).mean()),
        "fraction_length_capped": float((~is_stop).all(-1).float().mean()),
        "max_behavior_replay_error": max_error,
        "prefixes": len(records), "rollouts": len(notes),
        "status": "CPU toy/reference measurement, not a Qwen or Trainium result",
    }


def vanilla_update(model: nn.Module, optimizer: torch.optim.Optimizer,
                   records: list[Record], fmt: Format, cfg: UpdateConfig) -> dict:
    """Same raw prefix + TRAINED future horizon; extra stored future is unused."""
    model.eval()
    seqs = [list(r.prefix + r.future[:cfg.horizon]) for r in records]
    batch = causal_batch(seqs, [1] * len(seqs), pad_id=fmt.pad_id)
    device = next(model.parameters()).device
    optimizer.zero_grad(set_to_none=True)
    logits = model(batch.input_ids.to(device))
    ll = target_log_likelihood(logits, batch.labels.to(device), batch.loss_mask.to(device))
    loss = -ll.sum() / batch.loss_mask.sum().to(device)
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip,
                                         error_if_nonfinite=True)
    optimizer.step()
    return {"ntp_loss": float(loss.detach()), "gradient_norm_before_clip": float(norm),
            "supervised_real_tokens": int(batch.loss_mask.sum()),
            "status": "CPU toy/reference measurement, not a Qwen or Trainium result"}

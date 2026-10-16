"""Ordinary causal cached Qwen backend, with injected execution boundaries.

No scientific launch is authorized by importing this module. CPU execution is
for parity tests; production qualification must run the same code on Neuron.
"""
from __future__ import annotations

from contextlib import nullcontext
from typing import Callable

import torch
from transformers.cache_utils import Cache, CacheLayerMixin

from weekend_thinking.backend_contract import Rollouts
from weekend_thinking.core import Format, action_mask, policy_log_probs, sample_actions


class _CausalCacheLayer(CacheLayerMixin):
    is_sliding = False
    is_compileable = True

    def __init__(self, holder):
        self.holder = holder
        self.keys = self.values = None
        self.is_initialized = False

    def lazy_initialization(self, key_states, value_states):
        b, h, _, d = key_states.shape
        self.keys = torch.zeros((b, h, self.holder['width'], d),
                                dtype=key_states.dtype, device=key_states.device)
        self.values = torch.zeros_like(self.keys)
        self.is_initialized = True

    def update(self, key_states, value_states, *args, **kwargs):
        if not self.is_initialized:
            self.lazy_initialization(key_states, value_states)
        self.keys.index_copy_(2, self.holder['positions'], key_states)
        self.values.index_copy_(2, self.holder['positions'], value_states)
        return self.keys, self.values

    def get_seq_length(self):
        # Callers always provide actual position_ids and a complete mask.
        return self.holder['width'] if self.is_initialized else 0

    def get_max_length(self):
        return self.holder['width']

    def get_mask_sizes(self, query_length):
        return self.holder['width'], 0

    def reset(self):
        if self.is_initialized:
            self.keys.zero_()
            self.values.zero_()


class CausalCache(Cache):
    """Fixed-size cache: tensor-valued positions prevent per-position graphs."""
    def __init__(self, layers: int, width: int):
        self.holder = {'width': width, 'positions': None}
        super().__init__(layers=[_CausalCacheLayer(self.holder) for _ in range(layers)])

    def set_positions(self, positions):
        self.holder['positions'] = positions


def causal_mask(positions, width):
    """Explicit additive causal mask; no host extraction of position values."""
    keys = torch.arange(width, device=positions.device)
    return torch.where(keys[None, :] <= positions[:, None], 0.0, -1e9)[None, None]


def pack_reader(prefix_ids, actions, fmt: Format, future_ids):
    """Compact content+close+future into a fixed shape, entirely on device.

    Input actions [B,G,L] include the sampled terminator. The terminator is
    stripped from content; the close delimiter follows immediately. Returns
    input ids, real lengths, and the positions predicting each future token.
    """
    b, g, cap = actions.shape
    device = actions.device
    prompt = torch.cat((torch.tensor(fmt.header_ids, device=device, dtype=torch.long).expand(b, -1),
                        prefix_ids,
                        torch.tensor(fmt.open_ids, device=device, dtype=torch.long).expand(b, -1)), -1)
    prompt = prompt.repeat_interleave(g, dim=0)
    flat = actions.reshape(b * g, cap)
    mask = action_mask(flat, fmt.stop_id, fmt.additional_stop_ids)
    stop = flat.eq(fmt.stop_id)
    for value in fmt.additional_stop_ids:
        stop = stop | flat.eq(value)
    content_lengths = (mask & ~stop).sum(-1)
    p, c, k = prompt.shape[-1], len(fmt.close_ids), future_ids.shape[-1]
    width = p + cap + c + k
    j = torch.arange(width, device=device)[None, :].expand(b * g, -1)
    content_end = p + content_lengths[:, None]
    # Clamp gather indices for inactive branches: all torch.where arms execute.
    prompt_values = prompt.gather(-1, j.clamp(0, p - 1))
    content_values = flat.gather(-1, (j - p).clamp(0, cap - 1))
    close = torch.tensor(fmt.close_ids, device=device, dtype=torch.long).expand(b * g, -1)
    close_values = close.gather(-1, (j - content_end).clamp(0, c - 1))
    futures = future_ids.repeat_interleave(g, dim=0)
    future_values = futures.gather(-1, (j - content_end - c).clamp(0, k - 1))
    x = torch.full_like(j, fmt.pad_id)
    x = torch.where(j < p, prompt_values, x)
    x = torch.where((j >= p) & (j < content_end), content_values, x)
    x = torch.where((j >= content_end) & (j < content_end + c), close_values, x)
    x = torch.where((j >= content_end + c) & (j < content_end + c + k), future_values, x)
    pred_positions = content_end + c - 1 + torch.arange(k, device=device)[None, :]
    return x, (content_end[:, 0] + c + k), pred_positions


class NeuronCausalBackend:
    def __init__(self, model, *, amp: Callable = nullcontext,
                 sync: Callable = lambda: None, head_chunk: int = 32,
                 checkpoint_trunk: bool = True, banned_ids: tuple[int, ...] = (),
                 stop_check_interval: int = 32, head_amp: Callable | None = None):
        self.model = model
        self.amp = amp
        self.head_amp = amp if head_amp is None else head_amp
        self.sync = sync
        self.head_chunk = head_chunk
        self.checkpoint_trunk = checkpoint_trunk
        self.banned_ids = banned_ids
        if stop_check_interval < 1:
            raise ValueError('Positive stop-check interval required')
        self.stop_check_interval = stop_check_interval
        if model.config.model_type != 'qwen3' or any(t != 'full_attention' for t in model.config.layer_types):
            raise ValueError('This adapter supports ordinary causal Qwen3 only')
        self.model.eval()  # Deterministic dropout, without disabling gradients.

    @property
    def device(self):
        return next(self.model.parameters()).device

    def _prompt(self, prefix, fmt):
        b = prefix.shape[0]
        return torch.cat((torch.tensor(fmt.header_ids, device=self.device, dtype=torch.long).expand(b, -1),
                          prefix.to(self.device),
                          torch.tensor(fmt.open_ids, device=self.device, dtype=torch.long).expand(b, -1)), -1)

    def hidden(self, x):
        positions = torch.arange(x.shape[-1], device=x.device)
        mask = causal_mask(positions, x.shape[-1])
        if self.checkpoint_trunk and torch.is_grad_enabled():
            # HF's checkpointing requires train(); explicitly checkpoint here
            # so deterministic eval() and memory-bounded replay coexist.
            from torch.utils.checkpoint import checkpoint
            trunk = self.model.model
            with self.amp():
                h = trunk.embed_tokens(x)
                pe = trunk.rotary_emb(h, positions[None])
            for layer in trunk.layers:
                def block(h, cos, sin, layer=layer):
                    with self.amp():
                        return layer(h, attention_mask=mask, position_ids=positions[None],
                                     position_embeddings=(cos, sin), use_cache=False)
                h = checkpoint(block, h, pe[0], pe[1], use_reentrant=True,
                               preserve_rng_state=False)
            with self.amp():
                return trunk.norm(h)
        with self.amp():
            return self.model.model(input_ids=x, position_ids=positions[None],
                attention_mask={'full_attention': mask},
                use_cache=False).last_hidden_state

    def _next(self, tokens, positions, cache):
        cache.set_positions(positions)
        with self.amp():
            h = self.model.model(input_ids=tokens, position_ids=positions[None],
                attention_mask={'full_attention': causal_mask(positions, cache.holder['width'])},
                past_key_values=cache, use_cache=True).last_hidden_state
        with self.head_amp():
            return self.model.lm_head(h[:, -1]).float()

    @torch.no_grad()
    def sample(self, prefix_ids, *, fmt: Format, group_size: int, max_actions: int,
               temperature: float, policy_version: int, greedy: bool = False):
        if group_size < 1 or max_actions < 1 or temperature <= 0:
            raise ValueError('Positive group size, cap and temperature required')
        prompt = self._prompt(prefix_ids, fmt).repeat_interleave(group_size, 0)
        b = prefix_ids.shape[0]
        p = prompt.shape[-1]
        cache = CausalCache(self.model.config.num_hidden_layers, p + max_actions)
        logits = self._next(prompt, torch.arange(p, device=self.device), cache)
        alive = torch.ones(b * group_size, dtype=torch.bool, device=self.device)
        pos = torch.tensor([p], device=self.device, dtype=torch.long)
        # Static output buffers avoid variable assembly graphs when all samples
        # stop before the cap. Only an alive scalar crosses to host per chunk.
        actions = torch.full((b * group_size, max_actions), fmt.pad_id,
                             device=self.device, dtype=torch.long)
        behavior = torch.zeros((b * group_size, max_actions), device=self.device)
        for i in range(max_actions):
            if greedy:
                lp = policy_log_probs(logits, temperature, self.banned_ids)
                token = lp.argmax(-1)
                selected = lp.gather(-1, token[:, None])[:, 0]
            else:
                token, selected = sample_actions(logits, temperature, self.banned_ids)
            token = torch.where(alive, token, fmt.pad_id)
            selected = torch.where(alive, selected, 0.0)
            slot = pos - p
            actions.index_copy_(1, slot, token[:, None])
            behavior.index_copy_(1, slot, selected[:, None])
            for stop in (fmt.stop_id, *fmt.additional_stop_ids):
                alive = alive & token.ne(stop)
            self.sync()
            if (i + 1) % self.stop_check_interval == 0 and not bool(alive.any().cpu()):
                break
            if i + 1 < max_actions:
                logits = self._next(token[:, None], pos, cache)
                pos = pos + 1
        actions = actions.reshape(b, group_size, max_actions)
        behavior = behavior.reshape_as(actions)
        return Rollouts(actions, action_mask(actions, fmt.stop_id, fmt.additional_stop_ids),
                        behavior, policy_version, i + 1)

    @torch.no_grad()
    def score(self, prefix_ids, future_ids, rollout, *, fmt: Format, horizon: int):
        if not 1 <= horizon <= future_ids.shape[-1]:
            raise ValueError('Requested horizon unavailable')
        x, _, positions = pack_reader(prefix_ids.to(self.device), rollout.actions,
                                      fmt, future_ids[:, :horizon].to(self.device))
        hidden = self.hidden(x)
        h = hidden.gather(1, positions[:, :, None].expand(-1, -1, hidden.shape[-1]))
        with self.head_amp():
            logits = self.model.lm_head(h).float()
        targets = future_ids[:, :horizon].to(self.device).repeat_interleave(rollout.actions.shape[1], 0)
        ll = logits.log_softmax(-1).gather(-1, targets[:, :, None])[:, :, 0].sum(-1)
        return ll.reshape(rollout.actions.shape[:2]).detach()

    @torch.no_grad()
    def clean_score(self, prefix_ids, future_ids, horizon=1):
        seq = torch.cat((prefix_ids.to(self.device), future_ids[:, :horizon].to(self.device)), -1)
        h = self.hidden(seq)[:, prefix_ids.shape[-1] - 1:-1]
        with self.head_amp():
            logits = self.model.lm_head(h).float()
        return logits.log_softmax(-1).gather(-1, future_ids[:, :horizon, None].to(self.device))[:, :, 0].sum(-1)

    def replay(self, prefix_ids, rollout, *, fmt: Format, temperature: float):
        b, g, cap = rollout.actions.shape
        prompt = self._prompt(prefix_ids, fmt).repeat_interleave(g, 0)
        x = torch.cat((prompt, rollout.actions.reshape(b*g, cap)[:, :-1]), -1)
        hidden = self.hidden(x)[:, prompt.shape[-1] - 1:]
        parts = []
        # Use replay_one_loss for bounded-memory backward. This method is for
        # short trace qualification where all differentiable action logps help.
        for start in range(0, cap, self.head_chunk):
            with self.head_amp():
                logits = self.model.lm_head(hidden[:, start:start+self.head_chunk])
            lp = policy_log_probs(logits, temperature, self.banned_ids)
            actions = rollout.actions.reshape(b*g, cap)[:, start:start+self.head_chunk]
            parts.append(lp.gather(-1, actions[:, :, None])[:, :, 0])
        return torch.cat(parts, -1).reshape(b, g, cap)

    def replay_one_loss(self, prefix_ids, actions, mask, advantage, *, fmt,
                        temperature, divisor):
        """One completion's normalized signed contribution; no reader gradient."""
        from torch.utils.checkpoint import checkpoint
        prompt = self._prompt(prefix_ids, fmt)
        x = torch.cat((prompt, actions[:, :-1]), -1)
        hidden = self.hidden(x)[:, prompt.shape[-1] - 1:]
        def part(h, y, m):
            with self.head_amp():
                logits = self.model.lm_head(h)
            lp = policy_log_probs(logits, temperature, self.banned_ids)
            return (lp.gather(-1, y[:, :, None])[:, :, 0] * m).sum()
        total = torch.zeros((), device=self.device)
        for start in range(0, actions.shape[-1], self.head_chunk):
            sl = slice(start, start + self.head_chunk)
            total = total + checkpoint(part, hidden[:, sl], actions[:, sl], mask[:, sl],
                                       use_reentrant=True, preserve_rng_state=False)
        # This installed Neuron stack returns -1 from S64 clamp_min(1).
        # FP32 represents every possible action count (<=1024) exactly and
        # preserves signed sequence normalization without that integer clamp.
        length = mask.float().sum().clamp_min(1.0)
        return -advantage.detach() * total / length / divisor

    def vanilla_loss(self, prefix_ids, future_ids):
        """Train the raw prefix and exactly one real future token."""
        from torch.utils.checkpoint import checkpoint
        x = torch.cat((prefix_ids.to(self.device), future_ids[:, :1].to(self.device)), -1)
        hidden = self.hidden(x[:, :-1])
        labels = x[:, 1:]
        def part(h, y):
            with self.head_amp():
                logits = self.model.lm_head(h).float()
            return torch.nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                                                      y.reshape(-1), reduction='sum')
        total = torch.zeros((), device=self.device)
        for start in range(0, labels.shape[-1], self.head_chunk):
            sl = slice(start, start+self.head_chunk)
            total = total + checkpoint(part, hidden[:, sl], labels[:, sl],
                                       use_reentrant=True, preserve_rng_state=False)
        return total / labels.numel()

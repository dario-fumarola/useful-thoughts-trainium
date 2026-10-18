"""Plain continuation control for a future, separately identified SFT recipe.

The running W/C/T hardware qualification does not import this module. A raw
continuation loss gives the plain control the same anchor emphasis as W's
note-conditioned reader loss. Its inputs contain no note or re-anchoring text.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from train.masked_causal_ce import masked_causal_ce
from train.useful_note_sft_data import packed


def pack_raw_continuation(prefix_ids, continuation_ids, *, width, target_slots=16):
    """Ordinary causal CE on exactly the original next16 tokens.

    Future tokens enter as teacher-forced CE labels and causal inputs only.
    This API accepts no teacher note, boundary proof, or alternative future.
    Use the reader's fixed shape to compare equal label horizons; padding is
    masked by explicit target positions rather than by the EOS token value.
    """
    past = list(prefix_ids)
    future = list(continuation_ids)
    if (not 128 <= len(past) <= 512 or len(future) != 16 or target_slots != 16
            or any(type(t) is not int or not 0 <= t <= 151668 for t in past + future)):
        raise ValueError('Require the native prefix and exactly16 original continuation tokens')
    ids = past + future
    return packed(ids, range(len(past), len(ids)), note_positions=(),
                  width=width, target_slots=target_slots)


@dataclass
class PlainControlCE:
    loss: torch.Tensor
    continuation_loss: torch.Tensor
    source_loss: torch.Tensor


def plain_control_ce(backend, *, input_ids, target_positions, labels,
                     note_close_mask, future_mask, source_cut):
    """Compute raw-next16 mean +0.5 full-source mean in ONE causal pass.

    The existing two-role kernel is used with disjoint continuation/complement
    masks. Its internal ``note_close`` role here means the16 anchor labels;
    there are no inserted notes. Both reported means retain their actual names.
    This is exactly the two-pass objective, not a token-count approximation.
    """
    if (type(source_cut) is not int or not 128 <= source_cut <= 512
            or input_ids.device.type != 'cpu' or input_ids.ndim != 2 or input_ids.shape[0] != 1
            or target_positions.device.type != 'cpu' or labels.device.type != 'cpu'
            or note_close_mask.device.type != 'cpu' or future_mask.device.type != 'cpu'
            or target_positions.shape != labels.shape or target_positions.shape != future_mask.shape
            or note_close_mask.shape != future_mask.shape or note_close_mask.any()
            or future_mask.dtype != torch.bool or note_close_mask.dtype != torch.bool
            or int(future_mask.sum()) != 527
            or not torch.equal(target_positions[future_mask], torch.arange(1, 528))):
        raise ValueError('Plain control requires one complete original528-token source and causal cut')
    if not torch.equal(labels[future_mask], input_ids.gather(1, target_positions)[future_mask]):
        raise ValueError('Plain control labels must be the original source tokens')
    anchor = future_mask & (target_positions >= source_cut) & (target_positions < source_cut + 16)
    complement = future_mask & ~anchor
    ce = masked_causal_ce(backend, input_ids, target_positions, labels, anchor, complement,
                         note_weight=1. + .5 * 16 / 527,
                         future_weight=.5 * 511 / 527)
    source = (16 * ce.note_close_loss + 511 * ce.future_loss) / 527
    return PlainControlCE(ce.loss, ce.note_close_loss, source)

"""Shared source-prefix NTP plus separately normalized direct or CoT targets.

The existing causal CE engine averages each component within each example.
Source and task each receive weight 0.5, independent of their token counts.
Both arms preserve the identical native source prefix. The calculation opening
is supplied, never predicted; computation labels depend only on the prefix.
"""
from train import real_target_packing
from train.useful_note_sft_data import packed

ARMS = ('source_direct', 'source_cot')


def training_pack(row, tokenizer, arm, *, width=640, target_slots=640):
    if arm not in ARMS:
        raise ValueError('Declared source-balanced arm required')
    target_arm = 'target_direct' if arm == 'source_direct' else 'target_cot'
    task = real_target_packing.training_pack(
        row, tokenizer, target_arm, width=width, target_slots=target_slots)
    task_positions = task['target_positions'][task['future_mask']].tolist()
    source_positions = list(range(1, len(row['prefix_ids'])))
    assert source_positions and task_positions
    assert set(source_positions).isdisjoint(task_positions)
    ids = task['input_ids'][0, :task_positions[-1]+1].tolist()
    # These are generic component masks. In this new recipe note_close_mask
    # denotes source-prefix NTP; future_mask denotes the complete task target.
    # Reuse the verified packing/CE primitives without changing old schemas.
    return packed(ids, source_positions + task_positions,
                  note_positions=source_positions, width=width,
                  target_slots=target_slots)

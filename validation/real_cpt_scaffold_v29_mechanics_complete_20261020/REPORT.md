# V29 training mechanics

The actual two-forward objective completes eight TRAIN-only updates and a fresh-process full-state restore at step4. The independent audit checks16 native initialization vectors, all4,960 FP32 master shards and their optimizer moments across full states4/8, exact saved active-model bytes, and eight actual optimizer restorations. The CPU first-step component and total-loss bounds pass.

Source and auxiliary gradients accumulate before one ZeRO update. All three view roles execute, and both full checkpoints are retained. The terminal8 checkpoint and run envelope have a complete SHA readback and directory fsync on same-instance EBS; RAM originals remain. This is local protection, not an off-instance backup.

The original supervisor completed fresh4 but refused to start restoration near its operational deadline. An explicit continuation, released before execution, completed the original eight-update scope using the unchanged protocol and remaining budget. Total reservation is1.506 chip-hours against2.0. Cold compilation dominates the first two updates (1,511s and888s); subsequent updates are6–12s, with58s for the first restored update. These are individual bounded observations.

This verifies mechanics and recovery. It does not establish useful learning, uninterrupted-trajectory equality or sustained memory stability. No held-out model outcomes were inspected, and the current numerical driver still rejects a long scientific run.

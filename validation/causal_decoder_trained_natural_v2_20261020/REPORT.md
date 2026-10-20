# Trained-model natural-stop batching

The unchanged batched greedy decoder matches the original serial decoder on all actions, content IDs, text and stop types for eight fixed TRAIN-only inputs under a frozen V17 trained model. All eight notes stop naturally at the native answer boundary, at22–77 actions; the production cap is128. Independent NumPy FP64 normalization reproduces all24 saved full-vocabulary native checks.

Three rotating timed repeats preserve the same complete outputs. Warm medians for the eight inputs are29.255s serial,10.003s at batch4 and6.000s at batch8:2.92× and4.88× descriptive speedups. These are resident-model microbenchmarks under concurrent separate compilation, not end-to-end serving measurements. Reservation is0.177 whole-chip-hours; the process session and device close cleanly.

The numerical vectors remain on EBS with full hashes in the machine report. All eight generated notes and token arrays are published here. This is an engineering result on equal-length greedy writer inputs, not a capability score, universal equivalence guarantee, sampling/ragged-batch result or qualification of the scalar-answer reader. It was not introduced into the closed V28 study.

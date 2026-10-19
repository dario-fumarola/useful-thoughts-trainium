# V15 evaluation implementation

The prospective scientific protocol is on main in
`validation/real_target_sft_v15_design_20261019.json`. This branch implements
its new `DEV_fresh384` cohort without changing the legacy 256-item identities.
Old source branches and their original outputs remain intact.

Prefix-only note generation uses four shards, width544, cap128, native EOS or
the first native `Answer:` token. Reader contexts retain every original note,
including capped, empty and failed attempts, followed by the same full boundary.
Unrelated controls use whole donor notes under the existing deterministic
derangement. Gold never enters a generation loader.

Scalar continuation uses four shards, width672, cap32 and native EOS only.
The unchanged initial-scalar-v2 parser runs after physical process closure.
All384 documents stay in each denominator, with explicit unsupported, format,
incomplete and execution-error counts. A CPU fixture exercises all1920 native
attempts and verifies that a correct-looking failed-note path cannot count.

The canonical likelihood path remains CPU FP32. The optional new-study path
uses one explicitly selected logical Trainium core and the SAME no-gradient
span primitive. For EACH model/panel, first4 fixed documents in every mode
receive CPU FP32 references at width672. The native pass must reproduce every
token within0.001 and absolute mean span drift within0.001 before scoring the
whole panel. Both reference arrays, ordering, timings and errors are saved.
Failing parity preserves the attempt and requires a separately identified CPU
run; it does not justify changing the checkpoint, note, cohort or tolerance.

These fixed-reference checks do not establish all-panel CPU parity for each
trained model. The earlier B0v14 all1280-row parity/throughput profile is separate.
Actual native/cached generation checks still run for every model; CPU tests
alone qualify neither decoding nor native scoring. No RL-step throughput,
equal-compute inference or controlled serving-latency claim follows.

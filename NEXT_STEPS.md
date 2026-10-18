# Current queue

1. Run one batched frozen-B0 interface question on>=256 DEV documents, all six
   interfaces and raw/empty/correct/corrupt/shuffled/other conditions. Full numeral
   plus closing boundary or general natural16 is primary; K1 secondary. Report
   tax, oracle gain and sensitivity with paired document intervals. Thresholds:
   >=−0.5, >=+2, >=+2 nats respectively. Target-copy oracle is plumbing only.
2. If no interface passes, test scalar logit mixing. Zero weight establishes
   zero tax, not utility. If still none passes, select the best transparently and
   learn the reader with SFT, then repeat the complete oracle.
3. Prepare>=5000 real TRAIN examples and>=256 held DEV. Relax arithmetic to
   computable equations; general hindsight notes use high-surprisal anchors and
   frozen-reader positive full-span gain filtering. Save all acceptance/rejection
   rates. Future-aware teacher labels remain separate from prefix-only policy inputs.
4. Qualify BF16 compute/FP32 master updates, then W versus a credible ordinary-CE
   control, and a thinking-augmented CE arm. Use two epochs, global32, LR1e−5,
   warmup/cosine. Lock final objectives/exposure before launch. Fresh notes, raw,
   matched empty and unrelated controls; GSM8K TRAIN>=128, preferably512. Evaluate
   downstream regardless of local sign.512-example draft is superseded, never launched.
5. RL only after useful fresh generated notes, with fullspan rewards, G16/cap256,
   W reference guardrail and stored behavior probabilities. Profile the fixed
   overhead first; no256-step study until useful policy evidence and<=90s steps.

Keep STATUS entries<=15 lines/experiment; hashes and archives live in JSON.
Never read TEST. Report control performance beside every thought-mode claim.
The owner allows adapting this queue with reasoned, explicit new experiment IDs.

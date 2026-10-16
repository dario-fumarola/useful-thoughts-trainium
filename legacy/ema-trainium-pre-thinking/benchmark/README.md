# benchmark/ — does thinking-in-pretraining produce a better model?

Zero-shot likelihood evaluation. No fine-tuning, no task heads: the same
frozen weights answer every task, scored by ranking the log-likelihood of
each candidate continuation. This is the standard base-LM protocol (GPT-3,
Pythia, OLMo, LLaMA all report this family). Fine-tuning-based suites
(GLUE/SuperGLUE) were abandoned for autoregressive LMs because the number
you get depends more on the fine-tuning recipe than on the pretrained model,
and it breaks the same-weights comparison that ablations need.

## Tasks (`fetch_data.py`, 500 examples each, deterministic subsample)

| task | form | chance |
|---|---|---|
| lambada | final-word prediction | — |
| hellaswag | 4-way ending | 25% |
| winogrande | 2-way pronoun | 50% |
| arc_easy / arc_challenge | 4-way science QA | 25% |
| openbookqa | 4-way | 25% |
| sciq | 4-way science QA (with support passage) | 25% |
| boolq | yes/no over a passage | 50% |

piqa is skipped: its HF copy still requires a deprecated loading script.

**Expect most of these to be near chance.** 0.6B parameters at 9B tokens is
small -- Pythia-410M sees 300B tokens (30x ours) and reaches only ~34% on
HellaSwag. The tasks that carry signal at this scale are **lambada**, **sciq**
and held-out perplexity (`runs/eval_ckpt_cpu.py`, the project's ledger).
Absolute scores are not the point; the paired deltas below are.

## The three arms (`run_all.sh`)

1. vanilla base, `clean` -- the checkpoint thinking was trained FROM
2. thinking checkpoint, `clean` -- **did the objective make a better base LM?**
3. thinking checkpoint, `think` -- **does thinking at inference help?**

(1 vs 2) isolates the training objective; (2 vs 3) isolates thinking itself
on identical weights.

## Think mode

One thought, generated greedily at the end of the prompt, spliced in as
`<T>...</T>`, and the answer is then scored with it visible. This is
INFERENCE-TIME thinking, not the training-time dense-anchor layout -- it is
the question a benchmark actually asks, and being independent of anchor
geometry it lets one script compare checkpoints from runs with different
`k`, `L` and gating.

## Reading the output (`compare.py`)

Every arm scores identical examples in identical order, so comparisons are
**paired** -- which matters at n=500, where an unpaired 2-point difference is
noise but a paired one often is not. Reported per task:

- accuracy delta with a **McNemar** exact test on discordant pairs;
- mean **gold log-prob** delta with a paired t-test. This is the more
  sensitive instrument at this scale because it uses the margin rather than
  just the argmax, and it moves before accuracy does.

Use `--norm` for length-normalised accuracy (`acc_norm`), which is the more
standard headline for multiple-choice tasks with uneven answer lengths.

## Running

    bash benchmark/run_all.sh                       # all three arms, then the report
    THINK=/path/to/ckpt.pt TAG=myrun bash benchmark/run_all.sh

CPU-only by design (`USE_TORCH_XLA=0`), so training keeps the Neuron device.
Clean arm ~1h, think arm ~5h (16 sequential decode steps per example).

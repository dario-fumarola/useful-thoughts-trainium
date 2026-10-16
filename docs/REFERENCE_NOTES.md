# Reference notes and verification dates

## Prior reviewed scientific basis (carried forward, not rerun here)

RLP motivates a mature Qwen3-1.7B base and thought-policy learning from future-text
predictive reward. Our G=4/L=1024/context=512/single-update configuration is a
combined engineering choice. It is not the complete published recipe, a tested
reproduction, or a guaranteed positive result. Common-baseline cancellation is
an algebraic property of the chosen LOO advantages; it does not fix a moving
reader or make variable-length normalization unbiased.

Quiet-STaR already uses a multi-token non-myopic prediction objective; extending
beyond the immediate next token is not novelty by itself. BoLT's teacher warm start,
TPT's thought-augmented data, and mature reasoning checkpoints in other work all
mean that different sources of initial competence must be separated experimentally.
Fast Quiet-STaR is relevant to later compression/internalization, not proof that
an extremely short initial thought budget will bootstrap this model.

Safety reflection and deliberative safety methods are already active research
areas. Keep their objectives distinct from predicting generic text or producing
plausible social commentary. The roadmap identifies narrower causal questions.
The primary citations and versioned paper links are in SOURCES.md.

## Packaging-time official documentation checks — 2026-10-16

- Codex reads project AGENTS.md instructions. The root file here is kept short;
  the launch script passes the separate opening task without changing permissions.
  https://developers.openai.com/codex/guides/agents-md
  https://developers.openai.com/codex/cli
- AWS documents trn2 logical-core configuration and runtime visibility controls.
  Device isolation must still be verified on the actual process layout.
  https://awsdocs-neuron.readthedocs-hosted.com/en/v2.28.1/about-neuron/arch/neuron-features/logical-neuroncore-config.html
  https://awsdocs-neuron.readthedocs-hosted.com/en/latest/neuron-runtime/guides/configuration-guide.html
- The XLA ZeRO-1 tutorial says its optimizer step handles distributed reduction;
  adding xm.optimizer_step or manual reduction is not appropriate for that path.
  The page also warns of a future XLA-training support transition. Preserve the
  working installed stack; do not infer that a newly installed release is compatible.
  https://awsdocs-neuron.readthedocs-hosted.com/en/v2.30.0/frameworks/torch/torch-neuronx/tutorials/training/zero1_gpt2.html
- Hugging Face documents commit-specific snapshot downloads and streaming datasets.
  The bundled helpers use explicit bounded network actions; live download APIs
  were not executed during packaging and still need instance-side checks.
  https://huggingface.co/docs/huggingface_hub/guides/download
  https://huggingface.co/docs/datasets/stream

These checks support interface choices, not hardware performance or scientific
claims. Pin resolved asset revisions and inspect installed library APIs at use.

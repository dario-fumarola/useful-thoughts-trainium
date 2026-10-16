# Sources and what they support

Checked October 16, 2026. URLs below are provenance for this research kit, not statements that code or model training was executed here. Primary sources were used for technical claims.

## User-provided material

`ema-trainium-pre-thinking (1).zip`, `Thinking_in_Pre_training (1).zip`, and `trascrizione-2026-10-13.txt`. The current implementation work used the first archive's actual code. The transcript supplies the owner's willingness to trade some NTP quality for downstream benefits, interest in mature checkpoints, uncertainty about checkpoint backups, and explicit exclusion of Bedrock. The weekend recipe and time allocation are recommendations, not decisions claimed to have been made in the call.

The source manifest records exact hashes for the three files patched. The old model initialization function explicitly uses `from_config` followed by fresh initialization. The OGeo label defect was reproduced and its narrow repair tested on the actual class. Neither finding certifies every other file in the archive.

## RLP

- Paper v2: https://arxiv.org/html/2510.01265v2
- PDF: https://arxiv.org/pdf/2510.01265v2 — prompt visually checked on printed page 21; the HTML omitted the prompt box.
- Authors' project summary: https://research.nvidia.com/labs/adlr/RLP/
- Authors' code repository: https://github.com/NVlabs/RLP — the checked repository lists training code as forthcoming; no public trainer was present in its displayed tree.

Supports using a mature Qwen3-1.7B-Base as a reference condition, stopped information-gain rewards and group-relative thought updates. The detailed method excludes an ordinary NTP loss even though the introduction contains less precise wording. The length ablation motivates giving thoughts substantially more room. Our G=4/L=1024 combination, short context, smaller global batch, source data, prompt serialization and single-update implementation differ from the complete published recipe. Its reported gains are not our expected performance guarantee.

The algebraic cancellation of a common no-thought baseline in leave-one-out advantages is independently derived in the kit and tested by exact small-bandit enumeration. It is not an empirical replication of the authors' EMA ablation, and does not remove self-referential-reader concerns.

## Model and data

- Qwen3-1.7B-Base: https://huggingface.co/Qwen/Qwen3-1.7B-Base
- Config: https://huggingface.co/Qwen/Qwen3-1.7B-Base/raw/main/config.json
- Qwen3-0.6B-Base fallback: https://huggingface.co/Qwen/Qwen3-0.6B-Base
- FineMath: https://huggingface.co/datasets/HuggingFaceTB/finemath

The model card distinguishes pretrained base weights from post-trained Qwen models. The model config specifies tied word embeddings. FineMath describes mathematical educational web text, a high-quality `finemath-4plus` subset, licensing and benchmark decontamination. The preparation script is not a substitute for verifying selected-source rights, near-duplicates and benchmark overlap.

## Hardware and optimizer

- Trn2 instance specification: https://aws.amazon.com/ec2/instance-types/trn2/
- Logical-core configuration: https://awsdocs-neuron.readthedocs-hosted.com/en/latest/about-neuron/arch/neuron-features/logical-neuroncore-config.html
- Neuron device listing: https://awsdocs-neuron.readthedocs-hosted.com/en/v2.29.1/tools/neuron-sys-tools/neuron-ls.html
- ZeRO-1 tutorial: https://awsdocs-neuron.readthedocs-hosted.com/en/v2.30.0/frameworks/torch/torch-neuronx/tutorials/training/zero1_gpt2.html
- Current release notes: https://awsdocs-neuron.readthedocs-hosted.com/en/latest/about-neuron/whats-new.html

Supports the 16-chip trn2.48xlarge, default LNC2 organization, per-core rather than aggregate memory constraint, and the need not to double-reduce gradients on the ZeroRedundancyOptimizer path. These pages do not establish that this custom workload compiles or fits on the user's actual AMI. Current serving support is not equivalent to a qualified training/rollout pipeline.

## Longer-term context

- Quiet-STaR: https://arxiv.org/abs/2403.09629
- BoLT: https://arxiv.org/abs/2503.18866
- Fast Quiet-STaR: https://arxiv.org/abs/2505.17746
- Thinking Augmented Pre-training: https://arxiv.org/abs/2509.20186
- Beyond Safe Data: Pretraining-Stage Alignment with Regular Safety Reflection: https://arxiv.org/abs/2606.19168
- Do Thinking Tokens Help with Safety?: https://arxiv.org/abs/2606.25013

These motivate specific comparisons, not a claim that all methods transfer to the proposed small-model regime. The plan deliberately postpones social/safety implementation rather than pretending broad novelty in already active areas.

## Validation limits

The tests exercise objective arithmetic, causal target alignment, prefix/future separation, stop masks, eager behavior/replay consistency, deterministic record splitting, paired statistics, budget arithmetic and the extracted legacy buffer. The small-model smoke executes real CPU optimizer updates. **No Qwen checkpoint was downloaded or evaluated; no Neuron device was available; no distributed Neuron update, compiled sampler, actual throughput or full resume was tested.** Those are required instance-side acceptance checks.

You are continuing Dario's Trainium thinking-pretraining research on the existing
instance. This folder is the handoff repo. Read AGENTS.md, STATUS.md, NEXT_STEPS.md,
context/PROJECT_CONTEXT.md, and context/EVIDENCE_LEDGER.md before choosing work.
Then read the fixed plan and implementation handoff; do not redesign the study.

The immediate goal is a trustworthy mature-model comparison: released
Qwen3-1.7B-Base, ordinary continuation C versus signed thought-policy training R,
with both direct and thought-enabled evaluation. The long-term question is what
initial competence and guidance make useful computation learnable, how early it
can arise, and how to compress/internalize it. Social and safety extensions must
remain visible in the roadmap, but are not extra objectives for this weekend.

The package has runnable CPU primitives, tests, save/resume demonstrations,
asset/data tools, and relevant original source. It DOES NOT yet have an optimized,
hardware-qualified Neuron trainer. Your first substantive job is implementing
and qualifying that missing driver using the installed working stack and the
reference semantics. Do not pretend an import or a toy test qualifies hardware.

Start by checking existing local/checks reports and environment, resolving only
unknown instance/asset facts, and running the tests if not already run on this
source revision. Work from the current NEXT_STEPS item. Implement the smallest
ordinary-causal Neuron path, not the historical custom insertion geometry.
Qualify real pretrained loading, targets, prefix-only generation, cached rollout
versus replay, one gradient reduction, finite FP32 updates, and save/resume.
Measure complete optimizer steps before locking the exposure.

Use the existing instance only. Do not call Bedrock, provision resources, change
IAM/networking, install a different accelerator stack, publish this repository,
or transmit private material to an unapproved location. Keep secrets out of files.
Discover the capacity cutoff and approved backup destination if available; ask
only when required information cannot be resolved locally. Do not guess either.

Once qualified, run the fixed C/R pair within the declared deadline and reserve
evaluation/checkpoint time. Prefer a second seed to another reward variant.
Evaluate downstream even when local likelihood is negative. Show separate
training and inference benefits; retain failures and report actual exposure.
No silent recipe changes, optimistic checkpoint selection, or oracle deployment.

Keep STATUS.md compact and current, with the exact next command, active job and
checkpoint. Use DECISIONS.md for material changes. Make small local commits and
leave a restartable handoff before ending a session. Begin with useful work now;
do not return only another proposal or rewrite the paper.

# Active Neuron implementation

`neuron_driver.py` implements the C/R path, with a bounded `--qualify` mode and a
separate frozen-specification requirement for scientific runs. `neuron_backend.py`
owns causal generation, reader scoring, and gradient replay. The production
candidate uses FP32 after long-trace BF16 replay failed numerical qualification.

The `qualify_*.py` entrypoints check released weights, replay, real collectives,
and reader behavior. A CPU run is labeled as such. Development accelerator
inference has run through `evaluate.py`; likelihood and local-content evaluators
have their own qualification requirements.

See [STATUS.md](../STATUS.md) for actual jobs and remaining checks. No scientific
C/R pair has launched yet. Full-model numerical resumption and complete-step
timing remain prerequisites. The specification is in
[NEURON_IMPLEMENTATION.md](../docs/NEURON_IMPLEMENTATION.md); historical code stays
under `legacy/` and is not an active launcher.

# Active production implementation goes here

Implement `neuron_driver.py` as specified in `../docs/NEURON_IMPLEMENTATION.md`.
There is deliberately no fake executable launcher or unverified trainer in this
folder. `../weekend_thinking/backend_contract.py` defines the input boundary;
`reference.py` defines CPU semantics; `legacy/` supplies runtime examples.

The next agent's job is to fill this specific gap, qualify it on the existing
instance, then execute the fixed comparison. CPU toy sessions are already runnable
through `../scripts/reference_session.py`, but are not Neuron training.

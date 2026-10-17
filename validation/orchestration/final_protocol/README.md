# Final evaluation preparation

These controllers prepare the first study's final evaluation on the existing node.
The numerical evaluators remain the sealed `source-256-001` implementations.
Exact reviewed copies are staged in `local/evaluation/final-protocol-v1`.
No final plan, model lock, projection or evaluation freeze was created during
controller development. The accompanying tests use synthetic fixtures.

The sequence is:

1. `prepare_final.py declare` records metadata after the complete B0 DEV512 audit.
   It copies the actual development protocol dictionaries, fixes terminal256
   model paths and canonical source identities, and remains `PREPARED_NOT_LOCKED`.
2. `prepare_final.py lock` requires all four real terminal256 runs, successful
   exits, process cleanup and verified terminal backups. It calls the existing
   full model/checkpoint integrity checks, then writes an immutable pre-gold lock.
3. `prepare_final.py project-gsm8k` runs the existing canonical numeric projector
   after that lock. `postlock_project.py` projects ARC and LAMBADA after checking
   the lock, five identities, source closure, canonical revisions and deadline.
4. `prepare_final.py assemble` binds projected hashes to the locked recipe and
   writes `evaluation-freeze-v2` with its actual later timestamp and lock hash.

Use the qualified virtual environment. Every output path must be new. The CLI
requires the plan or lock's explicit SHA256; it never chooses models or launches
inference. For example, after each preceding stage is actually complete:

```bash
python prepare_final.py declare --output plan-v1.json
python prepare_final.py lock --plan plan-v1.json --plan-sha256 PLAN_SHA --output pre-gold-lock.json
python prepare_final.py project-gsm8k --pre-gold-lock pre-gold-lock.json --lock-sha256 LOCK_SHA --output projections/gsm8k
python postlock_project.py --pre-gold-lock pre-gold-lock.json --lock-sha256 LOCK_SHA --task arc_challenge --output projections/arc_challenge
python postlock_project.py --pre-gold-lock pre-gold-lock.json --lock-sha256 LOCK_SHA --task lambada --output projections/lambada
python prepare_final.py assemble --pre-gold-lock pre-gold-lock.json --lock-sha256 LOCK_SHA --projections projections --output evaluation-freeze.json
```

The final inventory has B0 plus C/R for both seeds at terminal256. Its13 unique
protocols cover three GSM8K modes, four secondary task/mode combinations, local
TEST2048 generation and five writer-specific local scoring protocols. Final
inference uses new TEST bundles, `--frozen-protocol`, and full item coverage with
failures retained. Root separately releases finite hardware allocations.

The projection tests check choice ordering, label mapping, literal LAMBADA word
boundaries, complete coverage and rejection before canonical I/O. Planner tests
check source additions, incomplete cohorts, wrong run identities, manifest
substitution and disabled assertions. Test results are recorded in
`validation/final_evaluation_preparation_20261017.json`.

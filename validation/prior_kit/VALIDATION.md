# Executed validation

- Python: 3.13.5
- PyTorch: 2.10.0+cpu
- NumPy: 2.3.5
- Unit tests: **44 passed**; exact captured output in `TEST_REPORT.txt`.
- CPU smoke: six paired thought/vanilla optimizer updates on a small recurrent model; finite updates and behavior/replay agreement checked.
- Legacy target repair: 12 actual extracted-geometry configurations; corrected target with unchanged layout/branch targets.
- Pretokenized record preparation, paired analysis, synthetic budget arithmetic and the read-only host reporter executed.
- Python source compiled successfully with `compileall`.

The smoke, timing example and example evaluation scores are synthetic plumbing checks, not research results. The supplied host report is from a non-Neuron environment.

**Not executed:** Qwen loading or tokenization, actual dataset downloading, Neuron compilation, cached accelerator decoding, device memory/throughput, real distributed reduction, hardware partition isolation, exact multi-rank resume, downstream model evaluation, or checkpoint upload. Those remain required acceptance work on the instance.

# Short execution queue

Mark complete only from an actual artifact. Current commands/jobs are in STATUS.md.

1. Hardware-check the integrated checkpoint RNG ordering repair. Preserve the
   old all-rank RNG failure; require exact next stochastic draws.
2. Complete the repaired full-Qwen signed-gradient comparison on core 2.
   Loss now agrees; backward is compiling. Keep R32 off until it passes.
3. Use one ordinary C32 first-step checkpoint to diagnose the remaining strict
   CPU per-value comparison failure. Tiny identical-gradient AdamW already passes;
   do not hide the 754 failed coordinates behind passing aggregate metrics.
4. Refresh both qualification arms to one reviewed source. Require full model,
   masters, moments, counters, exact RNG and first resumed R actions/rewards at
   tight original tolerances, plus ten complete steady steps per arm.
5. Run full-cap local/secondary accelerator evaluation checks on freed upper
   cores. Preserve failures and keep official final test outcomes unread.
6. Freeze two-seed exposure from actual complete-step timing, fresh-data supply
   and the prospective 14-day schedule. Start identical released weights for C/R;
   supervise finite jobs and verified EBS checkpoint copies.
7. Evaluate terminal direct/thought policies, base reference, local controls and
   crossed readers. Save costs, failures, examples and uncertainty; choose the
   next experiment from the resulting evidence.

Active driver: `train/neuron_driver.py`. CPU semantics: `weekend_thinking/core.py`
and `reference.py`. Never execute archived launchers or use a long run as a
substitute for an unqualified gradient/update/resume path.

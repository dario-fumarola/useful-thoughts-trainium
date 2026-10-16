"""Digest monitor for a think run.  Writes MONITOR.md, prints a one-line status.

Usage: python monitor_think.py <metrics.jsonl> <MONITOR.md>

WHY THIS WAS REWRITTEN (2026-10-07 08:07 tick, G154 item 6).  Every reported
quantity used to be `r.get("<run-13 key>", 0.0)`.  Runs 14-95 do not log those
keys, so MONITOR.md -- the at-a-glance health report, regenerated every 5
minutes and read by the owner -- printed `+0.00` for ten of its twelve columns
for weeks.  A default of 0.0 for an absent key turns "we did not measure this"
into "we measured this and it is zero", which is the worst failure a report can
have: it is indistinguishable from a dead signal.  So:

  * A SLOT IS LOOKED UP ACROSS A CANDIDATE LIST and prints `n/a` when none of
    its keys is in the file.  Nothing is ever defaulted to zero.
  * THE HEADER NAMES THE SCHEMA IT MATCHED and lists, explicitly, every key
    present in the metrics that this script does not interpret -- so the next
    trainer's columns announce themselves instead of vanishing.
  * The title comes from the run directory, not from a hard-coded "run 13".

ntp1 health is unchanged: RUNAWAY appears when ntp1's 20-step rolling mean sits
>0.5 nats above the best seen.  Auto-kill was DISABLED 2026-09-05 by user
directive, so the condition is reported as "DRIFT!", a token the supervisor's
case pattern deliberately does not match.  Stopping is manual.
"""
import json
import os
import sys
import time

import numpy as np

# Each slot: (label, [candidate keys, best first], fmt, direction note).
# Candidates are ordered modern-first; the legacy run-13 key is kept last so an
# old metrics file still reads correctly.
SLOTS = [
    # drop_full/drop_best are the selection-line trainers' names for the same
    # two quantities -- BUT ONLY IN THAT FAMILY.  The 2026-10-07 02:07 journal stated
    # "drop_full == gain_mean" as a universal and it is FALSE on think_hintmix.py (runs
    # 96/97/98), which sets drop_full = gain1_mean, the k=1 gain: verified on all 5,738
    # logged rows, max|drop_full - gain_mean| up to 4.94 nats (agent/scratchpad/coldefs_1007.out,
    # 2026-10-09 10:07 tick, G209).  gain_mean is FIRST in this list and present on those runs,
    # so this slot reads correctly today; do not reorder it, and do not quote drop_full
    # across trainer families as if it were one quantity.
    ("typical thought gain", ["gain_mean", "drop_full", "g_mean"], "+.2f",
     "mean counterfactual gain of a SAMPLED thought (the deployable arm)"),
    ("best-of-G gain", ["gain_best", "drop_best", "g_win"], "+.2f",   # g_win is the real best-of-R;
     # think_train logs g_best=gm == g_mean on every row of runs 20-80 (JOURNAL:10382), never a best.
     "gain of the best thought in the group (needs an oracle; not deployable)"),
    ("gain of KEPT thoughts", ["drop_kept"], "+.2f",
     "gain restricted to thoughts the filter kept"),
    ("gain vs own shuffle", ["own_shuf"], "+.3f",
     "thought beats its OWN shuffled tokens at the same prefix (> 0 = content, "
     "not format priming) -- the selection line's headline number"),
    ("abstention rate", ["abst"], ".3f",
     "fraction of anchors where the writer chose silence"),
    ("fool KL", ["fool_kl"], ".3f",
     "how far the reader can be moved by a thought (0 = the reader went deaf)"),
    ("kept length", ["len_kept"], ".1f", "mean length of kept thoughts"),
    ("control speak rate", ["ctrl_rate"], ".3f", "the control arm's speak rate"),
    ("frac thoughts that help", ["helps", "g_pos"], ".3f",
     "fraction of sampled thoughts with gain > 0"),
    ("frac groups with a helpful best", ["best_helps"], ".3f",
     "fraction of anchors whose BEST of the G samples helps -- an E[max] quantity. On think_bestof "
     "the design really does select the best, so this is deployable there; on the hintmix family "
     "(runs 96/97/98) NOTHING selects it, so there it is an ORACLE over G=8 unselected samples and "
     "is >= `helps` by construction (1.50x it on run 98, 0.605 vs 0.403, 0 violations in 1,403 rows). "
     "G209/G210: an oracle-selected rate must say so wherever it is quoted"),
    ("advantage of the best", ["adv_best"], "+.2f",
     "clipped A* actually applied to the writer"),
    ("keep rate", ["keep_rate"], ".3f",
     "fraction of thoughts kept by the filter -- BUT on the hintmix family (runs 96/97/98) this "
     "key is an alias of `helps`, verified identical on every row, because that design never "
     "gates: there a value does not mean a thought was kept, only that it helped"),
    ("reader CE", ["read_ce", "rd_ce"], ".3f",
     "reader's CE on the span with the thought (falling = learns to use it)"),
    ("KL to init", ["kl_ref"], ".3f",
     "how far the writer has moved from the frozen init (0 = still the init)"),
    ("gate pre BCE", ["gate_pre_bce"], ".3f", "before-the-fact gate head loss"),
    ("gate post acc", ["gate_post_acc"], ".3f",
     "after-the-fact discriminator accuracy (read against the majority class)"),
    # 2026-10-09 12:07 tick, G210. All three of these are computed from SAMPLE 0 of the G samples on
    # every trainer that has a G dimension (think_grpo.py:469/471, think_grpo_gate.py:580/582,
    # think_oracle.py:370/372, think_bestof.py:426/428, think_hintmix.py:329/331). Only
    # think_entropy.py, which generates one thought per anchor, watches all of them. Read the labels.
    ("modal 3-tok prefix share", ["open_share"], ".3f",
     "share of the most common FIRST-THREE-TOKEN prefix within one micro-batch -- not an 'opener' "
     "token, and on G-sampling trainers computed over SAMPLE 0 only (collapse bar: > 0.80). A modal "
     "share over m items cannot fall below 1/m, so READ THIS AGAINST 1/n_valid, NOT AGAINST 0: the "
     "floor is 0.167 at n_valid 6 and 0.126 at n_valid 8, and every non-collapsed run sits ON it "
     "(runs 90/95/98 never leave it by more than 0.015 over 2,898 rows). It is not blind -- it "
     "caught run 89 at 7.9x its floor and run 85 at 0.699 -- but 'open_share 0.13, flat' is the "
     "statistic's minimum, not a measurement of diversity"),
    ("digit share", ["digit_share"], ".3f",
     "mean share of single-digit tokens per thought (collapse bar: > 0.50). A MEAN, so sample 0 is "
     "an unbiased estimator of it with G-fold variance -- unlike the modal share above"),
    ("repeat share (sample 0 / kept)", ["rep_kept"], ".3f",
     "repetition within one thought (collapse bar: > 0.50). WHICH thought differs by trainer: the "
     "SELECTED best on think_bestof, the kept mask on think_grpo/_gate/think_oracle, and SAMPLE 0 "
     "on the hintmix family -- which keeps nothing at all, so 'kept' is empty there. Note also that "
     "etrig_monitor.sh SKIPS the D2 rep_kept clause whenever `r_std` is logged, which the hintmix "
     "family does: on run 98 this axis was never gated on for any of its 1,403 steps"),
    ("thought length", ["len_gen", "len_mean"], ".1f", "mean generated length"),
    # legacy run-13 slots, printed only when the keys are actually there
    ("imitation CE", ["imit_ce"], ".2f", "generator imitation loss (run-13 line)"),
    ("oracle gain", ["or_g"], "+.2f", "reader gain on oracle thoughts (run-13 line)"),
    ("think rate", ["think_rate"], ".2f", "learned gate's speak rate (run-13 line)"),
]

# Columns for the last-15-steps table, in priority order; only present keys are
# used, and the table is capped so it stays readable in a terminal.
TABLE_PRIORITY = [
    ("ntp1", ".3f"), ("gain_mean", "+.2f"), ("drop_full", "+.2f"),
    ("gain_best", "+.2f"), ("drop_best", "+.2f"), ("own_shuf", "+.3f"),
    ("abst", ".2f"),
    ("helps", ".2f"), ("best_helps", ".2f"), ("adv_best", "+.2f"),
    ("keep_rate", ".2f"), ("read_ce", ".2f"), ("kl_ref", ".3f"),
    ("gate_pre_bce", ".2f"), ("gate_post_acc", ".2f"), ("open_share", ".2f"),
    ("len_gen", ".1f"), ("g_mean", "+.2f"), ("g_pos", ".2f"),
    ("or_g", "+.2f"), ("imit_ce", ".2f"), ("dt", ".0f"),
]
MAX_COLS = 11

# Keys that are bookkeeping, not signals: present-but-unreported is expected.
BOOKKEEPING = {"step", "dt", "lr", "world", "tokens_seen", "n_sm", "n_or",
               "ntp1",   # the base-health headline, reported above the slots
               "n_ng", "n_th", "n_anch", "n_elig", "gnorm", "gnorm_h",
               "t_gen", "t_score", "n_cand", "n_valid"}


def main():
    metrics_path, out_path = sys.argv[1], sys.argv[2]
    try:
        rows = [json.loads(l) for l in open(metrics_path) if l.strip()]
    except FileNotFoundError:
        print("no metrics yet")
        return
    rows = [r for r in rows if "ntp1" in r and "step" in r]
    if not rows:
        print("no metrics yet")
        return

    run = os.path.basename(os.path.dirname(os.path.abspath(metrics_path)))
    present = set()
    for r in rows:
        present |= set(r)

    ntp1 = np.array([r["ntp1"] for r in rows], dtype=float)
    n = len(rows)
    W = min(20, n)
    roll = np.convolve(ntp1, np.ones(W) / W, mode="valid")
    cur, best = float(roll[-1]), float(roll.min())
    drift = cur - best
    # Run 14, user directive 2026-09-05: let the run go, do NOT auto-kill.
    runaway = " DRIFT!" if (n >= 30 and drift > 0.5) else ""

    def pick(cands):
        for k in cands:
            if k in present:
                return k
        return None

    def trend(key, w=20):
        """(first-w mean, last-w mean) over rows that actually carry the key.
        Rows missing the key are SKIPPED, never counted as zero."""
        vals = [float(r[key]) for r in rows if key in r and r[key] is not None]
        if not vals:
            return None, None
        return float(np.mean(vals[:w])), float(np.mean(vals[-w:]))

    live, dead = [], []
    for label, cands, spec, note in SLOTS:
        k = pick(cands)
        if k is None:
            dead.append(label)
            continue
        a, b = trend(k)
        if a is None:
            dead.append(label)
            continue
        live.append((label, k, a, b, spec, note))

    # one-line status: ntp1 health plus the live signals, compactly
    bits = [f"step {rows[-1]['step']} ntp1 {rows[-1]['ntp1']:.3f} "
            f"(roll20 {cur:.3f}, drift {drift:+.3f}){runaway}"]
    for label, k, a, b, spec, _ in live:
        bits.append(f"{k} {format(a, spec)}->{format(b, spec)}")
    line = " | ".join(bits)
    print(line)

    reported = {k for _, cands, _, _ in SLOTS for k in cands}
    unreported = sorted(present - reported - BOOKKEEPING)

    md = [f"# Monitor: {run}", "",
          f"Updated {time.strftime('%F %T')} | {n} logged steps | "
          f"{len(live)} of {len(SLOTS)} signal slots present in this file", "",
          f"- **latest**: {line}",
          f"- **ntp1 (base health)**: roll20 {cur:.3f}, best {best:.3f}, "
          f"drift {drift:+.3f} (auto-kill DISABLED 2026-09-05 by user "
          f"directive; >+0.5 is flagged as DRIFT! only)",
          "",
          "## signals (first-20-step mean -> last-20-step mean)", ""]
    for label, k, a, b, spec, note in live:
        md.append(f"- **{label}** (`{k}`): {format(a, spec)} -> "
                  f"{format(b, spec)}  — {note}")
    md += ["",
           "## not logged by this trainer (printed as n/a, NOT as zero)", "",
           ("- " + ", ".join(dead)) if dead else "- (none: every slot is live)"]
    if unreported:
        md += ["",
               "## logged but not interpreted by this script", "",
               "- " + ", ".join(f"`{k}`" for k in unreported),
               "",
               "  Add them to `SLOTS` in `pre-training/train/monitor_think.py` "
               "if they matter; they are listed here so a new trainer's "
               "columns cannot disappear silently."]

    cols = [(k, s) for k, s in TABLE_PRIORITY
            if k in present and k != "dt"][:MAX_COLS]
    if "dt" in present:                      # s/step is always worth a column
        cols.append(("dt", ".0f"))
    md += ["", "## last 15 steps", "",
           "| step | " + " | ".join(k for k, _ in cols) + " |",
           "|---|" + "---|" * len(cols)]
    for r in rows[-15:]:
        cells = []
        for k, s in cols:
            v = r.get(k)
            cells.append("n/a" if v is None else format(float(v), s))
        md.append(f"| {r['step']} | " + " | ".join(cells) + " |")

    with open(out_path, "w") as f:
        f.write("\n".join(md) + "\n")


if __name__ == "__main__":
    main()

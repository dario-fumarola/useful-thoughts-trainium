"""NTP-loss + thinking-signal graph daemon for a long-horizon think run.

Parses the run's metrics jsonl and redraws <out>.png every --interval
seconds. Top panel mirrors ntp_monitor.py: x = total corpus tokens in
MILLIONS (base checkpoint tokens + tokens consumed by this run), y =
pass-1 cross-entropy in nats/token (raw + rolling mean, right axis =
perplexity). Bottom panel is the thinking run's health: counterfactual
gain of the best sampled thought (gB) vs the model's own greedy guess
(gS) -- the gB-gS margin is the whole point of run 10 -- plus the
gate's oracle/negative weights (wOr/wNg, discrimination) on the right.

2026-10-07 08:07 tick (G154 item 6): the bottom panel used
`r.get("<run-13 key>", 0.0)`, so for every trainer since run 13 it drew
flat zero lines that look exactly like a dead signal.  Series are now
resolved across a candidate list and a series that is ABSENT is NOT
DRAWN -- the panel says which keys are missing instead of plotting them
as zero.

  python think_ntp_monitor.py --metrics <jsonl> --out <png> --base-m 870.3
  python think_ntp_monitor.py ... --once     # single redraw
"""
import argparse
import json
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

GRAD_ACCUM = 8      # micros per step; only used when metrics lack n_sm
_DISC_WARNED = False   # see main(): announce a skipped panel exactly once


def parse(path):
    by_step = {}
    with open(path, errors="ignore") as f:
        for line in f:
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if "step" in r and "ntp1" in r:
                by_step[int(r["step"])] = r     # resume overlap: last wins
    return [by_step[s] for s in sorted(by_step)]


def roll(y, w=20):
    w = min(w, len(y))
    return np.convolve(y, np.ones(w) / w, mode="valid"), w


def draw(metrics_path, out_path, base_m, tok_per_step, title):
    recs = parse(metrics_path)
    fig, (ax, ax2) = plt.subplots(
        2, 1, figsize=(11, 8.5), sharex=True,
        gridspec_kw={"height_ratios": [3, 2]})
    if recs:
        steps = np.array([r["step"] for r in recs])
        x = base_m + steps * tok_per_step / 1e6
        y = np.array([r["ntp1"] for r in recs])
        ax.plot(x, y, lw=0.6, alpha=0.30, color="#4878cf",
                label="NTP CE pass-1 (per step)")
        if len(y) >= 5:
            r20, w = roll(y)
            ax.plot(x[w - 1:], r20, lw=2.0, color="#1f4e9c",
                    label=f"NTP CE (rolling {w})")
        ax.axhline(y=float(y[0]), color="grey", lw=0.8, ls="--",
                   alpha=0.6, label=f"start ({y[0]:.3f})")
        last = f"step {steps[-1]}  ntp1 {np.mean(y[-20:]):.3f}"
        ax.set_title(f"{title} — NTP loss vs tokens   [{last}]   "
                     f"updated {time.strftime('%H:%M:%S')}")
        sec = ax.secondary_yaxis("right", functions=(np.exp, np.log))
        sec.set_ylabel("perplexity")

        # thinking signals.  NEVER default a missing key to 0.0: an absent
        # series must be absent from the plot, not drawn as a flat zero.
        present = set()
        for r in recs:
            present |= set(r)

        def series(*cands):
            """(values, key) for the first candidate key present, else None."""
            for k in cands:
                if k in present:
                    v = np.array([r.get(k, np.nan) for r in recs], dtype=float)
                    if np.isfinite(v).sum() >= 5:
                        return v, k
            return None

        missing = []
        gain_best = series("gain_best", "drop_best", "g_win")   # g_win = real best-of-R; g_best is a
                                                            # mislabelled copy of g_mean (JOURNAL:10382)
        gain_mean = series("gain_mean", "drop_full", "g_mean")
        if gain_best is None:
            missing.append("best-thought gain")
        if gain_mean is None:
            missing.append("typical-thought gain")
        if gain_best is not None and gain_mean is not None:
            (vB, kB), (vS, kS) = gain_best, gain_mean
            rB, w = roll(vB); rS, _ = roll(vS)
            ax2.plot(x[w - 1:], rB, lw=1.8, color="#2ca02c",
                     label=f"best thought in the group ({kB}, rolling)")
            ax2.plot(x[w - 1:], rS, lw=1.8, color="#ff7f0e",
                     label=f"typical sampled thought ({kS}, rolling)")
            ax2.fill_between(x[w - 1:], rS, rB, where=rB > rS,
                             color="#2ca02c", alpha=0.15,
                             label="selection margin (best − typical)")
        else:
            for got in (gain_best, gain_mean):
                if got is not None:
                    v, k = got
                    rv, w = roll(v)
                    ax2.plot(x[w - 1:], rv, lw=1.8, color="#2ca02c",
                             label=f"{k} (rolling)")
        ax2.axhline(0.0, color="grey", lw=0.8, alpha=0.6)
        ax2.set_ylabel("counterfactual gain (nats, k-window)")

        axg = ax2.twinx()
        for cands, colour, lab in (
                (("best_helps",), "#9467bd", "frac groups with a helpful best"),
                (("helps", "g_pos"), "#c44e52", "frac thoughts that help"),
                (("keep_rate",), "#8c8c8c", "keep rate"),
                (("own_shuf",), "#1f77b4", "gain vs own shuffle")):
            got = series(*cands)
            if got is None:
                missing.append(lab)
                continue
            v, k = got
            rv, w = roll(v)
            axg.plot(x[w - 1:], rv, lw=1.0, alpha=0.8, color=colour,
                     label=f"{lab} ({k})")
        axg.set_ylabel("fraction / rate")
        h1, l1 = ax2.get_legend_handles_labels()
        h2, l2 = axg.get_legend_handles_labels()
        if h1 or h2:
            ax2.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=8)
        if missing:
            ax2.text(0.99, 0.02,
                     "not logged by this trainer (not plotted, NOT zero): "
                     + ", ".join(missing),
                     transform=ax2.transAxes, ha="right", va="bottom",
                     fontsize=7, color="#777777")
    else:
        ax.set_title(f"{title} — waiting for the first training steps "
                     f"(compiles)   updated {time.strftime('%H:%M:%S')}")
    ax.set_ylabel("cross-entropy (nats/token)")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right", fontsize=9)
    ax2.grid(alpha=0.3)
    ax2.set_xlabel(f"total corpus tokens (millions, incl. {base_m:.0f}M "
                   f"in the init checkpoint)")
    fig.tight_layout()
    tmp = out_path + ".tmp.png"
    fig.savefig(tmp, dpi=110)
    plt.close(fig)
    os.replace(tmp, out_path)


def draw_discarded(metrics_path, out_path, title):
    """Discarded thoughts per step: sampled thoughts that failed to beat
    the model's own greedy guess by distill-eps and were NOT distilled.
    Sample-slot count is estimated as n_th * n_sm / (all micros) since
    n_th spans all micro kinds; rolling means make the estimate solid."""
    def sm(r):   # metrics before 2026-09-01 lack n_sm; infer from accum
        return r.get("n_sm", GRAD_ACCUM - r.get("n_or", 0)
                     - r.get("n_ng", 0))
    recs = [r for r in parse(metrics_path) if sm(r) > 0]
    fig, ax = plt.subplots(figsize=(11, 5))
    if len(recs) >= 5:
        steps = np.array([r["step"] for r in recs])
        n_th = np.array([r["n_th"] for r in recs], dtype=float)
        n_sm = np.array([sm(r) for r in recs], dtype=float)
        micros = n_sm + np.array([r.get("n_or", 0) + r.get("n_ng", 0)
                                  for r in recs], dtype=float)
        est = n_th * n_sm / np.maximum(micros, 1.0)
        elig = np.array([r.get("n_elig", 0) for r in recs], dtype=float)
        disc = np.clip(est - elig, 0.0, None)
        ax.plot(steps, disc, lw=0.5, alpha=0.25, color="#c44e52")
        rd, w = roll(disc)
        ax.plot(steps[w - 1:], rd, lw=2.0, color="#a03035",
                label=f"discarded thoughts / step (rolling {w})")
        ax.set_ylabel("discarded thoughts per step", color="#a03035")
        axr = ax.twinx()
        rate = 100.0 * disc / np.maximum(est, 1.0)
        rr, w = roll(rate)
        axr.plot(steps[w - 1:], rr, lw=1.6, ls="--", color="#7a7a7a",
                 label=f"discard rate % (rolling {w})")
        axr.set_ylim(0, 100)
        axr.set_ylabel("discard rate (%)")
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = axr.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, loc="upper right", fontsize=9)
        last = f"step {steps[-1]}  rate {np.mean(rate[-20:]):.0f}%"
        ax.set_title(f"{title} — discarded thoughts (failed to beat the "
                     f"model's own guess)   [{last}]   "
                     f"updated {time.strftime('%H:%M:%S')}")
    else:
        ax.set_title(f"{title} — discarded thoughts: waiting for steps   "
                     f"updated {time.strftime('%H:%M:%S')}")
    ax.set_xlabel("training step (one step ≈ 32.8K corpus tokens)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    tmp = out_path + ".tmp.png"
    fig.savefig(tmp, dpi=110)
    plt.close(fig)
    os.replace(tmp, out_path)


KIND_LABEL = {
    "sample": "SAMPLED — the model's own free-form thought",
    "oracle": "oracle probe (spliced future tokens; calibration only)",
    "neg": "negative probe (garbage; the gate should reject it)",
}


def clean(s, n=None):
    s = s.replace("\n", " ⏎ ").replace("`", "'")
    return s if n is None or len(s) <= n else "…" + s[-n:]


def write_thoughts(dumps_path, out_md, title, keep=15):
    entries = []
    with open(dumps_path, errors="ignore") as f:
        for line in f:
            try:
                entries.append(json.loads(line))
            except ValueError:
                continue
    parts = [f"# {title} — recent thought dumps",
             f"updated {time.strftime('%F %H:%M:%S')} · "
             f"{len(entries)} dumps so far\n"]
    for e in entries[-keep:][::-1]:
        kind = KIND_LABEL.get(e.get("kind", "sample"), e.get("kind", "?"))
        parts.append(
            f"---\n### step {e.get('step', '?')} · {kind} · "
            f"gain {e.get('R_raw', 0.0):+.2f} nats\n"
            f"- **context tail:** …{clean(e.get('context', ''), 220)}\n"
            f"- **thought:** `{clean(e.get('thought', ''))}`\n"
            f"- **window to predict (next 64):** "
            f"{clean(e.get('window', ''), 260)}\n")
    tmp = out_md + ".tmp"
    with open(tmp, "w") as f:
        f.write("\n".join(parts) + "\n")
    os.replace(tmp, out_md)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--base-m", type=float, required=True,
                    help="tokens (M) already in the init checkpoint")
    ap.add_argument("--tok-per-step", type=int, default=32768,
                    help="visible NTP tokens per step (accum x world x T)")
    ap.add_argument("--title", default="think10")
    ap.add_argument("--interval", type=int, default=180)
    ap.add_argument("--once", action="store_true")
    args = ap.parse_args()
    out_dir = os.path.dirname(os.path.abspath(args.out))
    dumps = args.metrics.replace("_metrics.jsonl", "_dumps.jsonl")
    while True:
        try:
            draw(args.metrics, args.out, args.base_m, args.tok_per_step,
                 args.title)
        except FileNotFoundError:
            pass
        try:
            draw_discarded(args.metrics,
                           os.path.join(out_dir, "discarded_thoughts.png"),
                           args.title)
        except (FileNotFoundError, KeyError) as e:
            # GRPO trainers do not log n_th; a KeyError here used to kill the whole daemon
            # (run 94, 2026-10-05 14:27).  2026-10-07 10:07: catching it was right, but the panel
            # has then been SILENTLY ABSENT for every run since run 13 and nothing said so --
            # the same "missing is indistinguishable from empty" failure the 08:07 bindings are
            # about, one level up (a missing FIGURE instead of a missing number).  Say it once.
            global _DISC_WARNED
            if not _DISC_WARNED:
                _DISC_WARNED = True
                print(f"[think_ntp_monitor] discarded_thoughts.png NOT DRAWN: this run does not log "
                      f"{e!r}. The panel is absent, not empty.", flush=True)
        try:
            if dumps != args.metrics:
                write_thoughts(dumps, os.path.join(out_dir, "THOUGHTS.md"),
                               args.title)
        except FileNotFoundError:
            pass
        if args.once:
            break
        time.sleep(args.interval)


if __name__ == "__main__":
    main()

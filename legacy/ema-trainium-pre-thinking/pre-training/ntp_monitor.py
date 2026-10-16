"""NTP-loss graph daemon for the vanilla3 ladder run.

Parses the training log and redraws pre-training/ntp_loss.png every
--interval seconds:  x = total corpus tokens in MILLIONS (including the
372M already in the base checkpoint), y = cross-entropy in nats/token
(train loss raw + rolling mean, held-out mixture loss from the in-loop
evals, right axis = perplexity). Ladder rungs (~50M) shown as ticks.

  python ntp_monitor.py            # daemon (every 180s)
  python ntp_monitor.py --once     # single redraw
"""
import argparse
import os
import re
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BASE_M = 372.0                      # tokens already in vanilla_p0/step_1420
TOK_PER_STEP = 262144               # grad-accum 64 x world 4 x T 1024

STEP_RE = re.compile(
    r"^step\s+(\d+)/(\d+)\s+loss\s+([\d.]+).*?seen\s+([\d.]+)B")
EVAL_RE = re.compile(r"eval @ step (\d+):.*?mix=([\d.]+)")


def parse(log_path):
    steps, losses, seen = [], [], []
    ev_steps, ev_loss = [], []
    with open(log_path, errors="ignore") as f:
        for line in f:
            m = STEP_RE.match(line)
            if m:
                steps.append(int(m.group(1)))
                losses.append(float(m.group(3)))
                seen.append(float(m.group(4)) * 1000.0)   # B -> M
                continue
            m = EVAL_RE.search(line)
            if m:
                ev_steps.append(int(m.group(1)))
                ev_loss.append(float(m.group(2)))
    return steps, losses, seen, ev_steps, ev_loss


def draw(log_path, out_path):
    steps, losses, seen, ev_steps, ev_loss = parse(log_path)
    fig, ax = plt.subplots(figsize=(11, 6))
    if steps:
        x = BASE_M + np.array(steps) * TOK_PER_STEP / 1e6
        y = np.array(losses)
        ax.plot(x, y, lw=0.6, alpha=0.30, color="#4878cf",
                label="train CE (per step)")
        w = min(20, len(y))
        if len(y) >= 5:
            roll = np.convolve(y, np.ones(w) / w, mode="valid")
            ax.plot(x[w - 1:], roll, lw=2.0, color="#1f4e9c",
                    label=f"train CE (rolling {w})")
        if ev_steps:
            ex = BASE_M + np.array(ev_steps) * TOK_PER_STEP / 1e6
            ax.plot(ex, ev_loss, "o-", ms=5, lw=1.5, color="#c44e52",
                    label="val CE (mixture)")
        # ladder rungs
        for r in range(190, steps[-1] + 191, 190):
            ax.axvline(BASE_M + r * TOK_PER_STEP / 1e6, color="grey",
                       lw=0.4, alpha=0.25)
        last = (f"step {steps[-1]}/{7700}  "
                f"train {np.mean(losses[-20:]):.3f}"
                + (f"  val {ev_loss[-1]:.3f}" if ev_loss else ""))
        ax.set_title(f"vanilla3 — NTP loss vs tokens   [{last}]   "
                     f"updated {time.strftime('%H:%M:%S')}")
    else:
        ax.set_title("vanilla3 — waiting for the first training steps "
                     f"(compiles)   updated {time.strftime('%H:%M:%S')}")
    ax.set_xlabel("total corpus tokens (millions, incl. 372M base; "
                  "grey lines = ~50M ladder rungs)")
    ax.set_ylabel("cross-entropy (nats/token)")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right")
    sec = ax.secondary_yaxis(
        "right", functions=(np.exp, np.log))
    sec.set_ylabel("perplexity")
    fig.tight_layout()
    tmp = out_path + ".tmp.png"
    fig.savefig(tmp, dpi=110)
    plt.close(fig)
    os.replace(tmp, out_path)


def main():
    ap = argparse.ArgumentParser()
    base = os.path.dirname(os.path.abspath(__file__))
    ap.add_argument("--log", default=os.path.join(base, "..", "runs",
                                                  "vanilla3.log"))
    ap.add_argument("--out", default=os.path.join(base, "ntp_loss.png"))
    ap.add_argument("--interval", type=int, default=180)
    ap.add_argument("--once", action="store_true")
    args = ap.parse_args()
    while True:
        try:
            draw(args.log, args.out)
        except FileNotFoundError:
            pass
        if args.once:
            break
        time.sleep(args.interval)


if __name__ == "__main__":
    main()

"""Paired comparison of two (or more) score files from benchmark/score.py.

Every run scores identical examples in identical order, so comparisons are
PAIRED per example -- which matters a great deal at 500 examples, where an
unpaired difference of 2 points is inside the noise but a paired one often
is not.

  accuracy      McNemar exact test on the discordant pairs
  gold logprob  paired t-test (the more sensitive instrument at this scale,
                because it uses the margin, not just the argmax)

  python benchmark/compare.py --base results/vanilla_clean.jsonl \
      --cmp results/think18_clean.jsonl [--cmp results/think18_think.jsonl]
"""
import argparse, json, math
from collections import defaultdict


def load(p):
    d = defaultdict(dict)
    for l in open(p):
        r = json.loads(l)
        d[r["task"]][r["idx"]] = r
    return d


def mcnemar(b, c):
    """Exact two-sided binomial p on discordant pairs b (base only) / c (cmp only)."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(0, k + 1)) / (2 ** n)
    return min(1.0, 2 * p)


def paired_t(d):
    n = len(d)
    if n < 3:
        return 0.0, 1.0
    m = sum(d) / n
    var = sum((x - m) ** 2 for x in d) / (n - 1)
    se = math.sqrt(var / n) if var > 0 else 1e-12
    t = m / se
    # normal approximation is fine at n=500
    p = math.erfc(abs(t) / math.sqrt(2))
    return m, p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--cmp", action="append", required=True)
    ap.add_argument("--norm", action="store_true",
                    help="use length-normalised accuracy (acc_norm)")
    a = ap.parse_args()
    key = "correct_norm" if a.norm else "correct"
    B = load(a.base)
    print(f"base: {a.base}\n")
    for cp in a.cmp:
        C = load(cp)
        print(f"=== {cp} ===")
        print(f"{'task':<15}{'base':>8}{'cmp':>8}{'delta':>8}{'p(acc)':>9}"
              f"{'d logP':>9}{'p(logP)':>9}{'n':>6}")
        tot_b = tot_c = tot_n = 0
        for task in sorted(B):
            if task not in C:
                continue
            idx = sorted(set(B[task]) & set(C[task]))
            if not idx:
                continue
            bb = [B[task][i][key] for i in idx]
            cc = [C[task][i][key] for i in idx]
            b_only = sum(1 for x, y in zip(bb, cc) if x and not y)
            c_only = sum(1 for x, y in zip(bb, cc) if y and not x)
            dlp = [C[task][i]["gold_logprob"] - B[task][i]["gold_logprob"]
                   for i in idx]
            m, p_lp = paired_t(dlp)
            ab, ac = 100 * sum(bb) / len(bb), 100 * sum(cc) / len(cc)
            print(f"{task:<15}{ab:>7.1f}%{ac:>7.1f}%{ac-ab:>+7.1f}"
                  f"{mcnemar(b_only, c_only):>9.3f}{m:>+9.3f}{p_lp:>9.3f}"
                  f"{len(idx):>6}")
            tot_b += sum(bb); tot_c += sum(cc); tot_n += len(idx)
        if tot_n:
            print(f"{'MACRO(all ex)':<15}{100*tot_b/tot_n:>7.1f}%"
                  f"{100*tot_c/tot_n:>7.1f}%{100*(tot_c-tot_b)/tot_n:>+7.1f}"
                  f"{'':>9}{'':>9}{'':>9}{tot_n:>6}")
        print()


if __name__ == "__main__":
    main()

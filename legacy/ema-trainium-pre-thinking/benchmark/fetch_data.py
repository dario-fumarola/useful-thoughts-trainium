"""Download and normalise standard zero-shot likelihood benchmarks.

These are the tasks the base-LM literature reports for small models
(GPT-3, Pythia, OLMo, LLaMA all use this family, scored by likelihood
ranking with NO fine-tuning and no task heads -- the same frozen weights
answer everything).

Two record shapes:
  lambada        {"task", "context", "target"}          final-word prediction
  everything else{"task", "context", "choices", "gold"} N-way choice

Choices keep their leading space. Subsampling is deterministic (seed 0) so
every checkpoint sees identical examples and comparisons stay paired.

  python benchmark/fetch_data.py --limit 500
"""
import argparse, json, os, random

TASKS = {}


def task(name):
    def deco(fn):
        TASKS[name] = fn
        return fn
    return deco


@task("lambada")
def _lambada(ld):
    ds = ld("EleutherAI/lambada_openai", "en", split="test")
    out = []
    for ex in ds:
        t = ex["text"].rsplit(" ", 1)
        if len(t) == 2:
            out.append(dict(context=t[0], target=" " + t[1]))
    return out


@task("hellaswag")
def _hellaswag(ld):
    ds = ld("Rowan/hellaswag", split="validation")
    return [dict(context=ex["ctx"], choices=[" " + e for e in ex["endings"]],
                 gold=int(ex["label"])) for ex in ds if ex["label"] != ""]


@task("piqa")
def _piqa(ld):
    ds = ld("ybisk/piqa", split="validation", trust_remote_code=True)
    return [dict(context="Question: " + ex["goal"] + "\nAnswer:",
                 choices=[" " + ex["sol1"], " " + ex["sol2"]],
                 gold=int(ex["label"])) for ex in ds]


@task("winogrande")
def _winogrande(ld):
    ds = ld("allenai/winogrande", "winogrande_xl", split="validation")
    out = []
    for ex in ds:
        if ex["answer"] not in ("1", "2"):
            continue
        pre, post = ex["sentence"].split("_", 1)
        out.append(dict(context=pre.rstrip(),
                        choices=[" " + ex["option1"] + post,
                                 " " + ex["option2"] + post],
                        gold=int(ex["answer"]) - 1))
    return out


def _arc(ld, cfg):
    ds = ld("allenai/ai2_arc", cfg, split="test")
    out = []
    for ex in ds:
        labels = list(ex["choices"]["label"])
        if ex["answerKey"] not in labels:
            continue
        out.append(dict(context="Question: " + ex["question"] + "\nAnswer:",
                        choices=[" " + t for t in ex["choices"]["text"]],
                        gold=labels.index(ex["answerKey"])))
    return out


@task("arc_easy")
def _arc_e(ld): return _arc(ld, "ARC-Easy")


@task("arc_challenge")
def _arc_c(ld): return _arc(ld, "ARC-Challenge")


@task("openbookqa")
def _obqa(ld):
    ds = ld("allenai/openbookqa", "main", split="test")
    out = []
    for ex in ds:
        labels = list(ex["choices"]["label"])
        if ex["answerKey"] not in labels:
            continue
        out.append(dict(context=ex["question_stem"],
                        choices=[" " + t for t in ex["choices"]["text"]],
                        gold=labels.index(ex["answerKey"])))
    return out


@task("sciq")
def _sciq(ld):
    ds = ld("allenai/sciq", split="test")
    out = []
    rng = random.Random(0)
    for ex in ds:
        cor = ex["correct_answer"]
        opts = [ex["distractor1"], ex["distractor2"], ex["distractor3"], cor]
        rng.shuffle(opts)
        sup = (ex["support"].strip() + "\n") if ex["support"].strip() else ""
        out.append(dict(context=sup + "Question: " + ex["question"] + "\nAnswer:",
                        choices=[" " + o for o in opts],
                        gold=opts.index(cor)))
    return out


@task("boolq")
def _boolq(ld):
    ds = ld("google/boolq", split="validation")
    return [dict(context=ex["passage"] + "\nQuestion: " + ex["question"]
                 + "?\nAnswer:", choices=[" no", " yes"],
                 gold=int(ex["answer"])) for ex in ds]


def main():
    ap = argparse.ArgumentParser()
    here = os.path.dirname(os.path.abspath(__file__))
    ap.add_argument("--out-dir", default=os.path.join(here, "data"))
    ap.add_argument("--limit", type=int, default=500)
    ap.add_argument("--tasks", default="all")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    from datasets import load_dataset
    want = list(TASKS) if a.tasks == "all" else a.tasks.split(",")
    for name in want:
        try:
            rows = TASKS[name](load_dataset)
        except Exception as e:
            print(f"  SKIP {name}: {type(e).__name__} {str(e)[:110]}")
            continue
        rng = random.Random(0)
        if len(rows) > a.limit:
            rows = rng.sample(rows, a.limit)
        for r in rows:
            r["task"] = name
        p = os.path.join(a.out_dir, name + ".jsonl")
        with open(p, "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        nch = len(rows[0].get("choices", [])) if rows else 0
        print(f"  {name:14s} {len(rows):5d} examples"
              + (f"  {nch}-way (chance {100/nch:.0f}%)" if nch else "  final-word"))


if __name__ == "__main__":
    main()

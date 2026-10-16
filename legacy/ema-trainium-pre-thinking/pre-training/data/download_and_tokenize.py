"""Stream the pretraining mixture, tokenize it, write packed uint32 shards.

Layout produced under --out:

    shards/<domain>_train_000.bin      raw little-endian uint32 token ids
    shards/<domain>_val.bin
    manifest.json                      token counts + shard list per domain
    state.json                         consumed source files (for --resume)

Every document is terminated with <|endoftext|> (151643) so the packed stream
carries document boundaries. Note this is NOT the tokenizer's `eos_token_id`
(151645 = <|im_end|>), which is the *chat* terminator; 151643 is the right
separator for raw pretraining text.

uint32 is mandatory: Qwen3's vocab is 151936, so uint16 would silently wrap
every id above 65535.

Raw source files are deleted immediately after tokenizing, so peak disk stays
at roughly one source shard plus the growing .bin output.
"""

import argparse
import io
import json
import os
import shutil
import sys
import time

import numpy as np

os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")

EOD_TOKEN = 151643          # <|endoftext|>
DTYPE = np.uint32
TOKENIZER_ID = "Qwen/Qwen3-0.6B"

# Weights renormalised to 3 domains: the 5% code slice was dropped because
# smollm-corpus/python-edu (and stack-edu, the suggested substitute) store code
# by blob_id with no text column, and the gated pretraining-grade alternatives
# (starcoderdata, the-stack-v2) are not accessible with this account's token.
DOMAINS = {
    "fineweb-edu": dict(
        weight=70 / 95,
        repo="HuggingFaceFW/fineweb-edu",
        kind="parquet",
        prefix="sample/10BT/",
        text_column="text",
    ),
    "finemath": dict(
        weight=15 / 95,
        repo="HuggingFaceTB/finemath",
        kind="parquet",
        prefix="finemath-4plus/",
        text_column="text",
    ),
    "dclm": dict(
        weight=10 / 95,
        repo="mlfoundations/dclm-baseline-1.0",
        kind="jsonl_zst",
        prefix="global-shard_01_of_10/local-shard_0_of_10/",
        text_column="text",
    ),
}


class ShardWriter:
    """Appends token ids to <stem>_NNN.bin, rolling over at shard_tokens."""

    def __init__(self, out_dir, stem, shard_tokens):
        self.out_dir = out_dir
        self.stem = stem
        self.shard_tokens = shard_tokens
        self.shards = []
        self.total = 0
        self._fh = None
        self._in_shard = 0

    def _roll(self):
        if self._fh is not None:
            self._fh.close()
        name = f"{self.stem}_{len(self.shards):03d}.bin"
        self.shards.append(name)
        self._fh = open(os.path.join(self.out_dir, name), "wb")
        self._in_shard = 0

    def write(self, arr):
        if self._fh is None:
            self._roll()
        pos = 0
        while pos < len(arr):
            room = self.shard_tokens - self._in_shard
            if room == 0:
                self._roll()
                room = self.shard_tokens
            chunk = arr[pos:pos + room]
            self._fh.write(chunk.tobytes())
            n = len(chunk)
            pos += n
            self._in_shard += n
            self.total += n

    def close(self):
        if self._fh is not None:
            self._fh.close()
            self._fh = None


def iter_source_files(repo, prefix):
    from huggingface_hub import HfApi
    files = HfApi().list_repo_files(repo, repo_type="dataset")
    return sorted(f for f in files if f.startswith(prefix))


def read_texts_parquet(path, column, batch_rows):
    import pyarrow.parquet as pq
    pf = pq.ParquetFile(path)
    for batch in pf.iter_batches(batch_size=batch_rows, columns=[column]):
        yield [t for t in batch.column(column).to_pylist() if t]


def read_texts_jsonl_zst(path, column, batch_rows):
    import zstandard
    dctx = zstandard.ZstdDecompressor()
    buf = []
    with open(path, "rb") as fh, dctx.stream_reader(fh) as reader:
        for line in io.TextIOWrapper(reader, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            t = rec.get(column)
            if t:
                buf.append(t)
            if len(buf) >= batch_rows:
                yield buf
                buf = []
    if buf:
        yield buf


def tokenize_domain(name, spec, budget, val_tokens, out_dir, args, tok, state):
    """Fill val first, then train, stopping as soon as the budget is met."""
    from huggingface_hub import hf_hub_download

    raw_dir = os.path.join(args.out, "_raw")
    os.makedirs(raw_dir, exist_ok=True)

    train_w = ShardWriter(out_dir, f"{name}_train", args.shard_tokens)
    val_w = ShardWriter(out_dir, f"{name}_val", args.shard_tokens)

    files = iter_source_files(spec["repo"], spec["prefix"])
    done = set(state.get(name, {}).get("consumed", []))
    reader = read_texts_parquet if spec["kind"] == "parquet" else read_texts_jsonl_zst

    need = budget + val_tokens
    t_start = time.time()
    print(f"\n=== {name}: target {need/1e6:.0f}M tokens "
          f"({val_tokens/1e6:.0f}M val + {budget/1e6:.0f}M train) "
          f"from {len(files)} available source files", flush=True)

    for fname in files:
        if train_w.total + val_w.total >= need:
            break
        if fname in done:
            continue

        local = None
        try:
            local = hf_hub_download(spec["repo"], fname, repo_type="dataset",
                                    local_dir=raw_dir)
            for texts in reader(local, spec["text_column"], args.batch_rows):
                if train_w.total + val_w.total >= need:
                    break
                enc = tok(texts, add_special_tokens=False)["input_ids"]
                flat = []
                for ids in enc:
                    flat.extend(ids)
                    flat.append(EOD_TOKEN)
                arr = np.fromiter(flat, dtype=DTYPE, count=len(flat))

                # Held-out split is taken from the head of the stream and is
                # never written to the train shards, so the two cannot overlap.
                if val_w.total < val_tokens:
                    take = min(val_tokens - val_w.total, len(arr))
                    val_w.write(arr[:take])
                    arr = arr[take:]
                if len(arr):
                    remaining = budget - train_w.total
                    if remaining > 0:
                        train_w.write(arr[:remaining])

                got = train_w.total + val_w.total
                el = time.time() - t_start
                print(f"\r    {name}: {got/1e6:8.1f}M / {need/1e6:.0f}M tokens "
                      f"({100*got/need:5.1f}%)  {got/max(el,1e-9)/1e6:.2f}M tok/s",
                      end="", flush=True)
        finally:
            if local and os.path.exists(local) and not args.keep_raw:
                os.remove(local)

        done.add(fname)
        state.setdefault(name, {})["consumed"] = sorted(done)
        _save_state(args, state)

    train_w.close()
    val_w.close()
    print(f"\n    {name}: DONE train={train_w.total/1e6:.1f}M "
          f"val={val_w.total/1e6:.1f}M in {time.time()-t_start:.0f}s", flush=True)
    return dict(
        weight=spec["weight"], repo=spec["repo"], prefix=spec["prefix"],
        train_tokens=train_w.total, val_tokens=val_w.total,
        train_shards=train_w.shards, val_shards=val_w.shards,
    )


def _state_path(args):
    return os.path.join(args.out, "state.json")


def _save_state(args, state):
    with open(_state_path(args), "w") as f:
        json.dump(state, f, indent=2)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "corpus"))
    p.add_argument("--total-tokens", type=float, default=3e9)
    p.add_argument("--val-tokens", type=float, default=10e6,
                   help="held-out tokens PER DOMAIN")
    p.add_argument("--shard-tokens", type=int, default=256_000_000)
    p.add_argument("--batch-rows", type=int, default=2000)
    p.add_argument("--domains", default=",".join(DOMAINS))
    p.add_argument("--keep-raw", action="store_true")
    p.add_argument("--min-free-gb", type=float, default=40.0)
    args = p.parse_args()

    shard_dir = os.path.join(args.out, "shards")
    os.makedirs(shard_dir, exist_ok=True)

    free = shutil.disk_usage(args.out).free / 1e9
    est = args.total_tokens * 4 / 1e9
    print(f"disk free: {free:.0f} GB | est. output: {est:.1f} GB "
          f"(uint32, {args.total_tokens/1e9:.1f}B tokens)")
    if free < est + args.min_free_gb:
        sys.exit(f"ABORT: need ~{est + args.min_free_gb:.0f} GB free, have {free:.0f} GB")

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(TOKENIZER_ID)
    assert len(tok) < 2**32

    state = {}
    if os.path.exists(_state_path(args)):
        state = json.load(open(_state_path(args)))
        print(f"resuming: {sum(len(v.get('consumed', [])) for v in state.values())} "
              f"source files already consumed")

    selected = [d.strip() for d in args.domains.split(",") if d.strip()]
    manifest = {}
    mpath = os.path.join(args.out, "manifest.json")
    if os.path.exists(mpath):
        manifest = json.load(open(mpath)).get("domains", {})

    for name in selected:
        spec = DOMAINS[name]
        budget = int(args.total_tokens * spec["weight"])
        manifest[name] = tokenize_domain(
            name, spec, budget, int(args.val_tokens), shard_dir, args, tok, state)
        with open(mpath, "w") as f:
            json.dump(dict(
                tokenizer=TOKENIZER_ID, dtype="uint32", eod_token=EOD_TOKEN,
                seq_note="documents separated by <|endoftext|>",
                domains=manifest,
            ), f, indent=2)

    raw_dir = os.path.join(args.out, "_raw")
    if os.path.isdir(raw_dir) and not args.keep_raw:
        shutil.rmtree(raw_dir, ignore_errors=True)

    tot_tr = sum(d["train_tokens"] for d in manifest.values())
    tot_va = sum(d["val_tokens"] for d in manifest.values())
    print("\n" + "=" * 62)
    print(f"{'domain':<14}{'train (M)':>12}{'val (M)':>10}{'actual %':>10}")
    for n, d in manifest.items():
        print(f"{n:<14}{d['train_tokens']/1e6:>12.1f}{d['val_tokens']/1e6:>10.1f}"
              f"{100*d['train_tokens']/max(tot_tr,1):>9.1f}%")
    print(f"{'TOTAL':<14}{tot_tr/1e6:>12.1f}{tot_va/1e6:>10.1f}")
    print(f"manifest: {mpath}")


if __name__ == "__main__":
    main()

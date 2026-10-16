#!/usr/bin/env python3
"""Pin canonical benchmark sources and project question/prefix columns only.

These files are for corpus exclusion, never prompt or outcome selection. Parquet
answer columns are not read. LAMBADA's last whitespace-delimited target is removed.
"""
from __future__ import annotations
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from weekend_thinking.artifacts import atomic_json, sha256_file, utc_now


SOURCES = (
    ("openai/gsm8k", "740312add88f781978c0658806c59bc2815b9866", "gsm8k", "question",
     ("main/train-00000-of-00001.parquet", "main/test-00000-of-00001.parquet")),
    ("allenai/ai2_arc", "210d026faf9955653af8916fad021475a3f00453", "arc_challenge", "question",
     ("ARC-Challenge/train-00000-of-00001.parquet", "ARC-Challenge/validation-00000-of-00001.parquet",
      "ARC-Challenge/test-00000-of-00001.parquet")),
    ("EleutherAI/lambada_openai", "900124bf3b8235c6daf21033af9948b3f07346c4", "lambada_openai", "text",
     ("en/test/en.parquet",)),
)


def question_projection(value: str, *, lambada: bool = False) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Empty/nontext question.")
    if lambada:
        parts = value.rstrip().rsplit(None, 1)
        if len(parts) != 2:
            raise ValueError("LAMBADA example has no prefix before its target.")
        return parts[0]
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--download-dir", type=Path, required=True)
    args = parser.parse_args()
    manifest_path = args.out.with_suffix(".manifest.json")
    if args.out.exists() or manifest_path.exists():
        raise SystemExit("Refusing to overwrite exclusion questions or manifest.")
    from huggingface_hub import hf_hub_download
    import pyarrow.parquet as pq

    args.out.parent.mkdir(parents=True, exist_ok=True)
    temp = args.out.with_name("." + args.out.name + ".incomplete")
    sources, counts = [], Counter()
    try:
        with temp.open("x") as dst:
            for repo, revision, task, column, files in SOURCES:
                source = {"dataset_id": repo, "revision": revision, "projected_column": column, "files": []}
                local_dir = args.download_dir / repo.replace("/", "--")
                for filename in files:
                    path = Path(hf_hub_download(repo, filename, repo_type="dataset", revision=revision,
                                               local_dir=local_dir))
                    # Projection happens in the Parquet reader, before Python rows.
                    values = pq.read_table(path, columns=[column])[column].to_pylist()
                    split = Path(filename).stem.split("-", 1)[0] if task != "lambada_openai" else "test"
                    for index, value in enumerate(values):
                        question = question_projection(value, lambada=task == "lambada_openai")
                        row = {"item_id": f"{task}:{split}:{index:05d}", "task": task,
                               "split": split, "question": question}
                        dst.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
                        counts[f"{task}:{split}"] += 1
                    source["files"].append({"path": filename, "sha256": sha256_file(path), "rows": len(values)})
                sources.append(source)
            dst.flush()
            os.fsync(dst.fileno())
        os.link(temp, args.out)
    finally:
        temp.unlink(missing_ok=True)
    manifest = {"created_utc": utc_now(), "purpose": "corpus_exclusion_only",
                "sources": sources, "counts": dict(counts), "sha256": sha256_file(args.out),
                "answer_columns_read": False, "lambada_final_target_removed": True,
                "limitations": ["Question-only screening does not establish absence of paraphrases.",
                                "No assertion about contamination in pretrained weights."]}
    atomic_json(manifest_path, manifest)
    print(json.dumps({"out": str(args.out), "counts": dict(counts), "sha256": manifest["sha256"]}, indent=2))


if __name__ == "__main__":
    main()

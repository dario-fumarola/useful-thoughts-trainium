#!/usr/bin/env python3
"""Compare two complete JSONL evaluations; no silent item intersection."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from weekend_thinking.stats import paired_cluster_bootstrap

p = argparse.ArgumentParser()
p.add_argument("a", type=Path); p.add_argument("b", type=Path)
p.add_argument("--out", type=Path, required=True)
p.add_argument("--repeats", type=int, default=5000)
a = p.parse_args()
read = lambda path: [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
result = paired_cluster_bootstrap(read(a.a), read(a.b), repeats=a.repeats)
result["comparison_files"] = {"a": str(a.a), "b": str(a.b)}
a.out.parent.mkdir(parents=True, exist_ok=True)
a.out.write_text(json.dumps(result, indent=2)+"\n")
print(json.dumps(result, indent=2))

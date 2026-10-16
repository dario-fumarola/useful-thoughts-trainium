#!/usr/bin/env python3
"""Reads one JSON object containing step_seconds and budget dimensions."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from weekend_thinking.qualification import project
p = argparse.ArgumentParser(); p.add_argument("measurements", type=Path)
p.add_argument("--out", type=Path, required=True); a = p.parse_args()
result = project(**json.loads(a.measurements.read_text()))
a.out.parent.mkdir(parents=True, exist_ok=True)
a.out.write_text(json.dumps(result, indent=2)+"\n")
print(json.dumps(result, indent=2))

#!/usr/bin/env python3
"""Score pre-existing outputs; never generates using gold answers."""
from pathlib import Path
import argparse, json, sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from weekend_thinking.evaluation import score_predictions
from weekend_thinking.artifacts import atomic_bytes, atomic_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--gold',type=Path,required=True)
    p.add_argument('--predictions',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    read=lambda path:[json.loads(x) for x in path.read_text().splitlines() if x.strip()]
    rows,summary=score_predictions(read(a.gold),read(a.predictions))
    atomic_bytes(a.out, ''.join(json.dumps(x)+'\n' for x in rows).encode())
    atomic_json(a.out.with_suffix('.summary.json'), summary)
    print(json.dumps(summary, indent=2))
if __name__=='__main__':main()

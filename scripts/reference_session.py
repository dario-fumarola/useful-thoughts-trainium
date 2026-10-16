#!/usr/bin/env python3
"""Real toy CPU updates + checkpointing; deliberately NOT large-model training."""
from pathlib import Path
import argparse, json, sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from weekend_thinking.session import run_reference


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--arm', choices=['thought','vanilla'], required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--steps', type=int, default=6)
    p.add_argument('--seed', type=int, default=1234)
    p.add_argument('--stop-after', type=int)
    p.add_argument('--resume', type=Path)
    a=p.parse_args()
    result=run_reference(**vars(a))
    print(json.dumps({k:v for k,v in result.items() if k!='metrics'}, indent=2))

if __name__=='__main__': main()

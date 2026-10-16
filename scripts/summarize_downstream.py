#!/usr/bin/env python3
from pathlib import Path
import argparse, json, sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from weekend_thinking.evaluation import downstream_contrasts
from weekend_thinking.artifacts import atomic_json


def main():
    p=argparse.ArgumentParser(description='All predeclared downstream contrasts, per training seed.')
    for flag in ('c-direct','c-thought','r-direct','r-thought'):
        p.add_argument('--'+flag,type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--repeats',type=int,default=5000)
    a=p.parse_args()
    read=lambda path:[json.loads(x) for x in path.read_text().splitlines() if x.strip()]
    result=downstream_contrasts(*[read(getattr(a,x)) for x in ('c_direct','c_thought','r_direct','r_thought')],repeats=a.repeats)
    atomic_json(a.out,result);print(json.dumps(result,indent=2))
if __name__=='__main__':main()

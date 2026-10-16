#!/usr/bin/env python3
from pathlib import Path
import argparse,json,sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from weekend_thinking.experiment import freeze
from weekend_thinking.artifacts import atomic_json


def main():
    p=argparse.ArgumentParser(description='Freeze actual assets and recipe. Does not launch or certify hardware.')
    p.add_argument('--proposal',type=Path,default=Path('configs/weekend.json'))
    p.add_argument('--site',type=Path,default=Path('configs/site.local.json'))
    p.add_argument('--experiment-id',required=True)
    p.add_argument('--steps',type=int,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();root=Path(__file__).resolve().parents[1]
    try:
        result=freeze(root,json.loads(a.proposal.read_text()),json.loads(a.site.read_text()),experiment_id=a.experiment_id,steps=a.steps)
        atomic_json(a.out,result)
    except (ValueError,FileNotFoundError,KeyError) as e:
        raise SystemExit(f'Not frozen: {e}')
    print(json.dumps({'experiment_id':result['experiment_id'],'spec_hash':result['spec_hash'],'status':result['status']},indent=2))
if __name__=='__main__':main()

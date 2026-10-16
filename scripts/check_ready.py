#!/usr/bin/env python3
from pathlib import Path
import argparse,json,sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from weekend_thinking.experiment import validate_qualification


def main():
    p=argparse.ArgumentParser(description='Check evidence bindings before any long launch; does not launch.')
    p.add_argument('--frozen',type=Path,required=True)
    p.add_argument('--receipt',type=Path,required=True)
    a=p.parse_args()
    try:validate_qualification(Path(__file__).resolve().parents[1],json.loads(a.frozen.read_text()),json.loads(a.receipt.read_text()))
    except (ValueError,FileNotFoundError,KeyError) as e:raise SystemExit(f'NOT READY: {e}')
    print('Recorded Neuron evidence matches this source/spec. Review the actual results and deadline before launching.')
if __name__=='__main__':main()

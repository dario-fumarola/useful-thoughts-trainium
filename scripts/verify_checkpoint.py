#!/usr/bin/env python3
from pathlib import Path
import argparse,json,sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from weekend_thinking.checkpoints import verify_checkpoint

p=argparse.ArgumentParser(description='Verify complete checkpoint files and hashes, not numerical correctness.')
p.add_argument('directory',type=Path);p.add_argument('--spec-hash')
a=p.parse_args()
print(json.dumps(verify_checkpoint(a.directory,expected_spec_hash=a.spec_hash),indent=2))

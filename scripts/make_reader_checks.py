#!/usr/bin/env python3
"""Generate prefix-derived diagnostic hints, NOT main training/evaluation data.

Helpful notes contain partial products, not the final target answer. Incorrect
notes change one partial product. Effects must be measured; no model result is
asserted by generating these cases.
"""
import argparse
import json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args()
rows=[]
for x in (17,23,36,48):
    for y in (24,37,56,68):
        tens,units=divmod(y,10)
        rows.append({'item_id':f'reader-diagnostic-{x}-{y}','split':'diagnostic_not_test',
          'prefix':f'There are {x} trays, each containing {y} items. The total number of items is',
          'future':f' {x*y}',
          'helpful_note':f'Split {y} into {10*tens} and {units}. The two partial products are {x*10*tens} and {x*units}; add them.',
          'incorrect_note':f'Split {y} into {10*tens} and {units}. The two partial products are {x*10*tens} and {x*units+1}; add them.',
          'qualification':'Everything needed is in the prefix. Token-length matching still requires the actual tokenizer. Never use these cases as the final generalization test.'})
a.out.parent.mkdir(parents=True,exist_ok=True)
a.out.write_text(''.join(json.dumps(r)+'\n' for r in rows))
print(f'Wrote {len(rows)} reader diagnostics; no model was evaluated.')

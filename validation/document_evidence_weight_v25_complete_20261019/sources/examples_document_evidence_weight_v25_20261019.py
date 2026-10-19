"""Reuse the eight original hash-selected questions; no outcome-based examples."""
from pathlib import Path
import hashlib
import json
R=Path('/home/ubuntu/dario');C=R/'local/diagnostics/document-evidence-weight-v25-ablation-v1'
read=lambda p:json.loads(p.read_text())
old_file=R/'validation/document_evidence_plan_v23_pilot_complete_20261019/fixed-hash-eight-examples.json'
old=read(old_file);current=read(C/'held-native-and-replay-scored.json')
assert read(C/'held-independent-check.json')['status'].startswith('PASS_')
selected=old['selection'];assert len(set(selected['item_ids']))==8
rows=[]
for key in selected['item_ids']:
 source=next(v for v in old['examples'] if v['question']['item_id']==key)
 record=dict(original_example=source,item_id=key,
     weight1={mode:values[key] for mode,values in current.items() if mode.startswith(('native:','answer-replay:'))},
     original_baselines={mode:values[key] for mode,values in current.items() if mode.startswith('old-')})
 assert all(v['native']['item_id']==key for v in record['weight1'].values())
 rows.append(record)
result=dict(selection=selected,original_example_file_sha256=hashlib.sha256(old_file.read_bytes()).hexdigest(),
 interpretation='Same original eight question identities, selected before V23 outputs. Previously inspected, descriptive only; no success/failure selection.',examples=rows)
with (C/'fixed-eight-examples.json').open('x') as f:json.dump(result,f,indent=2)
print('Saved the same eight fixed questions with every V25 condition and original baselines.')

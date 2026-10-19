"""Audit every actual native TRAIN row in both new source-balanced views."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib
import json
from transformers import AutoTokenizer
from train.partial_state_driver import ROOT,source_files
from train.real_source_balanced_packing import training_pack,ARMS

R=Path('/home/ubuntu/dario');W=R/'local/worktrees/real-source-balanced-v17';assert ROOT==W
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
data=R/'local/nvme-study/data/real-prefix-partial-labels-v9/TRAIN.jsonl'
assert sha(data)=='437c64bc9ce4b451f4d5da07fccca9e108ee9add75b3c955bc54d2a2eca6b915'
rows=[json.loads(v) for v in data.read_text().splitlines()]
assert len(rows)==len({v['document_id'] for v in rows})==5120
tok=AutoTokenizer.from_pretrained(R/'local/assets/qwen3-1.7b-base',local_files_only=True)
counts={arm:dict(documents=0,source_labels_one_pass=0,task_labels_one_pass=0,
                maximum_source_targets=0,maximum_task_targets=0,maximum_total_targets=0) for arm in ARMS}
for row in rows:
    views={}
    for arm in ARMS:
        pack=training_pack(row,tok,arm);views[arm]=pack
        source,task=pack['note_close_mask'],pack['future_mask']
        assert not (source&task).any() and pack['labels'][task].tolist()[-16:]==row['continuation16_ids']
        assert pack['labels'][source].tolist()==row['prefix_ids'][1:]
        assert pack['target_positions'][source].tolist()==list(range(1,len(row['prefix_ids'])))
        assert 151643 not in pack['labels'][source|task].tolist()
        expected_task=16 if arm=='source_direct' else len(row['full_calculation_ids'])+2+16
        assert int(task.sum())==expected_task
        n,m=int(source.sum()),int(task.sum());c=counts[arm];c['documents']+=1
        c['source_labels_one_pass']+=n;c['task_labels_one_pass']+=m
        for key,value in [('maximum_source_targets',n),('maximum_task_targets',m),('maximum_total_targets',n+m)]:c[key]=max(c[key],value)
    a,b=[views[arm] for arm in ARMS]
    assert a['labels'][a['note_close_mask']].tolist()==b['labels'][b['note_close_mask']].tolist()
for c in counts.values():
    c['source_labels_two_passes']=2*c['source_labels_one_pass']
    c['task_labels_two_passes']=2*c['task_labels_one_pass']
    c['total_active_labels_two_passes']=c['source_labels_two_passes']+c['task_labels_two_passes']
assert counts['source_direct']['task_labels_two_passes']==163840
assert counts['source_cot']['task_labels_two_passes']==512900
assert counts['source_direct']['source_labels_two_passes']==counts['source_cot']['source_labels_two_passes']
report=dict(status='ALL_10240_NATIVE_TRAIN_VIEWS_CAUSAL_MASKS_AND_LABEL_COUNTS_PASS',
    utc=datetime.now(timezone.utc).isoformat(),source_files=source_files(),source_root=str(W),
    data_sha256=sha(data),audit_source_sha256=sha(__file__),counts=counts,
    loss='Per-example source mean and per-example task mean each receive0.5;then average over the fixed globalbatch. No coefficient claim about gradient norms.',
    limits='Data/CPU check only. No hardware training, checkpoint qualification, held-model result or learning release.')
out=R/'local/analysis/real-source-balanced-v17-native-pack-audit-20261019.json'
with out.open('x') as stream:json.dump(report,stream,indent=2)
print(json.dumps(report,indent=2),flush=True)

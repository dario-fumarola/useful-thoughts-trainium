from pathlib import Path
from types import SimpleNamespace
from datetime import datetime,timezone
import json,hashlib,torch
from train.evaluate_local import setup_model
R=Path('/home/ubuntu/dario');C=R/'local/diagnostics/document-evidence-plan-v23-decode-qualification-v1';p=json.loads((C/'protocol.json').read_text());q=json.loads((C/'prompts-direct.jsonl').read_text().splitlines()[0]);torch.set_num_threads(16)
backend,tok,_,signature,_=setup_model(SimpleNamespace(model_dir=Path(p['model_dir']),checkpoint=None,device='cpu'));assert signature==p['signature']
x=torch.full((1,3584),151643,dtype=torch.long);x[0,:len(q['prompt_ids'])]=torch.tensor(q['prompt_ids'])
with torch.no_grad():
 h=backend.hidden(x);lp=backend.model.lm_head(h[:,len(q['prompt_ids'])-1]).float().log_softmax(-1)
assert lp.shape==(1,151936) and torch.isfinite(lp).all();torch.save(lp,C/'long-context-CPU-logps.pt')
r=dict(status='COMPLETE_TRAIN_ONLY_LONG_CONTEXT_CPU_FP32_REFERENCE',utc=datetime.now(timezone.utc).isoformat(),item_id=q['item_id'],prompt_tokens=len(q['prompt_ids']),width=3584,signature=signature,reference_sha256=hashlib.sha256((C/'long-context-CPU-logps.pt').read_bytes()).hexdigest())
(C/'long-context-CPU-reference.json').write_text(json.dumps(r,indent=2));print(json.dumps(r,indent=2),flush=True)

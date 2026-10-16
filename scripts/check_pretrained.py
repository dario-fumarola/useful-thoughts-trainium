#!/usr/bin/env python3
"""Optional host-side sanity check. Requires Transformers and downloaded weights."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from weekend_thinking.pretrained import load_base,sampled_fingerprint
p=argparse.ArgumentParser();p.add_argument('--model-dir',required=True);p.add_argument('--out',type=Path,required=True)
a=p.parse_args();torch.set_num_threads(min(8,torch.get_num_threads()))
model,tok=load_base(a.model_dir);model.eval()
x=tok('A triangle has three sides. Two plus two equals',return_tensors='pt')['input_ids']
with torch.no_grad():
    logits=model(x,use_cache=False).logits.float()
    loss=torch.nn.functional.cross_entropy(logits[:,:-1].reshape(-1,logits.shape[-1]),x[:,1:].reshape(-1))
    assert torch.isfinite(logits).all()
report=sampled_fingerprint(model)
report.update({'model_type':model.config.model_type,'host_sanity_loss':float(loss),
               'first_next_token':tok.decode([int(logits[0,-1].argmax())]),
               'claim':'Load/logits integrity only, not a research efficacy test or Trainium qualification.'})
a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))

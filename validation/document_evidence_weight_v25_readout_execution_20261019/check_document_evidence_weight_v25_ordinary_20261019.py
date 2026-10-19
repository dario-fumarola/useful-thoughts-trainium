"""Independent page-NLL aggregation from current and immutable baseline token scores."""
from pathlib import Path
import hashlib
import json
import math
import numpy as np
R=Path('/home/ubuntu/dario');C=R/'local/diagnostics/document-evidence-weight-v25-ablation-v1'
O=R/'local/nvme-cpt/evaluation/document-evidence-weight-v25-ordinary256-v1'
read=lambda p:json.loads(p.read_text())
rows=lambda p:[json.loads(v) for v in p.read_text().splitlines()]
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as stream:
  for b in iter(lambda:stream.read(8*2**20),b''):h.update(b)
 return h.hexdigest()
def values(output):
 result={}
 for shard in range(4):
  d=output/'scores'/f'shard-{shard}';report=read(d/'completion.json')
  assert report['scores_sha256']==sha(d/'scores.jsonl')
  for v in rows(d/'scores.jsonl'):
   if v['panel']!='ordinary':continue
   assert v['item_id'] not in result and v['status']=='ok'
   assert v['tokens']==len(v['token_logps'])==len(v['target_ids'])==32
   assert all(math.isfinite(x) for x in v['token_logps'])
   assert abs(sum(v['token_logps'])-v['full_logp'])<=1e-8
   result[v['item_id']]=v
 assert len(result)==256
 return result
p=read(O/'protocol.json');r=read(C/'ORDINARY/result.json')
assert r['protocol_sha256']==sha(O/'protocol.json')
assert read(C/'ORDINARY/GEN/closure.json')['assigned_physical_device_empty']
for shard in range(4):
 d=O/'scores'/f'shard-{shard}';run=read(d/'completion.json')
 assert run['rows']==64 and run['execution_errors']==run['synthetic_execution_rows']==0
 calibration=read(d/'own-model-TRAIN-calibration.json')
 assert set(calibration)=={'ordinary'} and calibration['ordinary']['passed']
 assert calibration['ordinary']['max_token_error']<=.001 and calibration['ordinary']['absolute_mean_span_error']<=.001
old=R/'local/diagnostics/document-evidence-plan-v23-pilot-v1'
paths={v['arm']:Path(v['output']) for v in read(old/'likelihood-registry-v2.json')['groups']}
ref=read(old/'readout-execution-plan-v2.json')['B0_ordinary_reference'];paths['B0']=Path(ref['output'])
assert all(sha(paths['B0']/name)==value for name,value in ref['file_sha256'].items())
current=values(O);ids=sorted(current)
expected_input=sha(O/'ordinary-raw.jsonl')
assert all(sha(path/'ordinary-raw.jsonl')==expected_input for path in paths.values())
mean=np.mean([-sum(current[k]['token_logps'])/32 for k in ids]);assert abs(r['tables']['weight1']['mean_nll_per_token']-mean)<=1e-12
for index,(name,path) in enumerate(sorted(paths.items())):
 base=values(path);assert set(base)==set(current)
 assert all(base[k]['target_ids']==current[k]['target_ids'] for k in ids)
 differences=np.array([(sum(base[k]['token_logps'])-sum(current[k]['token_logps']))/32 for k in ids])
 rng=np.random.default_rng(2026101925+100+index);means=differences[rng.integers(0,256,(20000,256))].mean(1)
 actual=r['paired'][name]
 assert abs(actual['mean_nll_change']-differences.mean())<=1e-12
 assert np.max(np.abs(np.quantile(means,[.025,.975])-actual['CI95']))<=1e-12
 assert abs(r['tables'][name]['mean_nll_per_token']-np.mean([-sum(base[k]['token_logps'])/32 for k in ids]))<=1e-12
result=dict(status='PASS_INDEPENDENT_ORDINARY256_TOKEN_SUMS_BASELINES_AND_PAIRED_PAGES',
 result_sha256=sha(C/'ORDINARY/result.json'),checker_sha256=sha(__file__),items=256,target_tokens=8192,
 actual_own_model_calibrations=4,comparisons=3,limits='Previously inspected ordinary panel, conditional checkpoint intervals, not a general-pretraining claim.')
with (C/'ORDINARY/independent-check.json').open('x') as stream:json.dump(result,stream,indent=2)
print(json.dumps(result))

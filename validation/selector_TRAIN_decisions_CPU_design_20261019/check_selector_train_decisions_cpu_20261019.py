"""Check native classification/token boundaries and four fixed HF forward spots."""
from pathlib import Path
from collections import Counter
from types import SimpleNamespace
import hashlib,inspect,json,math,re
import torch
from transformers import AutoTokenizer
from train.evaluate_local import setup_model

R=Path('/home/ubuntu/dario');C=R/'local/diagnostics/selector-TRAIN-decisions-CPU-v1'
T=R/'local/diagnostics/document-evidence-weight-v25-ablation-v1/TRAIN-fit'
OLD=R/'local/nvme-cpt/evaluation/document-evidence-weight-v25-TRAIN256-v1'
read=lambda p:json.loads(p.read_text())
rows=lambda p:[json.loads(l) for l in p.read_text().splitlines()]
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*2**20),b''):h.update(b)
    return h.hexdigest()
def classify(v,gold):
    text=v['native']['raw_text'];heads=list(re.finditer(r'(?im)^[ \t]*Final answer:[ \t]*',text))
    valid=len(heads)==1;ids=[]
    if valid:
        part=text[:heads[0].start()].strip()
        match=re.fullmatch(r'Evidence:\s*((?:\[[0-9]+\]\s*)*)',part,re.I)
        valid=match is not None
        if valid:ids=[int(x) for x in re.findall(r'\[([0-9]+)\]',match[1])]
    valid=valid and bool(ids) and len(ids)==len(set(ids)) and all(1<=x<=20 for x in ids)
    if not valid:return 'unsupported_selection'
    if set(ids)==set(gold):return 'exact_annotated_set'
    if len(ids)<len(gold) and ids==gold[:len(ids)]:return 'canonical_prefix_then_early_closure'
    if set(ids).issubset(gold):return 'annotated_subset_with_earlier_omission'
    return 'contains_nonannotated_ID'

design=read(C/'design.json');result=read(C/'result.json');contract=read(C/'check-contract.json')
assert read(C/'exit.json')['exit_code']==0 and result['items']==256
assert sha(C/'design.json')==result['design_sha256'] and sha(C/'observations.jsonl')==result['observations_sha256']
assert sha(C/'inputs.jsonl')==design['input_sha256'] and all(sha(p)==h for p,h in design['inputs'].items())
assert contract['checker_sha256']==sha(__file__) and contract['absolute_logp_tolerance']==1e-4
inputs=rows(C/'inputs.jsonl');observed=rows(C/'observations.jsonl')
labels={v['item_id']:v for v in rows(T/'TRAIN256-labels.jsonl')};native={v['native']['item_id']:v for v in rows(T/'native-and-scored.jsonl')}
prompts={v['item_id']:v['prompt_ids'] for v in rows(OLD/'prompts-selection.jsonl')}
p=read(OLD/'protocol.json');assert sha(OLD/'prompts-selection.jsonl')==p['files']['prompts-selection.jsonl']
assert [v['item_id'] for v in inputs]==[v['item_id'] for v in observed] and len({v['item_id'] for v in inputs})==256
tok=AutoTokenizer.from_pretrained(design['model_dir'],local_files_only=True,trust_remote_code=False)
byid={};branch_positions={};surface_pairs=Counter()
for item,value in zip(inputs,observed):
    key=item['item_id'];gold=labels[key]['support_indices'];prefix=prompts[key]
    assert len(gold)==item['support_count']==value['support_count']
    assert item['ids'][:len(prefix)]==prefix and item['prefix_tokens']==len(prefix)
    text=tok.decode(prefix,skip_special_tokens=False,clean_up_tokenization_spaces=False)
    full=tok.encode(text+'Evidence:'+''.join(f' [{x}]' for x in gold)+'\n',add_special_tokens=False)
    assert full==item['ids'] and value['native_category']==item['native_category']==classify(native[key],gold)
    assert len(value['branches'])==len(gold)-1;positions=[]
    for index,(spec,b) in enumerate(zip(item['branches'],value['branches']),1):
        closed=tok.encode(text+'Evidence:'+''.join(f' [{x}]' for x in gold[:index])+'\n',add_special_tokens=False)
        shared=next(i for i,(a,z) in enumerate(zip(full,closed)) if a!=z)
        assert shared==b['shared_prefix_tokens']==spec['shared_prefix_tokens'] and b['after_ID_count']==index
        assert b['required_token']==full[shared] and b['closure_token']==closed[shared]
        assert b['required_surface']==tok.decode([full[shared]]) and b['closure_surface']==tok.decode([closed[shared]])
        assert all(math.isfinite(b[k]) and b[k]<=1e-5 for k in ('required_logp','closure_logp'))
        assert abs(b['closure_logp']-b['required_logp']-b['closure_minus_required_logp'])<1e-12
        assert b['closure_preferred']==(b['closure_logp']>b['required_logp'])
        positions.append(shared-1);surface_pairs[(b['required_surface'],b['closure_surface'])]+=1
    byid[key]=value;branch_positions[key]=positions
for n in (2,3,4):
    sub=[v for v in observed if v['support_count']==n];branch=[b for v in sub for b in v['branches']]
    expected=dict(items=len(sub),native_categories=dict(Counter(v['native_category'] for v in sub)),required_continuation_decisions=len(branch),closure_preferred_decisions=sum(v['closure_preferred'] for v in branch),questions_with_any_closure_preferred=sum(any(b['closure_preferred'] for b in v['branches']) for v in sub))
    assert result['strata'][str(n)]==expected
assert read(C/'actual-CPU-model-check.json')['passed']
torch.set_num_threads(16);m=design['model']
backend,_,loaded,signature,_=setup_model(SimpleNamespace(model_dir=Path(design['model_dir']),checkpoint=Path(m['checkpoint']),device='cpu'))
assert signature==design['signature'] and all(loaded[k]==m[k] for k in ('run_id','step','spec_hash'))
assert 'logits_to_keep' in inspect.signature(backend.model.forward).parameters
spots=[]
with torch.no_grad():
    for key in contract['fixed_item_ids']:
        item=next(v for v in inputs if v['item_id']==key);pos=branch_positions[key]
        # A separate HF CausalLM forward, rather than the producer's hidden/head path.
        output=backend.model(input_ids=torch.tensor([item['ids']]),use_cache=False,logits_to_keep=torch.tensor(pos))
        lp=output.logits[0].float().log_softmax(-1);errors=[]
        for i,b in enumerate(byid[key]['branches']):
            errors.extend([abs(float(lp[i,b['required_token']])-b['required_logp']),abs(float(lp[i,b['closure_token']])-b['closure_logp'])])
        assert max(errors)<=contract['absolute_logp_tolerance'];spots.append(dict(item_id=key,max_logp_error=max(errors)))
        del output,lp
report=dict(status='PASS_INDEPENDENT_TRAIN_CATEGORIES_TOKEN_BOUNDARIES_COUNTS_AND_FIXED_HF_FORWARD_SPOTS',items=256,continuation_decisions=sum(surface_pairs.values()),HF_forward_spots=spots,
    result_sha256=sha(C/'result.json'),source_sha256=sha(__file__),contract_sha256=sha(C/'check-contract.json'),
    token_surface_pairs=[dict(required=a,closure=b,count=n) for (a,b),n in surface_pairs.items()],
    limit='Checks original observations and four fixed numerical spots, not every forward independently. First-divergent-token scores are not total field-termination probabilities or semantic/capability validation.')
with (C/'independent-check.json').open('x') as f:json.dump(report,f,indent=2)
print(json.dumps(report),flush=True)

"""Post-hoc accounting of fixed scalar errors;no changed endpoint or new forwards."""
from collections import Counter
from pathlib import Path
import gzip
import hashlib
import json

R=Path('/home/ubuntu/dario');p=R/'validation/real_source_balanced_v17_complete_20261019.json'
r=json.loads(p.read_text());out=p.with_suffix('')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert json.loads((out/'independent-readout-check.json').read_text())['report_sha256']==sha(p)
def load(model,mode):
    f=out/f'{model}-all2560-scalar-continuations.jsonl.gz'
    assert sha(f)==r['payloads'][str(f.relative_to(R))]['sha256']
    values=[json.loads(v) for v in gzip.decompress(f.read_bytes()).splitlines()]
    selected={v['item_id']:v for v in values if v['mode']==mode};assert len(selected)==512
    return selected
def state(v):return 'correct' if v['correct'] else 'valid_wrong' if v['category']=='valid' else v['category']
tables={}
for seed in r['seeds']:
    cot=f'source_cot-seed{seed}-step320';direct=f'source_direct-seed{seed}-step320'
    pairs={}
    for name,first,firstmode,last,lastmode in [
        ('cot_own_vs_direct_raw',direct,'raw',cot,'own'),
        ('cot_own_vs_cot_empty',cot,'empty',cot,'own'),
        ('cot_own_vs_B0raw','B0','raw',cot,'own')]:
        a,b=load(first,firstmode),load(last,lastmode);assert a.keys()==b.keys()
        counts=Counter((state(a[k]),state(b[k])) for k in a)
        bothvalid=[k for k in a if a[k]['category']==b[k]['category']=='valid']
        numeric_gain=sum(int(b[k]['correct'])-int(a[k]['correct']) for k in bothvalid)
        total_gain=sum(int(b[k]['correct'])-int(a[k]['correct']) for k in a)
        assert sum(counts.values())==512
        pairs[name]=dict(reference_model=first,reference_mode=firstmode,treatment_model=last,treatment_mode=lastmode,
            documents=512,transitions=[dict(reference=x,treatment=y,documents=n) for (x,y),n in sorted(counts.items())],
            both_parser_valid_documents=len(bothvalid),net_correct_gain_on_both_valid=numeric_gain,
            remaining_net_correct_gain=total_gain-numeric_gain,total_net_correct_gain=total_gain,
            limits='Fixed 512-item denominator is unchanged. Valid-pair accounting describes transitions;it is not a replacement conditional-accuracy endpoint.')
    tables[seed]=pairs
report=dict(status='POST_HOC_FIXED_SCALAR_ERROR_TRANSITIONS_NO_NEW_GATE',source_report_sha256=sha(p),
    source_sha256=sha(__file__),seeds=tables,
    limits='All inputs are original independently checked attempts. No new forwards, filtering, parser adjustment or endpoint change.')
f=out/'descriptive-scalar-error-transitions.json'
with f.open('x') as stream:json.dump(report,stream,indent=2)
for seed,values in tables.items():
    print(seed,{k:dict(net=v['total_net_correct_gain'],both_valid_net=v['net_correct_gain_on_both_valid'],remaining=v['remaining_net_correct_gain']) for k,v in values.items()})

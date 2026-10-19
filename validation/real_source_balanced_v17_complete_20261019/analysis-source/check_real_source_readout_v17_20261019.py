"""Independent native-text,exact-value,count and vectorized-bootstrap check."""
from collections import Counter
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path
import ast
import gzip
import hashlib
import json
import math
import re
import numpy as np
from tokenizers import Tokenizer

R=Path('/home/ubuntu/dario')
path=R/'validation/real_source_balanced_v17_complete_20261019.json'
report=json.loads(path.read_text())
assert report['status']=='COMPLETE_ELEVEN_MODEL_TWO_SEED_SOURCE_BALANCED_REFERENCE'
directory=path.with_suffix('')
tokenizer=Tokenizer.from_file(str(R/'local/assets/qwen3-1.7b-base/tokenizer.json'))
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()


def payload(name):
    file=directory/name
    relative=str(file.relative_to(R))
    assert sha(file)==report['payloads'][relative]['sha256']
    values=[json.loads(line) for line in gzip.decompress(file.read_bytes()).splitlines()]
    assert len(values)==report['payloads'][relative]['rows']
    return values


metadata=payload('fixed-source-target-metadata.jsonl.gz')
assert len(metadata)==len({v['document_id'] for v in metadata})==512
annotations={v['item_id']:v for v in metadata}


def exact(expression):
    tree=ast.parse(expression,mode='eval')
    def walk(node):
        if isinstance(node,ast.Constant):
            return Fraction(ast.get_source_segment(expression,node))
        if isinstance(node,ast.UnaryOp):
            value=walk(node.operand)
            assert isinstance(node.op,(ast.UAdd,ast.USub))
            return value if isinstance(node.op,ast.UAdd) else -value
        assert isinstance(node,ast.BinOp)
        a,b=walk(node.left),walk(node.right)
        if isinstance(node.op,ast.Add):return a+b
        if isinstance(node.op,ast.Sub):return a-b
        if isinstance(node.op,ast.Mult):return a*b
        if isinstance(node.op,ast.Div):return a/b
        assert isinstance(node.op,ast.Pow) and b.denominator==1
        return a**b.numerator
    return walk(tree.body)


for value in metadata:
    assert exact(value['normalized_expression'])==Fraction(value['expected_numeric_value'])
    ranges=value['component_ranges']
    flattened=[i for start,end in ranges.values() for i in range(start,end)]
    assert flattened==list(range(16))

NUMBER=r'[+-]?(?:[0-9]+(?:\.[0-9]+)?|\.[0-9]+)'
leading=re.compile(rf'({NUMBER})(?:[ \t]*/[ \t]*({NUMBER}))?')
scalar,LP,notes={},{},{}
scalar_rows=note_rows=LP_rows=0
for model,tables in report['tables'].items():
    values=payload(f'{model}-all2560-scalar-continuations.jsonl.gz')
    assert len(values)==len({(v['item_id'],v['mode']) for v in values})==2560
    scalar[model]={}
    for value in values:
        actions,content=value['actions'],value['content_ids']
        assert value['n_actions']==len(actions)<=32
        stopped=bool(actions and actions[-1]==151643)
        assert 151643 not in actions[:-1]
        assert content==(actions[:-1] if stopped else actions)
        assert value['text']==tokenizer.decode(content,skip_special_tokens=False)
        if value['status']=='ok':
            assert value['stop_type']==('token:151643' if stopped else 'length_cap')
            assert stopped or len(actions)==32
        expected=Fraction(annotations[value['item_id']]['expected_numeric_value'])
        assert Fraction(value['expected_numeric_value'])==expected
        parsed=value['parsed']
        valid=parsed['status']=='valid'
        actual=None
        if valid:
            match=leading.match(value['text'].lstrip(' \t\r\n'))
            assert match and match.group()==parsed['literal']
            actual=Fraction(match[1])/(Fraction(match[2]) if match[2] is not None else 1)
            assert actual==Fraction(parsed['value'])
        error=value['status']!='ok' or value['source_note_status']!='ok'
        assert value['category']==('execution_error' if error else parsed['status'])
        assert value['correct']==(not error and valid and actual==expected)
    for mode,table in tables.items():
        selected=[v for v in values if v['mode']==mode]
        assert len(selected)==len({v['document_id'] for v in selected})==512
        assert {v['item_id'] for v in selected}==set(annotations)
        assert table['correct']==sum(v['correct'] for v in selected)
        assert table['categories']==dict(Counter(v['category'] for v in selected))
        assert table['capped']==sum(v['stop_type']=='length_cap' for v in selected)
        assert table['generated_actions']==sum(v['n_actions'] for v in selected)
        scalar[model][mode]={v['item_id']:v for v in selected}
    scalar_rows+=len(values)
    values=payload(f'{model}-all512-original-notes.jsonl.gz')
    assert len(values)==len({v['document_id'] for v in values})==512
    for value in values:
        actions=value['actions']
        stopped=bool(actions and actions[-1] in (151643,16141))
        assert value['n_actions']==len(actions)<=128
        assert not any(v in (151643,16141) for v in actions[:-1])
        assert value['content_ids']==(actions[:-1] if stopped else actions)
        assert value['text']==tokenizer.decode(value['content_ids'],skip_special_tokens=False)
        if value['status']=='ok':
            assert value['stop_type']==(f'token:{actions[-1]}' if stopped else 'length_cap')
            assert stopped or len(actions)==128
    notes[model]=values
    note_rows+=len(values)
    costs=report['note_costs'][model]
    assert costs['generated_actions']==sum(v['n_actions'] for v in values)
    assert costs['capped']==sum(v['stop_type']=='length_cap' for v in values)
    assert costs['empty']==sum(not v['content_ids'] for v in values)
    assert costs['errors']==sum(v['status']!='ok' for v in values)
    LP[model]={}
    for panel,n in [('DEV_confirm512',2560),('DEV_ordinary256',256)]:
        values=payload(f'{model}-{panel}-all{n}-logps.jsonl.gz')
        assert len(values)==len({(v['item_id'],v['mode']) for v in values})==n
        LP_rows+=len(values)
        LP[model][panel]={}
        for mode,cell in report['absolute_likelihood'][model][panel].items():
            selected=[v for v in values if v['mode']==mode]
            assert len(selected)==len({v['document_id'] for v in selected})==cell['documents']
            for row in selected:
                assert len(row['logps16'])==16 and all(math.isfinite(v) for v in row['logps16'])
                assert abs(sum(row['logps16'])-row['span16_logprob'])<1e-8
                if panel=='DEV_confirm512':
                    assert set(row['component_logprobs'])==set(annotations[row['item_id']]['component_ranges'])
                    for part,(start,end) in annotations[row['item_id']]['component_ranges'].items():
                        assert abs(sum(row['logps16'][start:end])-row['component_logprobs'][part])<1e-8
            mean=np.mean([sum(v['logps16']) for v in selected])
            assert abs(mean-cell['mean_span16_logprob'])<1e-10
            assert abs(-mean/16-cell['mean_token_NLL'])<1e-10
            LP[model][panel][mode]={v['item_id']:v for v in selected}

indices={n:np.random.default_rng(2026101917).integers(0,n,size=(20000,n)) for n in [256,512]}


def check(reference,treatment,expected,metric='accuracy',part=None):
    assert reference.keys()==treatment.keys()
    order=sorted(reference)
    def val(v):
        if metric=='accuracy':return int(v['correct'])
        if metric=='NLL':return -sum(v['logps16'])/16
        return v['component_logprobs'][part] if part else sum(v['logps16'])
    delta=np.array([val(treatment[k])-val(reference[k]) for k in order],dtype=np.float64)
    assert len(delta)==expected['documents']==expected['items']
    assert expected['bootstrap_draws']==20000 and expected['bootstrap_seed']==2026101917
    assert abs(delta.mean()-expected['difference_b_minus_a'])<1e-10
    ci=np.quantile(delta[indices[len(delta)]].mean(1),[.025,.975])
    assert np.allclose(ci,expected['ci95'],atol=1e-10,rtol=0), (ci,expected['ci95'])


errors=sum(v['categories'].get('execution_error',0) for modes in report['tables'].values() for v in modes.values())
for seed,values in report['seeds'].items():
    direct=f'source_direct-seed{seed}-step320';cot=f'source_cot-seed{seed}-step320'
    old_direct=f'target_direct-seed{seed}-step320';old_cot=f'target_cot-seed{seed}-step320';prior=f'real_prior-seed{seed}-step320'
    pairs={
        'cot_own_minus_direct_raw':(direct,cot,'raw','own'),
        'cot_own_minus_cot_empty':(cot,cot,'empty','own'),
        'cot_own_minus_cot_raw':(cot,cot,'raw','own'),
        'cot_own_minus_cot_other':(cot,cot,'other','own'),
        'cot_own_minus_B0raw':('B0',cot,'raw','own'),
        'cot_raw_minus_direct_raw':(direct,cot,'raw','raw'),
        'cot_raw_minus_B0raw':('B0',cot,'raw','raw'),
        'direct_raw_minus_B0raw':('B0',direct,'raw','raw'),
        'direct_raw_minus_target_direct_raw':(old_direct,direct,'raw','raw'),
        'cot_own_minus_target_cot_own':(old_cot,cot,'own','own'),
        'cot_own_minus_CPTprior_own':(prior,cot,'own','own'),
        'target_cot_own_minus_target_direct_raw':(old_direct,old_cot,'raw','own')}
    for name,(a,b,am,bm) in pairs.items():
        check(scalar[a][am],scalar[b][bm],values['accuracy_contrasts'][name])
        check(LP[a]['DEV_confirm512'][am],LP[b]['DEV_confirm512'][bm],values['secondary_full16_contrasts'][name],metric='LL')
    for part,contrast in values['prospective_component_contrasts'].items():
        check(LP[direct]['DEV_confirm512']['raw'],LP[cot]['DEV_confirm512']['own'],contrast,metric='LL',part=part)
    for name,a,b in [('cot_minus_direct',direct,cot),('cot_minus_B0','B0',cot),('direct_minus_B0','B0',direct),('cot_minus_target_cot',old_cot,cot),('direct_minus_target_direct',old_direct,direct)]:
        check(LP[a]['DEV_ordinary256']['raw'],LP[b]['DEV_ordinary256']['raw'],values['ordinary_token_NLL_regression'][name],metric='NLL')
    c=values['accuracy_contrasts']
    passed=(c['cot_own_minus_direct_raw']['difference_b_minus_a']>=.03
            and c['cot_own_minus_direct_raw']['ci95'][0]>0
            and c['cot_own_minus_cot_empty']['ci95'][0]>0
            and c['cot_own_minus_B0raw']['ci95'][0]>0 and errors==0)
    assert passed==values['declared_primary_gate_pass']
    quality=values['ordinary_token_NLL_regression']['cot_minus_B0']['ci95'][1]<=.03
    assert quality==values['declared_ordinary_quality_gate_pass']
    assert bool(passed and quality)==values['declared_combined_gate_pass']
for key,field in [('accuracy_pass_both_seeds','declared_primary_gate_pass'),('ordinary_quality_pass_both_seeds','declared_ordinary_quality_gate_pass'),('combined_reference_pass_both_seeds','declared_combined_gate_pass')]:
    assert report['decisions'][key]==all(v[field] for v in report['seeds'].values())
assert report['decisions']['execution_error_rows_all_fixed_scalar_attempts']==errors
assert scalar_rows==28160 and note_rows==5632 and LP_rows==30976
result=dict(status='PASS_ALL_PUBLIC_NATIVE_TEXT_VALUES_AST_GOLD_COUNTS_CIS_AND_LITERAL_GATES',
    utc=datetime.now(timezone.utc).isoformat(),scalar_rows=scalar_rows,note_rows=note_rows,likelihood_rows=LP_rows,
    report_sha256=sha(path),checker_sha256=sha(__file__),
    limits='Native decode/stop and exact leading literal/value/AST checks plus independent vectorized resampling. Unsupported syntax statuses remain the frozen parser judgment;no model forwards rerun.')
with (directory/'independent-readout-check.json').open('x') as stream:
    json.dump(result,stream,indent=2)
print(json.dumps(result,indent=2),flush=True)

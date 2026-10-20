"""Independent all-document bootstrap reconstruction and fixed HF score spots.

Run only after all seven terminal readouts. No training, generation, filtering,
new decoding rule, or official TEST. CPU spot checks use fixed hash identities,
not high/low scores or correct/incorrect generations.
"""
import argparse
from fractions import Fraction
import gc
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from weekend_thinking.artifacts import atomic_json,sha256_file

parser=argparse.ArgumentParser(description=__doc__)
for name in ('index','report','output'):parser.add_argument('--'+name,type=Path,required=True)
args=parser.parse_args();read=lambda p:json.loads(Path(p).read_text())
lines=lambda p:[json.loads(s) for s in Path(p).open() if s.strip()]
index=read(args.index);report=read(args.report);release=read(index['release'])
assert report['status']=='COMPLETE_V29_ALL_FIXED_MODEL_READOUT'
assert sha256_file(index['release'])==index['release_sha256']
assert len(index['models'])==7 and set(index['models'])==set(release['models'])
for path,digest in report['source_pins'].items():assert sha256_file(path)==digest
args.output.mkdir(parents=True,exist_ok=False)
inventory={};targets={r['item_id']:r for r in lines(Path(index['data'])/'DEV_calculation.jsonl')}
ordinary_targets={r['item_id']:r for r in lines(Path(index['data'])/'DEV_ordinary.jsonl')}
assert len(targets)==768 and len(ordinary_targets)==4096
from transformers import AutoTokenizer
from train.evaluate import asset_provenance
from train.calculation_boundaries import render_value
base_ref=read(Path(release['references']['B0'])/'reference.json')
assert asset_provenance(Path(base_ref['model_dir']))==base_ref['assets']
tok=AutoTokenizer.from_pretrained(base_ref['model_dir'],local_files_only=True)
opening=tok.encode('\nCalculation:\n',add_special_tokens=False)
closing=tok.encode('Answer:\n',add_special_tokens=False)
assert closing[0]==16141
stop_ids=(151643,16141);defined=frozenset(tok.get_vocab().values())
ordered=sorted(targets,key=lambda key:hashlib.sha256(('v29-other:'+key).encode()).hexdigest())
donors={key:ordered[(i+1)%len(ordered)] for i,key in enumerate(ordered)}
corrupt={}
for key,row in targets.items():
    text=tok.decode(row['full_calculation_ids'],skip_special_tokens=False,clean_up_tokenization_spaces=False)
    assert text==row['full_calculation_text']
    head,last=text.rstrip('\n').rsplit(' = ',1)
    corrupt[key]=tok.encode(head+' = '+render_value(Fraction(last)+1)+'\n',add_special_tokens=False)

def expected_context(model,key,mode):
    q=targets[key];notes=inventory[model]['notes_by_id']
    bodies=dict(raw=None,empty=[],own=notes[key]['content_ids'],
        other=notes[donors[key]]['content_ids'],oracle=q['full_calculation_ids'],corrupt=corrupt[key])
    note=bodies[mode]
    return q['prefix_ids'] if note is None else q['prefix_ids']+opening+note+closing
for model,paths in index['models'].items():
    math_rows=lines(Path(paths['math'])/'logps.jsonl');ordinary=lines(Path(paths['ordinary'])/'logps.jsonl')
    assert len(math_rows)==4608 and len(ordinary)==4096
    notes=lines(Path(paths['writer'])/'notes.jsonl')
    assert len(notes)==768 and {r['item_id'] for r in notes}==set(targets)
    inventory[model]=dict(math=math_rows,ordinary=ordinary,notes=notes,notes_by_id={n['item_id']:n for n in notes})
    for note in notes:
        q=targets[note['item_id']];actions=note['actions'];stopped=bool(actions and actions[-1] in stop_ids)
        assert note['prompt_ids']==q['prefix_ids']+opening and note['document_id']==q['document_id']
        assert len(actions)<=128 and all(type(t) is int and t in defined for t in actions)
        assert not any(t in stop_ids for t in actions[:-1])
        assert note['content_ids']==(actions[:-1] if stopped else actions)
        text=tok.decode(note['content_ids'],skip_special_tokens=False,clean_up_tokenization_spaces=False)
        assert text==note['text']==note['raw_text'] and note['n_actions']==len(actions)
        assert note['status'] in ('ok','error')
        if note['status']=='ok':
            assert note['stop_type']==(f'token:{actions[-1]}' if stopped else 'length_cap')
            assert stopped or len(actions)==128
    assert len({(r['item_id'],r['mode']) for r in math_rows})==4608
    for row in math_rows:
        source=targets[row['item_id']]
        assert row['target_ids']==source['continuation16_ids'] and len(row['token_logps'])==16
        assert row['context_ids']==expected_context(model,row['item_id'],row['mode'])
    for row in ordinary:
        source=ordinary_targets[row['item_id']]
        assert row['context_ids']==source['prefix_ids'][:1]
        assert row['target_ids']==(source['prefix_ids']+source['continuation16_ids'])[1:]
        assert len(row['token_logps'])==527

def values(model,mode,metric):
    ordinary=metric=='mean_document_NLL_delta'
    result={}
    for row in inventory[model]['ordinary' if ordinary else 'math']:
        if row['mode']!=mode:continue
        if ordinary:score=-math.fsum(row['token_logps'])/527
        elif metric=='mean_document_span16_logp_gain':score=math.fsum(row['token_logps'])
        else:
            part=metric.removeprefix('mean_document_').removesuffix('_logp_gain')
            mask=row['component_masks'][part]
            assert len(mask)==16
            score=math.fsum(v for v,use in zip(row['token_logps'],mask) if use)
        result[row['item_id']]=(row['document_id'],score)
    assert len(result)==(4096 if ordinary else 768)
    assert len({v[0] for v in result.values()})==len(result)
    return result

checks=[]
for contrast in report['all_paired_contrasts']:
    a=values(contrast['reference_model'],contrast['reference_mode'],contrast['metric'])
    b=values(contrast['treatment_model'],contrast['treatment_mode'],contrast['metric'])
    assert set(a)==set(b) and all(a[k][0]==b[k][0] for k in a)
    delta=np.asarray([b[k][1]-a[k][1] for k in sorted(a)],dtype=np.float64)
    assert np.isfinite(delta).all();rng=np.random.default_rng(2026102029);sample=[]
    # A different implementation from the producer's one-draw-at-a-time loop.
    for start in range(0,10000,128):
        draw=rng.integers(0,len(delta),size=(min(128,10000-start),len(delta)))
        sample.extend(delta[draw].sum(axis=1)/len(delta))
    mean=float(delta.sum()/len(delta));interval=np.quantile(sample,[.025,.975]).tolist()
    error=max(abs(mean-contrast['difference_b_minus_a']),*(abs(a-b) for a,b in zip(interval,contrast['ci95'])))
    assert error<1e-10 and contrast['documents']==contrast['items']==len(delta),(contrast['name'],error)
    checks.append(dict(name=contrast['name'],documents=len(delta),summary_error=error))
    print('RECONSTRUCTED',contrast['name'],flush=True)
decisions={}
named={v['name']:v for v in report['all_paired_contrasts']}
assert len(named)==len(report['all_paired_contrasts'])
primary_names=dict(raw_control='training-vs-raw',format_control='training-vs-format',base='absolute-base',
    ordinary_base='ordinary-base-cost',own_raw='own-raw',own_empty='own-empty',own_raw_control='own-raw-control')
for seed in ('2026102029','2026102030'):
    c=report['by_order'][seed]
    assert all(c[key]==named[seed+':'+name] for key,name in primary_names.items())
    eligible=all(v['status']=='ok' for v in inventory['computation-seed'+seed]['notes'])
    assert eligible==c['inference_execution_complete']
    good=lambda v:v['difference_b_minus_a']>=.25 and v['ci95'][0]>0
    recipe=good(c['raw_control']) and good(c['format_control']);absolute=good(c['base'])
    retention=c['ordinary_base']['ci95'][1]<=.03
    inference=c['inference_execution_complete'] and all(good(c[k]) for k in ('own_raw','own_empty','own_raw_control'))
    decisions[seed]=dict(training_recipe_pass=recipe,absolute_base_pass=absolute,
        retention_target_met=retention,overall_CPT_criterion=recipe and absolute and retention,inference_gate_pass=inference)
assert decisions==report['decision']['by_order']

# Re-run the frozen parser, retaining unsupported, capped and failed attempts.
from train.scalar_completion import parse_initial_scalar_v2
for model,paths in index['models'].items():
    attempts=[v for path in paths['scalar'] for v in lines(Path(path)/'answers.jsonl')]
    notes={v['item_id']:v for v in inventory[model]['notes']}
    assert len(attempts)==2304 and len({(v['item_id'],v['mode']) for v in attempts})==2304
    for mode in ('raw','empty','own'):
        subset=[v for v in attempts if v['mode']==mode];assert len(subset)==768
        counts=dict(attempts=768,correct=0,unsupported=0,incomplete=0,format_errors=0,
            execution_errors=0,source_note_errors=0,parser_errors=0,capped=0,generated_actions=0)
        for row in subset:
            parsed=parse_initial_scalar_v2(row['text'],row['stop_type']);assert parsed==row['parsed']
            assert row['prompt_ids']==expected_context(model,row['item_id'],mode)
            source_status=notes[row['item_id']]['status'] if mode=='own' else 'ok'
            assert source_status==row['source_note_status'] and row['n_actions']==len(row['actions'])
            valid=row['status']=='ok' and source_status=='ok' and parsed['status']=='valid'
            counts['correct']+=int(valid and Fraction(parsed['value'])==Fraction(targets[row['item_id']]['numeric_value']))
            counts['unsupported']+=parsed['status']=='unsupported';counts['incomplete']+=parsed['status']=='incomplete'
            counts['format_errors']+=parsed['status']=='format_error';counts['parser_errors']+=parsed['status']=='error'
            counts['execution_errors']+=row['status']!='ok' or source_status!='ok'
            counts['source_note_errors']+=source_status!='ok';counts['capped']+=row['stop_type']=='length_cap'
            counts['generated_actions']+=len(row['actions'])
        original=next(v for v in report['scalar_secondary'] if v['model']==model and v['mode']==mode)
        assert all(original[k]==value for k,value in counts.items())

# Standard independent HF forwards on fixed held identities after all models
# and endpoints are fixed; this verifies scores, not generalization by itself.
from train.evaluate_local import setup_model
torch.set_num_threads(16);spots=[]
spot_stream=(args.output/'HF-spots.jsonl').open('x')
for model,paths in index['models'].items():
    binding=release['models'][model]
    ref=read(Path(release['references'][model])/'reference.json')
    backend,_,loaded,_,_=setup_model(SimpleNamespace(model_dir=Path(ref['model_dir']),
        checkpoint=Path(binding['checkpoint']) if binding['checkpoint'] else None,device='cpu'))
    if binding['checkpoint'] is not None:
        assert loaded['sha256']==binding['checkpoint_sha256'] and loaded['step']==1280
    selected=[]
    for mode in ('raw','empty','own','other','oracle','corrupt'):
        selected+=sorted([r for r in inventory[model]['math'] if r['mode']==mode],
            key=lambda r:hashlib.sha256(('v29-held-score-check:'+r['item_id']).encode()).hexdigest())[:2]
    selected+=sorted(inventory[model]['ordinary'],
        key=lambda r:hashlib.sha256(('v29-held-ordinary-check:'+r['item_id']).encode()).hexdigest())[:2]
    with torch.no_grad():
        for row in selected:
            context,target=row['context_ids'],row['target_ids'];ids=torch.tensor([context+target])
            lp=backend.model(ids).logits[0,len(context)-1:len(context)+len(target)-1].float().log_softmax(-1)
            actual=lp.gather(-1,ids[0,len(context):,None])[:,0].tolist()
            error=max(abs(a-b) for a,b in zip(actual,row['token_logps']))
            spot=dict(model=model,item_id=row['item_id'],mode=row['mode'],
                targets=len(actual),CPU_logps=actual,max_abs_error=error)
            spot_stream.write(json.dumps(spot,allow_nan=False)+'\n');spot_stream.flush()
            assert len(actual)==len(target) and error<=.001,(model,row['item_id'],row['mode'],error)
            spots.append(spot)
    del backend,loaded;gc.collect();print('INDEPENDENT_HF_SPOTS',model,flush=True)
spot_stream.close()
atomic_json(args.output/'independent-check.json',dict(status='PASS_ALL_PAIRED_DOCUMENT_CONTRASTS_DECISIONS_AND_FIXED_HF_SPOTS',
    contrast_checks=checks,decisions=decisions,HF_spots=spots,
    all_contexts_and_writer_prefix_only_inputs_reconstructed=True,
    index_sha256=sha256_file(args.index),report_sha256=sha256_file(args.report),source_sha256=sha256_file(__file__),
    HF_spots_sha256=sha256_file(args.output/'HF-spots.jsonl'),
    limits='All cohort statistical reconstruction plus98 independently executed HF score spots; not all-row independent HF forwards or external replication.'))
print('PASS_COMPLETE_READOUT_RECONSTRUCTION',flush=True)

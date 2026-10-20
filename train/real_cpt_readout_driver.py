"""V29 readout through the existing qualified causal model and decoders.

This module does not train or schedule jobs. References use fixed TRAIN rows
and independent ordinary HF forwards. Scientific workers require the complete
seven-model terminal release before they open any held question or target.
Each worker is finite, single-attempt, and retains original partial outputs.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time
from types import SimpleNamespace

import torch

from train.real_cpt_readout import (MATH_MODES, SCALAR_MODES, controls, donors,
    equal_length_batches, numeric_masks, validate_note)
from train.real_trace_contexts import (boundary_stop_ids, question_projection,
    reader_context, writer_prompt)
from weekend_thinking.artifacts import atomic_json, require_utc_deadline, sha256_file, utc_now

ROOT = Path(__file__).resolve().parents[1]
DATA_SHA = '77c8774d3fdeb283728878c44522dab541ec61a213d3f89ece471b3dcc45fb3f'
TRAIN_SHA = 'ef15ff29e4c972ed6162f05d293bdc1059967e2914a0e9973e7b0f95f71bd509'
PANELS = ('writer', 'math', 'ordinary', 'scalar')
HELPERS = ('train/real_trace_contexts.py', 'train/real_trace_evaluation.py',
    'train/document_qa_span_scores.py', 'train/batched_causal_decoder.py',
    'train/evaluate.py', 'train/evaluate_local.py', 'train/scalar_completion.py',
    'train/neuron_backend.py', 'train/real_cpt_scaffold.py')


def read(path):
    return json.loads(Path(path).read_text())


def rows(path):
    return [json.loads(s) for s in Path(path).open() if s.strip()]


def sources():
    names = (*HELPERS, 'train/real_cpt_readout.py', 'train/real_cpt_readout_driver.py',
             'train/partial_state_answer_driver.py', 'train/useful_note_precision.py',
             'train/computable_equations.py', 'train/calculation_boundaries.py',
             'train/symbolic_note_labels.py', 'train/real_trace_packing.py',
             'weekend_thinking/pretrained.py', 'weekend_thinking/runtime_policy.py')
    return {name: sha256_file(ROOT/name) for name in names}


def check_model(loaded, binding):
    if binding['checkpoint'] is None:
        if binding['id'] != 'B0' or loaded is not None:
            raise ValueError('Unchanged released B0 required')
    elif (loaded['sha256'] != binding['checkpoint_sha256']
            or any(loaded[k] != binding[k] for k in ('run_id', 'step', 'spec_hash'))):
        raise ValueError('Actual full checkpoint differs from frozen model binding')


def prepare(args):
    """Only TRAIN is opened here; no held forwards, questions, or labels."""
    from train.evaluate import asset_provenance
    from train.evaluate_local import setup_model
    from train.real_cpt_scaffold import verified_work
    from weekend_thinking.runtime_policy import runtime_policy
    if sha256_file(args.data/'manifest.json') != DATA_SHA or sha256_file(args.data/'TRAIN.jsonl') != TRAIN_SHA:
        raise ValueError('Original V29 data manifest and TRAIN bytes required')
    binding = read(args.binding)
    if args.purpose == 'science':
        if binding['id'] != 'B0':
            if (binding['status'] != 'AUDITED_V29_TERMINAL1280'
                    or binding['step'] != 1280
                    or sha256_file(binding['audit_file']) != binding['audit_sha256']):
                raise ValueError('Independently audited fixed terminal1280 required')
    elif binding['checkpoint'] is None or binding['step'] != 320:
        raise ValueError('Engineering reference requires the frozen V17 terminal320')
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(16)
    started = time.monotonic()
    train = rows(args.data/'TRAIN.jsonl')
    def select(kind, tag, count, full_prefix=False):
        selected = [r for r in train if r['kind'] == kind and
                    (not full_prefix or len(r['prefix_ids']) == 512)]
        selected.sort(key=lambda r: hashlib.sha256((tag+r['item_id']).encode()).hexdigest())
        return selected[:count]
    calculation = select('calculation', 'v29-native-calculation:', 8, True)
    ordinary = select('ordinary', 'v29-full-window:', 4)
    if len(calculation) != 8 or len(ordinary) != 4:
        raise ValueError('Complete fixed TRAIN reference inventory required')
    backend, tok, loaded, signature, _ = setup_model(SimpleNamespace(
        model_dir=args.model_dir, checkpoint=Path(binding['checkpoint']) if binding['checkpoint'] else None,
        device='cpu'))
    check_model(loaded, binding)
    if any(p.dtype != torch.float32 for p in backend.model.parameters()):
        raise ValueError('Independent HF reference uses FP32 parameters')
    q = [question_projection(r) for r in calculation]
    reference = {'writer': [], 'math': [], 'ordinary': [], 'scalar': []}
    inputs = dict(writer=[dict(question=v, context_ids=writer_prompt(v, tok)) for v in q],
                  math=[], ordinary=[], scalar=[])
    for i, row in enumerate(calculation[:4]):
        question = q[i]
        _, work = verified_work(question['prefix_ids'], tok)
        contexts = controls(question, tok, dict(content_ids=work), dict(content_ids=work))
        mode = ('raw', 'empty', 'oracle', 'corrupt')[i]
        inputs['math'].append(dict(item_id=row['item_id'], context_ids=contexts[mode],
                                   target_ids=row['continuation16_ids'], mode=mode))
        inputs['scalar'].append(dict(item_id=row['item_id'], context_ids=contexts[mode]))
    for row in ordinary:
        ids = row['prefix_ids']+row['continuation16_ids']
        inputs['ordinary'].append(dict(item_id=row['item_id'], context_ids=ids[:1], target_ids=ids[1:]))
    with torch.no_grad():
        for panel, values in inputs.items():
            for value in values:
                context, target = value['context_ids'], value.get('target_ids', [])
                ids = torch.tensor([context+target])
                lp = backend.model(ids).logits[0].float().log_softmax(-1)
                positions = [0, 263, 526] if panel == 'ordinary' else [len(context)-1]
                reference[panel].append(lp[positions].cpu())
                if target:
                    target_lp = lp[len(context)-1:len(context)+len(target)-1]
                    value['HF_target_logps'] = target_lp.gather(-1, ids[0, len(context):, None])[:, 0].tolist()
    native = {panel: torch.cat(values) for panel, values in reference.items()}
    torch.save(native, args.output/'CPU-native.pt')
    atomic_json(args.output/'TRAIN-inputs.json', inputs)
    contract = read(args.contract)
    if any(sha256_file(ROOT/name) != digest for name, digest in contract['source_files'].items()):
        raise ValueError('Numerical helpers differ from prospectively frozen readout')
    assets = asset_provenance(args.model_dir)
    if assets['model_id'] != 'Qwen/Qwen3-1.7B-Base':
        raise ValueError('Actual released base assets required')
    protocol = dict(status='PREPARED_TRAIN_ONLY_INDEPENDENT_HF_V29_REFERENCE', purpose=args.purpose,
        started_utc=utc_now(), model=binding, signature=signature, assets=assets,
        model_dir=str(args.model_dir.resolve()), data=str(args.data.resolve()),
        runtime_policy=runtime_policy(),
        manifest_sha256=DATA_SHA, TRAIN_sha256=TRAIN_SHA,
        contract=str(args.contract.resolve()), contract_sha256=sha256_file(args.contract),
        source_files=sources(), inputs_sha256=sha256_file(args.output/'TRAIN-inputs.json'),
        native_sha256=sha256_file(args.output/'CPU-native.pt'),
        wall_seconds=time.monotonic()-started,
        limits='TRAIN-only numeric execution reference; not learning, held-out evaluation or generalization evidence.')
    atomic_json(args.output/'reference.json', protocol)
    return protocol


def load_reference(directory):
    p = read(directory/'reference.json')
    if p['status'] != 'PREPARED_TRAIN_ONLY_INDEPENDENT_HF_V29_REFERENCE':
        raise ValueError('Actual independent fixed TRAIN reference required')
    if (p['source_files'] != sources() or sha256_file(p['contract']) != p['contract_sha256']
            or sha256_file(directory/'TRAIN-inputs.json') != p['inputs_sha256']
            or sha256_file(directory/'CPU-native.pt') != p['native_sha256']):
        raise ValueError('Frozen source/reference bytes changed')
    return p, read(directory/'TRAIN-inputs.json')


def release_for(args, p):
    """All terminal identities fixed before any scientific held worker opens data."""
    if args.qualify:
        return None
    if p['purpose'] != 'science' or args.release is None:
        raise ValueError('Scientific held worker requires explicit complete release')
    release = read(args.release)
    if (release['status'] != 'RELEASED_ALL_SEVEN_V29_FIXED_TERMINAL_READOUTS'
            or release['contract_sha256'] != p['contract_sha256']
            or release['source_files'] != sources() or len(release['models']) != 7
            or release['models'][p['model']['id']] != p['model']):
        raise ValueError('All seven fixed audited identities and exact-source release required')
    for key, model in release['models'].items():
        if key != model['id']:
            raise ValueError('Model inventory keys must match full original identities')
        if model['id'] == 'B0':
            if model['checkpoint'] is not None:
                raise ValueError('B0 must remain unchanged')
        elif (model['status'] != 'AUDITED_V29_TERMINAL1280' or model['step'] != 1280
                or sha256_file(model['audit_file']) != model['audit_sha256']
                or sha256_file(model['checkpoint']) != model['checkpoint_sha256']):
            raise ValueError('Complete fixed terminal model inventory changed')
    if set(release['models']) != {'B0'} | {f'{arm}-seed{seed}' for arm in ('raw','format','computation') for seed in (2026102029,2026102030)}:
        raise ValueError('Exact original six terminal models plus B0 required')
    return release


@torch.no_grad()
def native_gate(backend, inputs, reference, panel, output):
    """Every actual worker checks its own model before opening held rows."""
    from train.useful_note_precision import compare_distributions
    width = {'writer':544, 'math':672, 'ordinary':528, 'scalar':672}[panel]
    values = []
    for row in inputs[panel]:
        context, target = row['context_ids'], row.get('target_ids', [])
        x = torch.full((1,width),151643,dtype=torch.long)
        x[0,:len(context)+len(target)] = torch.tensor(context+target)
        if panel == 'ordinary':
            x = x[:,:-1]
        h = backend.hidden(x.to(backend.device))
        positions = [0,263,526] if panel == 'ordinary' else [len(context)-1]
        lp = backend.model.lm_head(h[:,positions]).float().log_softmax(-1)
        backend.sync(); values.append(lp[0].cpu())
    actual = torch.cat(values)
    torch.save(actual, output/'native-logps.pt')
    checks = compare_distributions(reference[panel], actual)
    atomic_json(output/'native-check.json',checks)
    if not all(v['passed'] and v['max_logp_error'] <= .001 for v in checks):
        raise RuntimeError('Actual model fixed TRAIN native distribution failed')
    return checks


def fixed_scores(backend, inputs, panel, output):
    from train.document_qa_span_scores import score_spans
    subset = inputs[panel]
    ordinary = panel == 'ordinary'
    actual = score_spans(backend,[r['context_ids'] for r in subset],[r['target_ids'] for r in subset],
        width=528 if ordinary else 672,target_slots=528 if ordinary else 16)
    error = max(abs(a-b) for row,value in zip(subset,actual)
                for a,b in zip(row['HF_target_logps'],value['token_logps']))
    if any(len(a['HF_target_logps']) != b['tokens'] for a,b in zip(subset,actual)) or error > .001:
        raise RuntimeError('Fixed independent HF full-span parity failed')
    atomic_json(output/'TRAIN-full-span.json',dict(passed=True,max_token_error=error,rows=actual))
    return actual


@torch.no_grad()
def cached_gate(backend, tok, inputs, reference, panel, output, shard):
    from train.batched_causal_decoder import BatchedGreedyDecoder
    from train.evaluate import GreedyDecoder
    from train.neuron_backend import CausalCache
    from train.partial_state_answer_driver import cache_prefill_check
    from train.useful_note_precision import compare_distributions
    decoder = GreedyDecoder(backend.model,tok,sync=backend.sync,banned_ids=backend.banned_ids)
    width, cap = (544,128) if panel == 'writer' else (672,32)
    index = 0 if panel == 'writer' else shard
    check, vectors = cache_prefill_check(backend,decoder,inputs[panel][index]['context_ids'],width=width,cap=cap)
    torch.save(vectors,output/'cache-prefill-logps.pt')
    atomic_json(output/'cache-prefill.json',check)
    if not check['passed'] or check['max_logp_error'] > .001:
        raise RuntimeError('Actual cache/full-forward distribution failed')
    if panel == 'writer':
        batched = BatchedGreedyDecoder(backend.model,tok,sync=backend.sync,banned_ids=backend.banned_ids)
        for size in (4,8):
            prompt = [r['context_ids'] for r in inputs['writer'][:size]]
            if len({len(p) for p in prompt}) != 1:
                raise ValueError('Fixed writer native rows must have equal real lengths')
            x = torch.full((size,width),151643,dtype=torch.long)
            x[:,:len(prompt[0])] = torch.tensor(prompt)
            cache = CausalCache(backend.model.config.num_hidden_layers,width+cap)
            lp = batched._next(x.to(backend.device),torch.arange(width,device=backend.device),cache,
                torch.tensor([len(prompt[0])-1],device=backend.device)).float().log_softmax(-1)
            backend.sync(); actual=lp.cpu()
            checks=compare_distributions(reference['writer'][:size],actual)
            torch.save(actual,output/f'cached-B{size}-native.pt')
            atomic_json(output/f'cached-B{size}-native.json',checks)
            if not all(v['passed'] and v['max_logp_error'] <= .001 for v in checks):
                raise RuntimeError('Actual batched-cache independent native parity failed')
        return decoder, batched
    return decoder, None


def cohort(p, filename, count):
    data = Path(p['data']); manifest = read(data/'manifest.json')
    if sha256_file(data/'manifest.json') != p['manifest_sha256']:
        raise ValueError('Frozen source manifest changed')
    expected = manifest['files'][filename]
    if expected['rows'] != count or sha256_file(data/filename) != expected['sha256']:
        raise ValueError('Frozen complete cohort bytes changed')
    values = rows(data/filename)
    if (len(values) != count or len({v['item_id'] for v in values}) != count
            or len({v['document_id'] for v in values}) != count):
        raise ValueError('One original attempt per unique held document required')
    return values


def load_notes(path, p, questions, tok):
    bundle = read(path)
    if (bundle['status'] != 'COMPLETE_V29_INTACT_ORIGINAL_NOTES'
            or bundle['model'] != p['model'] or bundle['question_ids'] != [q['item_id'] for q in questions]
            or bundle['reference_sha256'] != sha256_file(Path(bundle['reference'])/'reference.json')):
        raise ValueError('Original same-model same-question note bundle required')
    for name, digest in bundle['source_pins'].items():
        if sha256_file(name) != digest:
            raise ValueError('Original writer attempt evidence changed')
    original = rows(bundle['notes_file'])
    by_id = {v['item_id']:v for v in original}
    if len(original) != len(questions) or set(by_id) != {q['item_id'] for q in questions}:
        raise ValueError('Every original writer attempt, including failures, required')
    stops = boundary_stop_ids(tok)
    for q in questions:
        validate_note(by_id[q['item_id']],q,tok,stop_ids=stops)
    return by_id


def write_line(stream, value):
    stream.write(json.dumps(value,allow_nan=False)+'\n');stream.flush()


def writer(backend, tok, p, output, decoder, batched, check_deadline):
    from train.batched_causal_decoder import BatchGenerationFailure
    from train.evaluate import GenerationFailure
    # This file is the only held data the writer opens. It has no future/label.
    questions = cohort(p,'DEV_calculation-questions.jsonl',768)
    if any(set(q) != {'item_id','document_id','prefix_ids'} for q in questions):
        raise ValueError('Strict question-only held projection required')
    stops = boundary_stop_ids(tok)
    count = errors = 0
    with (output/'notes.jsonl').open('x') as stream:
        for batch_id, indices in enumerate(equal_length_batches(questions)):
            check_deadline(); prompts = [writer_prompt(questions[i],tok) for i in indices]
            before = time.monotonic(); error = None
            try:
                if len(indices) == 1:
                    generated = [decoder.decode(prompts[0],max_new_tokens=128,prefill_width=544,
                        pad_id=151643,stop_ids=stops,answer_line_stop=False)]
                else:
                    generated = batched.decode_batch(prompts,max_new_tokens=128,prefill_width=544,
                        pad_id=151643,stop_ids=stops)
            except (GenerationFailure,BatchGenerationFailure) as failure:
                generated = [failure.partial] if isinstance(failure,GenerationFailure) else failure.partials
                error = f'{type(failure).__name__}: {failure}'
            elapsed = time.monotonic()-before
            for i,prompt,value in zip(indices,prompts,generated):
                row = dict(item_id=questions[i]['item_id'],document_id=questions[i]['document_id'],
                    prompt_ids=prompt,status='error' if error else 'ok',**asdict(value),
                    batch_id=batch_id,batch_size=len(indices),batch_execution_seconds=elapsed,
                    allocated_share_seconds=elapsed/len(indices),n_actions=len(value.actions))
                if error:row['error']=error
                validate_note(row,questions[i],tok,stop_ids=stops)
                write_line(stream,row);count+=1;errors+=bool(error)
            if count % 64 == 0: print('WRITER_ATTEMPTS',count,flush=True)
    if count != 768:
        raise RuntimeError('Incomplete writer attempt inventory')
    return dict(rows=count,execution_errors=errors,ordered_question_ids=[q['item_id'] for q in questions])


def collect(args):
    """No model execution or repair: pin every original native writer attempt."""
    from transformers import AutoTokenizer
    p,_=load_reference(args.reference)
    if p['purpose']!='science':raise ValueError('Engineering outputs cannot become held notes')
    run=read(args.output/'run.json')
    if run['status']!='COMPLETE_ALL_DECLARED_ATTEMPTS' or run['panel']!='writer' or run['rows']!=768:
        raise ValueError('Complete original writer process required')
    # Physical closure is supplied by the external resource supervisor.
    closure=read(args.output/'closure.json')
    if closure['status']!='SESSION_AND_PHYSICAL_DEVICE_CLOSED':
        raise ValueError('Actual writer session/device closure required')
    qs=cohort(p,'DEV_calculation-questions.jsonl',768)
    tok=AutoTokenizer.from_pretrained(p['model_dir'],local_files_only=True)
    values=rows(args.output/'notes.jsonl');by_id={v['item_id']:v for v in values}
    if len(values)!=768 or set(by_id)!={q['item_id'] for q in qs}:
        raise ValueError('All original notes required without resampling')
    for q in qs:validate_note(by_id[q['item_id']],q,tok,stop_ids=boundary_stop_ids(tok))
    pins={str(path.resolve()):sha256_file(path) for path in args.output.iterdir() if path.is_file()}
    bundle=dict(status='COMPLETE_V29_INTACT_ORIGINAL_NOTES',model=p['model'],
        reference=str(args.reference.resolve()),reference_sha256=sha256_file(args.reference/'reference.json'),
        notes_file=str((args.output/'notes.jsonl').resolve()),source_pins=pins,
        question_ids=[q['item_id'] for q in qs],donors=donors([q['item_id'] for q in qs]),
        limits='All original caps/empty/failed attempts retained; an execution error makes inference gate ineligible.')
    atomic_json(args.output/'notes-bundle.json',bundle)
    return dict(status=bundle['status'],rows=768)


def likelihood(backend, tok, p, args, check_deadline):
    from train.document_qa_span_scores import score_spans
    ordinary=args.panel=='ordinary'
    values=cohort(p,'DEV_ordinary.jsonl' if ordinary else 'DEV_calculation.jsonl',4096 if ordinary else 768)
    notes=None if ordinary else load_notes(args.notes,p,[question_projection(v) for v in values],tok)
    donor=None if ordinary else donors([v['item_id'] for v in values])
    prepared={}
    if not ordinary:
        for row in values:
            key=row['item_id'];own,other=notes[key],notes[donor[key]]
            prepared[key]=(controls(question_projection(row),tok,own,other),numeric_masks(row,tok))
    count=0
    with (args.output/'logps.jsonl').open('x') as stream:
        for mode in ('raw',) if ordinary else MATH_MODES:
            for offset in range(0,len(values),4):
                check_deadline();batch=values[offset:offset+4];contexts=[];targets=[];states=[]
                for row in batch:
                    if ordinary:
                        ids=row['prefix_ids']+row['continuation16_ids']
                        contexts.append(ids[:1]);targets.append(ids[1:]);states.append('ok')
                    else:
                        key=row['item_id'];own,other=notes[key],notes[donor[key]]
                        contexts.append(prepared[key][0][mode])
                        targets.append(row['continuation16_ids'])
                        states.append(own['status'] if mode=='own' else other['status'] if mode=='other' else 'ok')
                result=score_spans(backend,contexts,targets,width=528 if ordinary else 672,
                                   target_slots=528 if ordinary else 16)
                for row,context,target,value,state in zip(batch,contexts,targets,result,states):
                    output=dict(item_id=row['item_id'],document_id=row['document_id'],mode=mode,
                        context_ids=context,target_ids=target,note_execution_status=state,**value)
                    if ordinary:
                        output.update(mean_NLL=-value['full_logp']/527,
                            suffix16_logps=value['token_logps'][-16:],suffix16_logp=sum(value['token_logps'][-16:]))
                    else:
                        masks=prepared[row['item_id']][1]
                        output['component_masks']=masks
                        output['component_logps']={k:sum(lp for lp,use in zip(value['token_logps'],mask) if use)
                                                  for k,mask in masks.items()}
                    write_line(stream,output);count+=1
                if count%256==0:print('SCORED',args.panel,mode,count,flush=True)
    if count!=(4096 if ordinary else 768*6):raise RuntimeError('Incomplete likelihood inventory')
    return dict(rows=count,modes=['raw'] if ordinary else list(MATH_MODES))


def scalar(backend, tok, p, args, decoder, check_deadline):
    from train.evaluate import GenerationFailure
    from train.scalar_completion import parse_initial_scalar_v2
    qs=cohort(p,'DEV_calculation-questions.jsonl',768)
    notes=load_notes(args.notes,p,qs,tok)
    count=errors=0
    with (args.output/'answers.jsonl').open('x') as stream:
        for q in qs[args.shard::4]:
            # No future, numeric answer, or teacher trace is opened here.
            for mode in SCALAR_MODES:
                check_deadline();key=q['item_id']
                content=None if mode=='raw' else [] if mode=='empty' else notes[key]['content_ids']
                prompt=reader_context(q,tok,content)
                before=time.monotonic();value=dict(actions=[],content_ids=[],text='',raw_text='',stop_type='error')
                error=None
                try:
                    value=asdict(decoder.decode(prompt,max_new_tokens=32,prefill_width=672,
                        pad_id=151643,stop_ids=(151643,),answer_line_stop=False))
                except GenerationFailure as failure:
                    value=asdict(failure.partial);error=f'{type(failure).__name__}: {failure}'
                result=dict(item_id=key,document_id=q['document_id'],mode=mode,prompt_ids=prompt,
                    status='error' if error else 'ok',**value,n_actions=len(value['actions']),
                    source_note_status=notes[key]['status'] if mode=='own' else 'ok',
                    wall_seconds=time.monotonic()-before)
                result['parsed']=parse_initial_scalar_v2(value['text'],value['stop_type'])
                if error:result['error']=error
                write_line(stream,result);count+=1;errors+=bool(error)
            if count%96==0:print('SCALAR_ATTEMPTS',args.shard,count,flush=True)
    if count!=768//4*3:raise RuntimeError('Incomplete scalar attempt inventory')
    return dict(rows=count,execution_errors=errors,shard=args.shard)


def run(args):
    from train.evaluate import asset_provenance
    from train.evaluate_local import setup_model
    from weekend_thinking.runtime_policy import require_runtime_policy, runtime_policy
    deadline=require_utc_deadline(args.deadline_utc)
    def check():
        if datetime.now(timezone.utc)>=deadline:raise TimeoutError('Finite V29 worker deadline; no automatic retry')
    check();p,inputs=load_reference(args.reference)
    require_runtime_policy(p['runtime_policy'],runtime_policy())
    release_for(args,p)
    if (os.environ.get('NEURONCORE_NUM_DEVICES')!='1'
            or not os.environ.get('NEURON_RT_VISIBLE_CORES','').isdigit()):
        raise ValueError('Exactly one explicitly isolated logical core per worker required')
    if args.panel=='scalar' and args.shard not in range(4):
        raise ValueError('Exactly one of four fixed scalar shards required')
    args.output.mkdir(parents=True,exist_ok=False);started=time.monotonic()
    report=dict(status='running',panel=args.panel,shard=args.shard,started_utc=utc_now(),
        reference=str(args.reference.resolve()),reference_sha256=sha256_file(args.reference/'reference.json'),
        release_sha256=sha256_file(args.release) if args.release else None,
        model=p['model'],source_files=sources(),runtime_policy=runtime_policy(),qualify_only=args.qualify)
    atomic_json(args.output/'run.json',report)
    try:
        if asset_provenance(Path(p['model_dir']))!=p['assets']:raise ValueError('Actual released asset bytes changed')
        backend,tok,loaded,signature,_=setup_model(SimpleNamespace(model_dir=Path(p['model_dir']),
            checkpoint=Path(p['model']['checkpoint']) if p['model']['checkpoint'] else None,device='xla'))
        check_model(loaded,p['model'])
        if signature!=p['signature'] or any(t.dtype!=torch.float32 for t in backend.model.parameters()):
            raise ValueError('Actual native model differs from FP32 CPU reference')
        native=torch.load(args.reference/'CPU-native.pt',map_location='cpu',weights_only=True)
        report['native']=native_gate(backend,inputs,native,args.panel,args.output)
        if args.panel in ('math','ordinary'):fixed_scores(backend,inputs,args.panel,args.output)
        else:decoder,batched=cached_gate(backend,tok,inputs,native,args.panel,args.output,args.shard)
        check()
        if args.qualify:report.update(status='PASS_FIXED_TRAIN_NATIVE_AND_PANEL_MECHANICS',rows=0)
        else:
            if args.panel=='writer':result=writer(backend,tok,p,args.output,decoder,batched,check)
            elif args.panel in ('math','ordinary'):result=likelihood(backend,tok,p,args,check)
            else:result=scalar(backend,tok,p,args,decoder,check)
            report.update(result,status='COMPLETE_ALL_DECLARED_ATTEMPTS')
        if sources()!=report['source_files']:raise ValueError('Source changed while worker was active')
        report['output_files']={f.name:sha256_file(f) for f in args.output.iterdir() if f.is_file() and f.name!='run.json'}
    except BaseException as error:
        report.update(status='FAILED_INCOMPLETE_REQUIRED_ENDPOINT',error=f'{type(error).__name__}: {error}')
        raise
    finally:
        report.update(finished_utc=utc_now(),wall_seconds=time.monotonic()-started)
        atomic_json(args.output/'run.json',report,overwrite=True)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('prepare','run','collect'))
    for name in ('output','data','model-dir','binding','contract','reference','release','notes'):
        parser.add_argument('--'+name,type=Path,required=name=='output')
    parser.add_argument('--purpose',choices=('science','engineering-train-only'),default='science')
    parser.add_argument('--panel',choices=PANELS);parser.add_argument('--shard',type=int,default=0)
    parser.add_argument('--qualify',action='store_true');parser.add_argument('--deadline-utc')
    args=parser.parse_args()
    if args.command=='prepare':result=prepare(args)
    elif args.command=='collect':result=collect(args)
    else:result=run(args)
    print(json.dumps(result,allow_nan=False),flush=True)


if __name__=='__main__':main()

"""No-training real-FineMath reader feasibility on frozen arithmetic boundaries."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
from fractions import Fraction
import json
import math
from pathlib import Path
import random
import re
import time
from types import SimpleNamespace

import torch

from train.evaluate import asset_provenance, GreedyDecoder, GenerationFailure
from train.evaluate_likelihood import score_continuations
from train.evaluate_local import setup_model
from train.evaluation_integrity import evaluation_dependencies
from train.neuron_backend import CausalCache
from weekend_thinking.artifacts import atomic_json, canonical_hash, require_utc_deadline, sha256_file, utc_now
from weekend_thinking.pretrained import load_base, sampled_fingerprint
from weekend_thinking.stats import paired_cluster_bootstrap

SCHEMA = 'finemath-calculation-boundary-reader-v1'
MODES = ('raw', 'empty', 'correct', 'corrupt', 'shuffled', 'other')
METRICS = ('numeric_nonwhitespace_ll', 'scalar_joint_ll', 'natural16_ll')
CONTRASTS = (('raw', 'empty'), ('raw', 'correct'), ('empty', 'correct'),
             ('corrupt', 'correct'), ('shuffled', 'correct'), ('other', 'correct'))
COUNTS = {'train': 256, 'dev': 64}
WIDTH, FUTURE, PAD, SHARDS = 1024, 16, 151643, 4
ANSWER_CAP = 32
MODEL, REVISION = 'Qwen/Qwen3-1.7B-Base', 'ea980cb0a6c2ae4b936e82123acc929f1cec04c1'


def read(p): return json.loads(Path(p).read_text())
def rows(p): return [json.loads(s) for s in Path(p).read_text().splitlines() if s.strip()]
def check_deadline(value):
    if datetime.now(timezone.utc) >= require_utc_deadline(value):
        raise RuntimeError('Finite probe deadline reached; no automatic retry')


def sources():
    root = Path(__file__).resolve().parents[1]
    return {**evaluation_dependencies(), **{name: sha256_file(root/name) for name in
        ('train/calculation_boundary_probe.py', 'train/calculation_boundaries.py',
         'train/evaluate_local.py', 'train/evaluate_likelihood.py')}}


def decode(tok, ids):
    return tok.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)


def answer_recipe(deadline):
    require_utc_deadline(deadline)
    return dict(split='dev', expected_rows=384, cap=ANSWER_CAP, deadline_utc=deadline,
        policy='single greedy; tokenizer-defined IDs; defined specials preserved; EOS151643 or cap32 only',
        support='backend.banned_ids excludes undefined IDs; native qualification and likelihood use the FULL released head',
        context='Exactly the frozen likelihood context; no suffix, QA/CoT prompt or parser-based stopping',
        parser='initial ASCII signed decimal/fraction v1; exact Fraction; EOS certifies terminal literal, cap needs right lookahead',
        denominator='All64 DEV documents in all six modes; errors/incomplete/unsupported/format failures score zero')


def parse_initial_scalar(text, stop):
    """Target-independent, bounded syntax only; never recover a later number."""
    if not isinstance(text,str) or len(text)>8192: return dict(status='unsupported')
    if stop not in ('length_cap',f'token:{PAD}'): return dict(status='error')
    terminal=stop==f'token:{PAD}'; s=text.lstrip()
    number=r'[+-]?(?:[0-9]+(?:\.[0-9]+)?|\.[0-9]+)'
    match=re.match(rf'({number})(?:[ \t]*/[ \t]*({number}))?',s)
    if not match:
        return dict(status='incomplete' if not terminal and s in ('','+','-','.') else 'format_error')
    literal=match.group(); tail=s[match.end():]; rest=tail.lstrip()
    if len(literal)>128 or sum(c.isdigit() for c in literal)>64: return dict(status='unsupported')
    if not rest:
        if not terminal: return dict(status='incomplete')
    else:
        # A decimal point/comma is a delimiter only with nonnumeric lookahead or EOS.
        if rest[0] in '.,;':
            following=rest[1:]
            if not following.strip():
                if not terminal: return dict(status='incomplete')
                rest=''
            elif following[:1].isspace(): rest=following.lstrip()
            else: return dict(status='unsupported')
        elif not tail[:1].isspace(): return dict(status='unsupported')
        if rest:
            word=re.match(r'[A-Za-z]+\b',rest)
            if (not word or len(word.group())==1 or word.group().lower() in
                    ('plus','minus','times','over','divide','divided','dividing','multiply','multiplied','multiplying',
                     'mod','modulo','squared','square','cubed','cube','power','powers','raised','root','sqrt',
                     'percent','percentage','factorial','choose','point','dot','by','added','subtract','subtracted',
                     'sin','cos','tan','log','ln','and','or','to','e','exp')):
                return dict(status='unsupported')
    try:
        value=Fraction(match[1])/(Fraction(match[2]) if match[2] is not None else 1)
    except (ValueError,ZeroDivisionError): return dict(status='unsupported')
    return dict(status='valid',value=str(value),literal=literal)


def dataset(directory, tok):
    """Verify the complete curator publication; never substitute a smaller split."""
    from train.calculation_boundaries import SCHEMA as DATA_SCHEMA, SOURCE_SHA, SALT
    m = read(directory/'manifest.json'); accepted = rows(directory/'accepted.jsonl')
    if (m['schema'] != DATA_SCHEMA or m['source_sha256'] != SOURCE_SHA or m['selection_salt'] != SALT
            or m['ready_for_feasibility'] is not True or m['tokenizer_vocab_sha256'] != canonical_hash(tok.get_vocab())):
        raise ValueError('Require the exact ready curator/source/tokenizer publication')
    root = Path(__file__).resolve().parents[1]
    if (m['source_files'] != {'train/calculation_boundaries.py': sha256_file(root/'train/calculation_boundaries.py')}
            or sha256_file(m['source_file']) != m['source_sha256']):
        raise ValueError('Exact curator and original source bytes required')
    for name, expected in m['source_files'].items():
        if sha256_file(root/name) != expected: raise ValueError('Curator source changed')
    for name in ('accepted', 'audit'):
        if sha256_file(directory/f'{name}.jsonl') != m[f'{name}_sha256']:
            raise ValueError('Curator accepted/audit bytes changed')
    if (len({r['item_id'] for r in accepted}) != len(accepted)
            or len({r['document_id'] for r in accepted}) != len(accepted)
            or any(r['split'] not in COUNTS for r in accepted)):
        raise ValueError('Unique one-boundary/document TRAIN/DEV rows required')
    selected = []
    for split, count in COUNTS.items():
        candidates = sorted((r for r in accepted if r['split'] == split), key=lambda r: canonical_hash(
            dict(salt=SALT, split=split, document_id=r['document_id'], item_id=r['item_id'])))
        selection = m['selection'][split]
        if (selection['requested'] != count or selection['available'] != len(candidates) or len(candidates) < count
                or selection['selected_item_ids'] != [r['item_id'] for r in candidates[:count]]
                or selection['reserve_item_ids'] != [r['item_id'] for r in candidates[count:]]):
            raise ValueError('Require full256 TRAIN/64 DEV in the curator hash order; no shrink/replacement')
        selected.extend(candidates[:count])
    defined = set(tok.get_vocab().values())
    for r in selected:
        past = r['writer_prefix_ids']; whole = r['ordinary_source_ids']; cut = len(past)
        natural = r['natural_continuation_ids'][:16]; target = r['numeric_target_ids']; rhs = r['verified_numeric_rhs']
        mask = r['numeric_token_mask']; note = r['boundary']['note']
        if (not 128 <= cut <= 512 or len(whole) != 528 or whole != r['original_prefix_ids']+r['original_future_ids']
                or (len(r['original_prefix_ids']), len(r['original_future_ids'])) != (512, 16)
                or past != r['prefix_ids'] or past != whole[:cut] or cut != r['boundary']['cut']
                or r['natural_continuation_ids'] != whole[cut:] or natural != r['natural_future16_ids'] or len(natural) != 16
                or not 1 <= len(target) <= 16 or target != natural[:len(target)]
                or target != rhs['token_ids'] or len(target) != rhs['token_count']
                or mask != rhs['numeric_token_mask'] or len(mask) != len(target)
                or any(type(v) is not bool for v in mask) or not any(mask)
                or mask != [bool(decode(tok, [t]).strip()) for t in target]
                or not any(any(c.isdigit() for c in decode(tok, [t])) for t in target)
                or r['source_sha256'] != m['source_sha256'] or Path(r['source_file']).resolve() != Path(m['source_file']).resolve()
                or r['note_ids'] != note['correct_ids'] or r['corrupted_note_ids'] != note['corrupted_ids']
                or not r['note_ids'] or len(r['note_ids']) != len(r['corrupted_note_ids']) or r['note_ids'] == r['corrupted_note_ids']
                or any(type(t) is not int or t not in defined for t in whole+r['note_ids']+r['corrupted_note_ids'])
                or decode(tok, past) != r['writer_prefix_text'] or r['prefix_text'] != r['writer_prefix_text']
                or not r['writer_prefix_text'].rstrip().endswith('=')
                or tok.encode(r['writer_prefix_text'],add_special_tokens=False) != past
                or decode(tok, whole) != r['writer_prefix_text']+decode(tok,whole[cut:])
                or decode(tok,r['note_ids']) != r['note_text'] or r['note_text'] != note['correct_text']
                or decode(tok,r['corrupted_note_ids']) != r['corrupted_note_text'] or r['corrupted_note_text'] != note['corrupted_text']
                or decode(tok, target) != rhs['serialized_text']):
            raise ValueError('Invalid causal source/span/mask/note contract')
    return m, selected


def make_contexts(selected):
    """Freeze controls globally within each split, before sharding or model use."""
    contexts = {}
    for split in COUNTS:
        group = [r for r in selected if r['split'] == split]
        for i, r in enumerate(group):
            note = list(r['note_ids']); donor = group[(i+1) % len(group)]
            other = donor['note_ids']; shuffled = list(note)
            seed = int(canonical_hash(dict(seed=17, split=split, item_id=r['item_id']))[:15], 16)
            random.Random(seed).shuffle(shuffled)
            matched = (other*((len(note)+len(other)-1)//len(other)))[:len(note)]
            content = dict(empty=[], correct=note, corrupt=r['corrupted_note_ids'], shuffled=shuffled, other=matched)
            past = r['writer_prefix_ids']
            contexts[r['item_id']] = dict(donor_item_id=donor['item_id'], shuffle_seed=seed,
                values=dict(raw=list(past), **{k:list(v)+[271]+past for k,v in content.items()}))
    return contexts


def prepare(output, directory, model_dir, deadline):
    from transformers import AutoTokenizer
    check_deadline(deadline); assets = asset_provenance(model_dir)
    if (assets.get('model_id'), assets.get('revision')) != (MODEL, REVISION): raise ValueError('Actual released B0 required')
    tok = AutoTokenizer.from_pretrained(model_dir, local_files_only=True, trust_remote_code=False)
    if tok.encode('\n\n', add_special_tokens=False) != [271] or tok.eos_token_id != PAD:
        raise ValueError('Native separator/EOS changed')
    manifest, selected = dataset(directory, tok); contexts = make_contexts(selected)
    if any(len(v)+16 > WIDTH for r in contexts.values() for v in r['values'].values()):
        raise ValueError('Over-width context; never truncate an accepted note or source prefix')
    refs = [dict(item_id=next(r for r in selected if r['split']==s and r['item_id']==manifest['selection'][s]['selected_item_ids'][i])['item_id'],
                 split=s, mode=mode) for s,i,mode in (('train',0,'raw'),('train',1,'correct'),('dev',0,'raw'),('dev',1,'correct'))]
    for r in refs:
        r['context_ids'] = contexts[r['item_id']]['values'][r['mode']]
        r['context_sha256'] = canonical_hash(r['context_ids'])
    check_deadline(deadline); output.mkdir(parents=True, exist_ok=False)
    atomic_json(output/'inputs.json', dict(rows=selected, contexts=contexts, qualification_refs=refs))
    p = dict(schema=SCHEMA, created_utc=utc_now(), source_files=sources(), model_dir=str(model_dir.resolve()), assets=assets,
        dataset=str(directory.resolve()), inputs_sha256=sha256_file(output/'inputs.json'),
        input_hashes={str(f.resolve()):sha256_file(f) for f in (directory/'manifest.json', directory/'accepted.jsonl', directory/'audit.jsonl', Path(manifest['source_file']))},
        item_ids=[r['item_id'] for r in selected], counts=COUNTS, modes=list(MODES), metrics=list(METRICS),
        compute='fp32', vocab_size=read(model_dir/'config.json')['vocab_size'], width=WIDTH, future_width=16, pad_id=PAD, separator_ids=[271], shards=4, expected_rows=1920,
        serialization='raw=exact causal past128..512; otherwise note+[271]+same full past; empty=[271]+past',
        primary='numeric_nonwhitespace_ll: sum native first-n token logps excluding pure-whitespace tokens',
        secondary='Full scalar joint LL and full natural16 LL; one natural16 forward/cell, same actual source spelling',
        qualification='Native full-head log-softmax first TRAIN raw/second TRAIN correct/first DEV raw/second DEV correct; shard i checks ref i before scores',
        forward_atol=.002, answers=answer_recipe(deadline), bootstrap=dict(unit='document', repeats=5000, seed=17),
        limitations=manifest['limits']+['External exact arithmetic feasibility, not student-generated thought or training benefit.',
            'Source-verified accepted boundaries are a curated subset; all preparer failures/reserves remain in the pinned audit.',
            'TRAIN and DEV contrasts are separate, conditional fixed-document estimates, not training-seed uncertainty.',
            'Teacher-forced likelihood and argmax-on-gold-prefix are not free-running answer accuracy.',
            'Excluding pure-whitespace tokens does not split whitespace fused into a numeric token.',
            'Numeric nonwhitespace LL conditions on observed whitespace and earlier numeric tokens; it is not a whitespace-marginalized probability.'])
    atomic_json(output/'protocol.json', p); return p


def load(output):
    p = read(output/'protocol.json')
    if (p['schema'] != SCHEMA or p['source_files'] != sources() or p['counts'] != COUNTS
            or p['modes'] != list(MODES) or p['metrics'] != list(METRICS) or p['compute'] != 'fp32'
            or p['answers'] != answer_recipe(p['answers']['deadline_utc']) or p['separator_ids'] != [271] or p['pad_id'] != PAD or p['forward_atol'] != .002
            or (p['width'],p['future_width'],p['shards'],p['expected_rows']) != (1024,16,4,1920)):
        raise ValueError('Probe recipe/source changed')
    for f,h in [(output/'inputs.json',p['inputs_sha256']), *p['input_hashes'].items(),
                (Path(p['model_dir'])/'ASSET_MANIFEST.json',p['assets']['manifest_sha256'])]:
        if sha256_file(f) != h: raise ValueError('Frozen source/model/input changed')
    x = read(output/'inputs.json')
    if len(x['rows']) != 320 or [r['item_id'] for r in x['rows']] != p['item_ids']:
        raise ValueError('Complete immutable320-document inventory required')
    return p,x


@torch.no_grad()
def cpu_reference(output, threads, deadline):
    if type(threads) is not int or not 1 <= threads <= 16: raise ValueError('CPU threads1..16 required')
    check_deadline(deadline); p,x = load(output); torch.set_num_threads(threads)
    if asset_provenance(Path(p['model_dir'])) != p['assets']: raise ValueError('Released model bytes changed')
    before=time.monotonic(); model,_ = load_base(p['model_dir']); model.eval(); setup_seconds=time.monotonic()-before
    if model.get_input_embeddings().weight is not model.get_output_embeddings().weight or any(t.dtype!=torch.float32 for t in model.parameters()):
        raise ValueError('Tied FP32 actual B0 required')
    d = output/'cpu-reference'; d.mkdir(exist_ok=False); values=[]; timings=[]
    for r in x['qualification_refs']:
        check_deadline(deadline); start=time.monotonic()
        h=model.model(input_ids=torch.tensor([r['context_ids']]),use_cache=False).last_hidden_state
        lp=model.lm_head(h[:,-1]).float().log_softmax(-1).cpu()
        if not torch.isfinite(lp).all(): raise ValueError('Nonfinite native full-head reference')
        values.append(lp); timings.append(time.monotonic()-start)
    value=torch.cat(values); torch.save(value,d/'next-logps.pt')
    atomic_json(d/'reference.json',dict(status='complete',protocol_sha256=sha256_file(output/'protocol.json'),
        signature=sampled_fingerprint(model),assets_sha256=canonical_hash(p['assets']),shape=list(value.shape),dtype='float32',tied=True,
        model_setup_wall_seconds=setup_seconds,
        refs=x['qualification_refs'],logps_sha256=sha256_file(d/'next-logps.pt'),forward_wall_seconds=timings,finished_utc=utc_now()))


def reference(output,p,x):
    r=read(output/'cpu-reference/reference.json'); t=torch.load(output/'cpu-reference/next-logps.pt',weights_only=True,map_location='cpu')
    if (r['status']!='complete' or r['protocol_sha256']!=sha256_file(output/'protocol.json') or not r['tied']
            or r['dtype']!='float32' or r['refs']!=x['qualification_refs'] or list(t.shape)!=r['shape']
            or r['assets_sha256']!=canonical_hash(p['assets']) or r['signature']['parameters']!=1720574976
            or list(t.shape)!=[4,p['vocab_size']] or t.dtype!=torch.float32 or not torch.isfinite(t).all()
            or r['logps_sha256']!=sha256_file(output/'cpu-reference/next-logps.pt')):
        raise ValueError('Complete native reference required')
    return r,t


def qualified_reader(output,p,x,ref,native,shard,d,report,*,generation=False):
    if asset_provenance(Path(p['model_dir']))!=p['assets']: raise ValueError('Released model bytes changed')
    start=time.monotonic()
    backend,tok,loaded,signature,_=setup_model(SimpleNamespace(model_dir=Path(p['model_dir']),checkpoint=None,device='xla'))
    report['model_setup_wall_seconds']=time.monotonic()-start; check_deadline(report['deadline_utc'])
    if (loaded is not None or signature!=ref['signature'] or any(t.dtype!=torch.float32 for t in backend.model.parameters())
            or backend.model.get_input_embeddings().weight is not backend.model.get_output_embeddings().weight):
        raise ValueError('Tied actual native FP32 B0 required')
    report['model_signature']=signature; q=x['qualification_refs'][shard]; start=time.monotonic(); decoder=None
    ids=torch.full((1,WIDTH),PAD,dtype=torch.long); ids[0,:len(q['context_ids'])]=torch.tensor(q['context_ids']); ids=ids.to(backend.device)
    if generation:
        banned=tuple(sorted(set(range(p['vocab_size']))-set(tok.get_vocab().values())))
        if tuple(backend.banned_ids)!=banned: raise ValueError('Tokenizer-defined generation support changed')
        report['banned_ids_sha256']=canonical_hash(banned)
        decoder=GreedyDecoder(backend.model,tok,sync=backend.sync,banned_ids=tuple(backend.banned_ids))
        logits=decoder._next(ids,torch.arange(WIDTH,device=backend.device),
            CausalCache(backend.model.config.num_hidden_layers,WIDTH+ANSWER_CAP),
            torch.tensor([len(q['context_ids'])-1],device=backend.device))
    else:
        h=backend.hidden(ids)
        if h.dtype!=torch.float32: raise ValueError('Actual FP32 trunk required')
        logits=backend.model.lm_head(h[:,len(q['context_ids'])-1])
    actual=logits.float().log_softmax(-1); backend.sync(); actual=actual.cpu(); torch.save(actual,d/'forward-logps.pt')
    error=(actual-native[shard:shard+1]).abs().max().item()
    report['forward_check']=dict(ref=q,max_error=error if math.isfinite(error) else str(error),atol=.002,
        forward_logps_sha256=sha256_file(d/'forward-logps.pt'),wall_seconds=time.monotonic()-start)
    if not math.isfinite(error) or error>.002: raise RuntimeError('Native CPU/device next-LP gate failed')
    atomic_json(d/'run.json',report,overwrite=True); return backend,tok,decoder


def verify_gate(d,r,p,x,ref,native,shard):
    gate=r['forward_check']; actual=torch.load(d/'forward-logps.pt',weights_only=True,map_location='cpu')
    if list(actual.shape)!=[1,p['vocab_size']] or actual.dtype!=torch.float32 or not torch.isfinite(actual).all():
        raise ValueError('Invalid actual qualification vector')
    error=(actual-native[shard:shard+1]).abs().max().item()
    if (gate['ref']!=x['qualification_refs'][shard] or gate['atol']!=.002 or not math.isfinite(error)
            or error!=gate['max_error'] or error>.002 or gate['forward_logps_sha256']!=sha256_file(d/'forward-logps.pt')
            or r['model_signature']!=ref['signature']): raise ValueError('Actual native/device qualification evidence differs')


@torch.no_grad()
def score(output,shard,deadline):
    if type(shard) is not int or shard not in range(4): raise ValueError('Four shards only')
    check_deadline(deadline); p,x=load(output); ref,native=reference(output,p,x)
    d=output/'scores'/f'shard-{shard}'; d.mkdir(parents=True,exist_ok=False)
    report=dict(status='running',shard=shard,rows=0,deadline_utc=deadline,protocol_sha256=sha256_file(output/'protocol.json'),started_utc=utc_now())
    atomic_json(d/'run.json',report)
    try:
        backend,_,_=qualified_reader(output,p,x,ref,native,shard,d,report)
        with (d/'scores.jsonl').open('x') as stream:
            for r in x['rows'][shard::4]:
                for mode in MODES:
                    check_deadline(deadline); start=time.monotonic(); context=x['contexts'][r['item_id']]['values'][mode]
                    v=dict(item_id=r['item_id'],document_id=r['document_id'],split=r['split'],mode=mode,context_sha256=canonical_hash(context))
                    try:
                        value=score_continuations(backend,context,[r['natural_future16_ids']],total_width=WIDTH,future_width=16,pad_id=PAD)[0]
                        v.update(status='ok',**asdict(value),**derive_metrics(value.token_log_likelihoods,r['numeric_token_mask']))
                    except Exception as e:
                        v.update(status='error',error=f'{type(e).__name__}: {e}',wall_seconds=time.monotonic()-start)
                        stream.write(json.dumps(v,allow_nan=False)+'\n'); stream.flush(); report['rows']+=1; raise
                    v['wall_seconds']=time.monotonic()-start; stream.write(json.dumps(v,allow_nan=False)+'\n'); stream.flush(); report['rows']+=1
                print(json.dumps(dict(shard=shard,rows=report['rows'],utc=utc_now())),flush=True)
        report.update(status='complete',scores_sha256=sha256_file(d/'scores.jsonl'))
    except BaseException as e: report.update(status='failed',error=f'{type(e).__name__}: {e}'); raise
    finally: report['finished_utc']=utc_now(); atomic_json(d/'run.json',report,overwrite=True)


def derive_metrics(logps,mask):
    if (len(logps)!=16 or not all(math.isfinite(v) for v in logps) or not 1<=len(mask)<=16
            or any(type(v) is not bool for v in mask) or not any(mask)):
        raise ValueError('Finite16-token vector and nonempty exact numeric mask required')
    return dict(numeric_nonwhitespace_ll=sum(v for v,on in zip(logps,mask) if on),
                scalar_joint_ll=sum(logps[:len(mask)]),natural16_ll=sum(logps))


@torch.no_grad()
def answers(output,shard,deadline):
    if type(shard) is not int or shard not in range(4): raise ValueError('Four shards only')
    check_deadline(deadline); p,x=load(output); ref,native=reference(output,p,x)
    if deadline!=p['answers']['deadline_utc']: raise ValueError('Frozen answer deadline required')
    selected=[r for r in x['rows'] if r['split']=='dev'][shard::4]
    d=output/'answers'/f'shard-{shard}'; d.mkdir(parents=True,exist_ok=False)
    report=dict(status='running',shard=shard,rows=0,deadline_utc=deadline,
        protocol_sha256=sha256_file(output/'protocol.json'),started_utc=utc_now())
    atomic_json(d/'run.json',report)
    try:
        _,tok,decoder=qualified_reader(output,p,x,ref,native,shard,d,report,generation=True)
        with (d/'predictions.jsonl').open('x') as stream:
            for r in selected:
                for mode in MODES:
                    check_deadline(deadline); start=time.monotonic(); context=x['contexts'][r['item_id']]['values'][mode]
                    v=dict(item_id=r['item_id'],document_id=r['document_id'],split='dev',mode=mode,status='ok',
                        context_sha256=canonical_hash(context),actions=[],content_ids=[],generated_text='',stop_type='error')
                    try:
                        result=decoder.decode(context,max_new_tokens=ANSWER_CAP,prefill_width=WIDTH,
                            pad_id=PAD,stop_ids=(PAD,),answer_line_stop=False)
                    except Exception as e:
                        v.update(status='error',error=f'{type(e).__name__}: {e}')
                        result=e.partial if isinstance(e,GenerationFailure) else None
                    if result is not None:
                        v.update(actions=result.actions,content_ids=result.content_ids,generated_text=result.raw_text,stop_type=result.stop_type)
                    v.update(raw_action_text=decode(tok,v['actions']),n_generated_tokens=len(v['actions']),
                        parsed=parse_initial_scalar(v['generated_text'],v['stop_type']) if v['status']=='ok' else dict(status='error'),
                        wall_seconds=time.monotonic()-start)
                    stream.write(json.dumps(v,allow_nan=False)+'\n'); stream.flush(); report['rows']+=1
                print(json.dumps(dict(shard=shard,rows=report['rows'],utc=utc_now())),flush=True)
        report.update(status='complete',predictions_sha256=sha256_file(d/'predictions.jsonl'))
    except BaseException as e: report.update(status='failed',error=f'{type(e).__name__}: {e}'); raise
    finally: report['finished_utc']=utc_now(); atomic_json(d/'run.json',report,overwrite=True)


def summarize_answers(output):
    from transformers import AutoTokenizer
    p,x=load(output); ref,native=reference(output,p,x); values=[]
    tok=AutoTokenizer.from_pretrained(p['model_dir'],local_files_only=True,trust_remote_code=False)
    defined=set(tok.get_vocab().values()); banned=tuple(sorted(set(range(p['vocab_size']))-defined))
    selected=[r for r in x['rows'] if r['split']=='dev']; by_id={r['item_id']:r for r in selected}
    for shard in range(4):
        d=output/'answers'/f'shard-{shard}'; r=read(d/'run.json'); v=rows(d/'predictions.jsonl')
        if (r['status']!='complete' or r['shard']!=shard or r['rows']!=96 or len(v)!=96
                or r['deadline_utc']!=p['answers']['deadline_utc'] or r['protocol_sha256']!=sha256_file(output/'protocol.json')
                or r['predictions_sha256']!=sha256_file(d/'predictions.jsonl') or r['banned_ids_sha256']!=canonical_hash(banned)
                or {(q['item_id'],q['mode']) for q in v}!={(q['item_id'],m) for q in selected[shard::4] for m in MODES}):
            raise ValueError('All384 unique complete DEV attempts required')
        verify_gate(d,r,p,x,ref,native,shard)
        for q in v:
            item=by_id[q['item_id']]; actions=q['actions']; stop=q['stop_type']; error=q['status']=='error'
            content=actions[:-1] if stop==f'token:{PAD}' else actions
            parsed=dict(status='error') if error else parse_initial_scalar(q['generated_text'],stop)
            if (q['status'] not in ('ok','error') or q['split']!='dev' or q['document_id']!=item['document_id']
                    or q['context_sha256']!=canonical_hash(x['contexts'][q['item_id']]['values'][q['mode']])
                    or any(type(t) is not int or t not in defined for t in actions) or len(actions)>ANSWER_CAP
                    or q['n_generated_tokens']!=len(actions) or q['content_ids']!=content
                    or q['raw_action_text']!=decode(tok,actions) or q['generated_text']!=decode(tok,content)
                    or q['parsed']!=parsed or not math.isfinite(q['wall_seconds']) or q['wall_seconds']<0
                    or (error and (stop!='error' or not q.get('error')))
                    or (not error and (not actions or (stop==f'token:{PAD}' and actions[-1]!=PAD)
                        or PAD in content or stop not in ('length_cap',f'token:{PAD}')
                        or (stop=='length_cap' and len(actions)!=ANSWER_CAP)))):
                raise ValueError('Changed answer actions, context, parser, stop or retained failure')
            correct=parsed['status']=='valid' and Fraction(parsed['value'])==Fraction(item['verified_numeric_rhs']['value'])
            q.update(score=int(correct),value_status=('correct' if correct else 'wrong_value') if parsed['status']=='valid' else parsed['status'])
        values.extend(v)
    groups={m:[dict(item_id=q['item_id'],document_id=q['document_id'],score=q['score']) for q in values if q['mode']==m] for m in MODES}
    result=dict(schema=SCHEMA,status='complete',protocol_sha256=sha256_file(output/'protocol.json'),rows=384,
        endpoint=p['answers'],absolute={m:dict(correct=sum(q['score'] for q in values if q['mode']==m),denominator=64,
            accuracy=sum(q['score'] for q in values if q['mode']==m)/64,
            statuses={s:sum(q['mode']==m and q['value_status']==s for q in values) for s in
                ('correct','wrong_value','format_error','unsupported','incomplete','error')}) for m in MODES},
        contrasts={f'{b}_minus_{a}':paired_cluster_bootstrap(groups[a],groups[b],seed=17,repeats=5000) for a,b in CONTRASTS},
        generated_tokens=sum(q['n_generated_tokens'] for q in values),generation_wall_seconds=sum(q['wall_seconds'] for q in values),
        answer_rows=values,limitations=p['limitations']+['Free initial scalar value accuracy, not argmax on gold or full continuation correctness.',
            'Unsupported syntax and generation errors count as zero; a valid wrong scalar does not identify the internal cause.'],created_utc=utc_now())
    atomic_json(output/'answers.summary.json',result); return result


def summarize(output):
    p,x=load(output); ref,native=reference(output,p,x); by_id={r['item_id']:r for r in x['rows']}; values=[]; signatures=[]
    for shard in range(4):
        d=output/'scores'/f'shard-{shard}'; r=read(d/'run.json'); v=rows(d/'scores.jsonl')
        expected={(q['item_id'],m) for q in x['rows'][shard::4] for m in MODES}
        if (r['status']!='complete' or r['rows']!=480 or len(v)!=480 or r['shard']!=shard
                or r['protocol_sha256']!=sha256_file(output/'protocol.json') or r['scores_sha256']!=sha256_file(d/'scores.jsonl')
                or {(q['item_id'],q['mode']) for q in v}!=expected): raise ValueError('All1920 unique complete cells required')
        verify_gate(d,r,p,x,ref,native,shard)
        for q in v:
            item=by_id[q['item_id']]; correct=derive_metrics(q['token_log_likelihoods'],item['numeric_token_mask'])
            if (q['status']!='ok' or q['document_id']!=item['document_id'] or q['split']!=item['split']
                    or q['target_token_ids']!=item['natural_future16_ids'] or q['context_sha256']!=canonical_hash(x['contexts'][q['item_id']]['values'][q['mode']])
                    or any(q[k]!=v for k,v in correct.items()) or q['log_likelihood']!=correct['natural16_ll']):
                raise ValueError('Nonfinite/changed target, pairing or metric derivation')
        signatures.append(r['model_signature']); values.extend(v)
    if len({canonical_hash(s) for s in signatures})!=1: raise ValueError('Reader model identities differ')
    result={}
    for split in COUNTS:
        result[split]={}
        for metric in METRICS:
            groups={m:[dict(item_id=v['item_id'],document_id=v['document_id'],score=v[metric]) for v in values if v['split']==split and v['mode']==m] for m in MODES}
            result[split][metric]=dict(absolute_means={m:sum(v['score'] for v in g)/COUNTS[split] for m,g in groups.items()},
                contrasts={f'{b}_minus_{a}':paired_cluster_bootstrap(groups[a],groups[b],seed=17,repeats=5000) for a,b in CONTRASTS})
    answer=dict(schema=SCHEMA,status='complete',protocol_sha256=sha256_file(output/'protocol.json'),rows=1920,
        by_split=result,score_rows=values,limitations=p['limitations'],created_utc=utc_now())
    atomic_json(output/'summary.json',answer); return answer


def main():
    a=argparse.ArgumentParser(description=__doc__); a.add_argument('command',choices=('prepare','cpu-reference','score','answers','summarize','summarize-answers'))
    a.add_argument('--output',type=Path,required=True); a.add_argument('--dataset',type=Path)
    a.add_argument('--model-dir',type=Path,default=Path('local/assets/qwen3-1.7b-base')); a.add_argument('--cpu-threads',type=int,default=16)
    a.add_argument('--shard-index',type=int); a.add_argument('--deadline-utc'); p=a.parse_args()
    if not p.command.startswith('summarize') and not p.deadline_utc: a.error('Explicit finite --deadline-utc required')
    if p.command=='prepare': prepare(p.output,p.dataset,p.model_dir,p.deadline_utc)
    elif p.command=='cpu-reference': cpu_reference(p.output,p.cpu_threads,p.deadline_utc)
    elif p.command=='score': score(p.output,p.shard_index,p.deadline_utc)
    elif p.command=='answers': answers(p.output,p.shard_index,p.deadline_utc)
    elif p.command=='summarize': summarize(p.output)
    else: summarize_answers(p.output)


if __name__=='__main__': main()

"""Finite V30 references, stochastic qualification and candidate execution.

Reuses the existing cached sampler and span scorer. No optimizer, policy update,
future-aware candidate choice, or job scheduler is implemented here.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time
from types import SimpleNamespace

import torch

from train.candidate_utility_v30 import item_seed
from train.real_trace_contexts import markers, question_projection, reader_context, writer_prompt
from weekend_thinking.artifacts import atomic_json, canonical_hash, sha256_file, utc_now
from weekend_thinking.core import Format, policy_log_probs

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = Path('/home/ubuntu/dario/local/assets/qwen3-1.7b-base')
DEPENDENCIES = ('train/candidate_utility_v30.py', 'train/candidate_utility_v30_driver.py',
    'train/neuron_backend.py', 'train/document_qa_span_scores.py', 'train/real_trace_contexts.py',
    'train/real_cpt_readout.py', 'train/real_cpt_scaffold.py', 'train/real_trace_packing.py',
    'train/evaluate_local.py', 'train/evaluate.py', 'train/useful_note_precision.py',
    'train/real_cpt_readout_driver.py', 'train/calculation_boundaries.py',
    'train/symbolic_note_labels.py', 'train/expression_identity.py',
    'train/runtime_compat.py', 'weekend_thinking/core.py', 'weekend_thinking/pretrained.py',
    'weekend_thinking/backend_contract.py', 'weekend_thinking/artifacts.py')


def read(path):
    return json.loads(Path(path).read_text())


def rows(path):
    return [json.loads(s) for s in Path(path).open() if s.strip()]


def sources():
    return {name: sha256_file(ROOT/name) for name in DEPENDENCIES}


def protocol(directory):
    p = read(directory/'protocol.json')
    assert p['status'] == 'FROZEN_V30_BEFORE_MODEL_CALLS'
    assert sha256_file(p['design_file']) == p['design_sha256']
    for name, digest in p['files'].items():
        assert sha256_file(directory/name) == digest
    assert sha256_file(p['binding_file']) == p['binding_sha256']
    return p


def load_model(p, device, model_id):
    from train.evaluate_local import setup_model
    from train.real_cpt_readout_driver import check_model
    b = p['model'] if model_id == 'T' else dict(id='B0', checkpoint=None)
    backend, tok, loaded, signature, seed = setup_model(SimpleNamespace(
        model_dir=MODEL_DIR, checkpoint=Path(b['checkpoint']) if b['checkpoint'] else None,
        device=device))
    check_model(loaded, b)
    return backend, tok, signature, seed


def check_deadline(deadline):
    if datetime.now(timezone.utc) >= datetime.fromisoformat(deadline):
        raise TimeoutError('Finite V30 deadline reached; no automatic retry')


def write_line(stream, value):
    stream.write(json.dumps(value, allow_nan=False)+'\n'); stream.flush()


@torch.no_grad()
def reference(a):
    p = protocol(a.protocol); a.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(16)
    backend, tok, signature, _ = load_model(p, 'cpu', a.model)
    questions = rows(a.protocol/'qualification-questions.jsonl')
    labels = rows(a.protocol/'qualification-labels.jsonl')
    from train.real_cpt_readout import controls
    native = []; inputs = []
    for i, q in enumerate(questions):
        check_deadline(a.deadline_utc)
        if a.model == 'T':
            context, target = writer_prompt(q, tok), []
        else:
            context, target = q['prefix_ids'], labels[i]['continuation16_ids']
        ids = torch.tensor([context+target]); z = backend.model(ids).logits[0].float()
        lp = z.log_softmax(-1)
        native.append(lp[len(context)-1].cpu())
        inputs.append(dict(item_id=q['item_id'], context_ids=context, target_ids=target,
            HF_target_logps=lp[len(context)-1:len(context)+len(target)-1].gather(
                -1, ids[0,len(context):,None])[:,0].tolist() if target else []))
    spans = []
    for i, q in enumerate(questions[:4]):
        check_deadline(a.deadline_utc)
        mode = ('raw', 'empty', 'oracle', 'corrupt')[i] if a.model == 'T' else 'raw'
        if a.model == 'T':
            contexts = controls(q, tok, dict(content_ids=[]), dict(content_ids=[]))
            context = contexts[mode]
        else: context = q['prefix_ids']
        future = labels[i]['continuation16_ids']; ids = torch.tensor([context+future])
        lp = backend.model(ids).logits[0,len(context)-1:-1].float().log_softmax(-1)
        spans.append(dict(item_id=q['item_id'], mode=mode, context_ids=context,
            target_ids=future, HF_target_logps=lp.gather(-1,ids[0,len(context):,None])[:,0].tolist()))
    torch.save(torch.stack(native), a.output/'native.pt')
    atomic_json(a.output/'inputs.json', dict(native=inputs, spans=spans))
    atomic_json(a.output/'reference.json', dict(status='V30_INDEPENDENT_HF_REFERENCE', model=a.model,
        protocol_sha256=sha256_file(a.protocol/'protocol.json'), sources=sources(), signature=signature,
        native_sha256=sha256_file(a.output/'native.pt'), inputs_sha256=sha256_file(a.output/'inputs.json'),
        completed_utc=utc_now()))
    print('REFERENCE_COMPLETE', a.model, flush=True)


@torch.no_grad()
def native_gate(backend, reference_dir, output):
    from train.document_qa_span_scores import score_spans
    from train.useful_note_precision import compare_distributions
    p = read(reference_dir/'reference.json')
    assert p['sources'] == sources()
    assert sha256_file(reference_dir/'native.pt') == p['native_sha256']
    assert sha256_file(reference_dir/'inputs.json') == p['inputs_sha256']
    inputs = read(reference_dir/'inputs.json'); expected = torch.load(reference_dir/'native.pt',weights_only=True)
    actual = []
    for r in inputs['native'][:2]:
        x = torch.full((1,672),151643,dtype=torch.long)
        x[0,:len(r['context_ids'])] = torch.tensor(r['context_ids'])
        h = backend.hidden(x.to(backend.device))[:,len(r['context_ids'])-1]
        backend.sync(); z = backend.model.lm_head(h).float().log_softmax(-1)
        backend.sync(); actual.append(z.cpu()[0])
    checks = compare_distributions(expected[:2], torch.stack(actual))
    assert all(v['passed'] and v['max_logp_error'] <= .001 for v in checks)
    spans = inputs['spans']
    scored = score_spans(backend,[r['context_ids'] for r in spans],
        [r['target_ids'] for r in spans],width=672,target_slots=16)
    error = max(abs(x-y) for r,s in zip(spans,scored)
                for x,y in zip(r['HF_target_logps'],s['token_logps']))
    assert error <= .001
    atomic_json(output/'native-gate.json',dict(distributions=checks,max_span_token_error=error,
        reference_sha256=sha256_file(reference_dir/'reference.json')))


def format_for(tok):
    opening, closing = markers(tok)
    return Format(open_ids=tuple(opening), close_ids=tuple(closing), stop_id=closing[0],
                  pad_id=151643, additional_stop_ids=(151643,))


@torch.no_grad()
def sample_question(backend, tok, set_seed, q, *, greedy):
    fmt = format_for(tok); seed = item_seed(q['item_id']); set_seed(seed)
    started = time.monotonic()
    rollout = backend.sample(torch.tensor([q['prefix_ids']]), fmt=fmt, group_size=1 if greedy else 8,
        max_actions=128,temperature=.7,policy_version=1280,greedy=greedy)
    backend.sync()
    actions, mask, logps = (v.cpu() for v in (rollout.actions,rollout.action_mask,rollout.behavior_logps))
    elapsed = time.monotonic()-started; result = []
    for i in range(actions.shape[1]):
        n = int(mask[0,i].sum()); chosen = actions[0,i,:n].tolist()
        stopped = chosen and chosen[-1] in (fmt.stop_id,151643)
        content = chosen[:-1] if stopped else chosen
        assert chosen and (stopped or n == 128)
        result.append(dict(item_id=q['item_id'],document_id=q['document_id'],
            mode='greedy' if greedy else f'sample{i}',status='ok',actions=chosen,content_ids=content,
            behavior_logps=logps[0,i,:n].tolist(),action_mask=mask[0,i].tolist(),
            text=tok.decode(content,skip_special_tokens=False,clean_up_tokenization_spaces=False),
            stop_type=f'token:{chosen[-1]}' if stopped else 'length_cap',
            seed=seed,group_size=actions.shape[1],temperature=.7,
            prompt_sha256=canonical_hash(writer_prompt(q,tok)),decode_iterations=rollout.decode_iterations,
            group_seconds=elapsed,allocated_share_seconds=elapsed/actions.shape[1]))
    return result


@torch.no_grad()
def qualify(a):
    p = protocol(a.protocol); a.output.mkdir(parents=True, exist_ok=False)
    backend,tok,signature,set_seed = load_model(p,'xla','T')
    assert signature == read(a.reference/'reference.json')['signature']
    native_gate(backend,a.reference,a.output)
    from train.neuron_backend import CausalCache
    from train.useful_note_precision import compare_distributions
    qs = rows(a.protocol/'qualification-questions.jsonl'); fmt = format_for(tok)
    prompt = backend._prompt(torch.tensor([qs[0]['prefix_ids']]),fmt).repeat_interleave(8,0)
    cache = CausalCache(backend.model.config.num_hidden_layers,prompt.shape[-1]+128)
    z = backend._next(prompt,torch.arange(prompt.shape[-1],device=backend.device),cache)
    backend.sync(); expected = torch.load(a.reference/'native.pt',weights_only=True)[0:1].expand(8,-1)
    checks = compare_distributions(expected,z.cpu().log_softmax(-1))
    assert all(v['passed'] and v['max_logp_error'] <= .001 for v in checks)
    atomic_json(a.output/'cached-prefill.json',checks)
    del cache,prompt,z
    all_notes = []
    with (a.output/'notes.jsonl').open('x') as stream:
        for q in qs:
            check_deadline(a.deadline_utc)
            for greedy in (True,False):
                notes = sample_question(backend,tok,set_seed,q,greedy=greedy)
                for note in notes: write_line(stream,note)
                all_notes.extend(notes)
            print('QUALIFICATION_GENERATED',q['item_id'],flush=True)
    replay = sample_question(backend,tok,set_seed,qs[0],greedy=False)
    original = [r for r in all_notes if r['item_id'] == qs[0]['item_id'] and r['mode'] != 'greedy']
    assert all(a['actions'] == b['actions'] and a['behavior_logps'] == b['behavior_logps'] for a,b in zip(original,replay))
    atomic_json(a.output/'sampling-complete.json',dict(status='V30_SAMPLING_QUALIFICATION_PENDING_CPU_BEHAVIOR_CHECK',
        sources=sources(),protocol_sha256=sha256_file(a.protocol/'protocol.json'),
        notes_sha256=sha256_file(a.output/'notes.jsonl'),RNG_same_seed_exact_reproduction=True,
        signature=signature,environment={k:os.environ.get(k) for k in ('NEURON_RT_VISIBLE_CORES','XLA_COMPILATION_CACHE_SIZE')},
        completed_utc=utc_now()))


@torch.no_grad()
def verify(a):
    p = protocol(a.protocol); a.output.mkdir(parents=True, exist_ok=False)
    complete = read(a.qualification/'sampling-complete.json')
    assert complete['sources'] == sources() and complete['notes_sha256'] == sha256_file(a.qualification/'notes.jsonl')
    backend,tok,_,_ = load_model(p,'cpu','T'); torch.set_num_threads(16)
    qs = {r['item_id']:r for r in rows(a.protocol/'qualification-questions.jsonl')}
    notes = [r for r in rows(a.qualification/'notes.jsonl') if r['mode'] == 'sample0']
    assert len(notes) == 8; errors = []
    for r in notes:
        check_deadline(a.deadline_utc)
        context = writer_prompt(qs[r['item_id']],tok); actions = r['actions']
        ids = torch.tensor([context+actions]); z = backend.model(ids).logits[0,len(context)-1:-1].float()
        lp = policy_log_probs(z,.7,backend.banned_ids).gather(-1,ids[0,len(context):,None])[:,0]
        error = max(abs(x-y) for x,y in zip(lp.tolist(),r['behavior_logps']))
        errors.append(dict(item_id=r['item_id'],tokens=len(actions),max_behavior_logp_error=error))
        assert error <= .001
    atomic_json(a.output/'qualification.json',dict(status='PASS_V30_ACTUAL_MODEL_STOCHASTIC_CACHE_NATIVE_BEHAVIOR_AND_RNG',
        sampling_complete_sha256=sha256_file(a.qualification/'sampling-complete.json'),
        protocol_sha256=sha256_file(a.protocol/'protocol.json'),sources=sources(),checks=errors,
        limits='FP32 actual-model sampling/scoring only; no BF16, optimizer, learning or capability claim.'))
    print('QUALIFICATION_PASS',flush=True)


def released(a):
    p = protocol(a.protocol); release = read(a.release)
    assert release['status'] == 'RELEASED_ONE_BOUNDED_V30_DIAGNOSTIC_NO_LEARNING'
    assert release['sources'] == sources()
    assert release['protocol_sha256'] == sha256_file(a.protocol/'protocol.json')
    assert sha256_file(release['qualification_file']) == release['qualification_sha256']
    assert read(release['qualification_file'])['status'] == 'PASS_V30_ACTUAL_MODEL_STOCHASTIC_CACHE_NATIVE_BEHAVIOR_AND_RNG'
    check_deadline(release['deadline_utc'])
    return p


@torch.no_grad()
def generate(a):
    p = released(a); a.output.mkdir(parents=True,exist_ok=False)
    backend,tok,signature,set_seed = load_model(p,'xla','T')
    assert signature == read(a.reference/'reference.json')['signature']
    native_gate(backend,a.reference,a.output)
    questions = rows(a.protocol/'diagnostic-questions.jsonl')
    assert len(questions) == 256 and all(set(q) == {'item_id','document_id','prefix_ids'} for q in questions)
    attempt_count = 0; started = time.monotonic()
    with (a.output/'notes.jsonl').open('x') as stream:
        for index,q in enumerate(questions):
            if index % 4 != a.shard: continue
            check_deadline(a.deadline_utc)
            for greedy in (True,False):
                try:
                    notes = sample_question(backend,tok,set_seed,q,greedy=greedy)
                except Exception as exc:
                    # The original partial journal remains. No made-up actions,
                    # replacement draw or numerical utility is substituted.
                    write_line(stream,dict(item_id=q['item_id'],document_id=q['document_id'],
                        status='error',phase='greedy' if greedy else 'sample8',error=repr(exc)))
                    raise
                for note in notes: write_line(stream,note); attempt_count += 1
            if (index // 4 + 1) % 8 == 0:
                print('GENERATED_DOCUMENTS',index // 4 + 1,'SECONDS',time.monotonic()-started,flush=True)
    assert attempt_count == 64*9
    atomic_json(a.output/'complete.json',dict(status='COMPLETE_V30_ORIGINAL_CANDIDATE_SHARD',
        shard=a.shard,attempts=attempt_count,sources=sources(),
        protocol_sha256=sha256_file(a.protocol/'protocol.json'),
        notes_sha256=sha256_file(a.output/'notes.jsonl'),seconds=time.monotonic()-started))


@torch.no_grad()
def score(a):
    p = released(a); a.output.mkdir(parents=True,exist_ok=False)
    backend,tok,signature,_ = load_model(p,'xla',a.model)
    assert signature == read(a.reference/'reference.json')['signature']
    native_gate(backend,a.reference,a.output)
    labels = rows(a.protocol/'diagnostic-labels.jsonl')
    if a.model == 'T':
        by_id = {}
        for shard in range(4):
            directory = a.candidates/f'generate-{shard}'
            complete = read(directory/'complete.json')
            assert complete['sources'] == sources()
            assert complete['notes_sha256'] == sha256_file(directory/'notes.jsonl')
            for r in rows(directory/'notes.jsonl'):
                assert r['status'] == 'ok'
                slot = by_id.setdefault(r['item_id'],{})
                assert r['mode'] not in slot; slot[r['mode']] = r
        assert set(by_id) == {r['item_id'] for r in labels}
        assert all(set(v) == {'greedy'}|{f'sample{i}' for i in range(8)} for v in by_id.values())
    from train.document_qa_span_scores import score_spans
    from train.real_cpt_readout import corrupt_work
    from train.real_cpt_scaffold import verified_work
    inventory = []
    for index,r in enumerate(labels):
        if index % 4 != a.shard: continue
        q = question_projection(r)
        notes = {'B0_raw':None} if a.model == 'B0' else dict(raw=None,empty=[])
        if a.model == 'T':
            notes.update({mode:value['content_ids'] for mode,value in by_id[r['item_id']].items()})
            notes['oracle'] = verified_work(q['prefix_ids'],tok)[1]
            notes['corrupt'] = corrupt_work(q,tok)[1]
            donor = labels[(index+1)%len(labels)]['item_id']
            notes['other'] = by_id[donor]['greedy']['content_ids']
        for mode,content in notes.items():
            context = reader_context(q,tok,content)
            assert len(context)+16 <= 672
            inventory.append(dict(item_id=r['item_id'],document_id=r['document_id'],mode=mode,
                context_ids=context,target_ids=r['continuation16_ids']))
    started=time.monotonic()
    with (a.output/'scores.jsonl').open('x') as stream:
        for start in range(0,len(inventory),4):
            check_deadline(a.deadline_utc); block=inventory[start:start+4]
            #14*64 and64 are divisible by4; no fake padded documents.
            assert len(block) == 4
            scored = score_spans(backend,[r['context_ids'] for r in block],
                [r['target_ids'] for r in block],width=672,target_slots=16)
            for r,value in zip(block,scored):
                write_line(stream,dict(**r,**value,status='ok'))
            if (start+4)%128 == 0: print('SCORED',start+4,'SECONDS',time.monotonic()-started,flush=True)
    assert len(inventory) == (64 if a.model == 'B0' else 64*14)
    atomic_json(a.output/'complete.json',dict(status='COMPLETE_V30_FIXED_READER_SCORE_SHARD',
        model=a.model,shard=a.shard,rows=len(inventory),sources=sources(),
        protocol_sha256=sha256_file(a.protocol/'protocol.json'),
        scores_sha256=sha256_file(a.output/'scores.jsonl'),seconds=time.monotonic()-started))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=('reference','qualify','verify','generate','score'))
    p.add_argument('--protocol',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--reference',type=Path)
    p.add_argument('--qualification',type=Path)
    p.add_argument('--release',type=Path)
    p.add_argument('--candidates',type=Path)
    p.add_argument('--shard',type=int,choices=range(4),default=0)
    p.add_argument('--model',choices=('T','B0'),default='T')
    p.add_argument('--deadline-utc',required=True)
    a = p.parse_args(); check_deadline(a.deadline_utc)
    {'reference':reference,'qualify':qualify,'verify':verify,'generate':generate,'score':score}[a.command](a)


if __name__ == '__main__':
    main()

"""Exploratory B0 question-only G4 self-notes, sealed before reader answers.

Sampling retains every action. Reader content is the declared decoded first-line
substring, re-encoded without special tokens; no answer or utility selects notes.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import time
from types import SimpleNamespace

import torch

from train.evaluate import GreedyDecoder, asset_provenance
from train.evaluate_local import setup_model
from train.evaluation_integrity import evaluation_dependencies
from train.neuron_backend import CausalCache
from weekend_thinking.artifacts import atomic_json, canonical_hash, require_utc_deadline, sha256_file, utc_now
from weekend_thinking.core import policy_log_probs, sample_actions
from weekend_thinking.pretrained import load_base, sampled_fingerprint

SCHEMA = 'gsm8k-self-notes-train32-g4-cap64-v1'
CAP, SAMPLES, SHARDS, EOS, TEMPERATURE = 64, 4, 4, 151643, .7
MODEL_ID, REVISION = 'Qwen/Qwen3-1.7B-Base', 'ea980cb0a6c2ae4b936e82123acc929f1cec04c1'
TRAIN_SHA = 'ea82612ea9582142387730c793eb67d3b12849002bc0b7fa6f8efafa7351419d'
INSTRUCTION = ('Write one short intermediate calculation that will help solve the question. '
               'Omit the final answer. Give only the calculation.')
DEMO_CALCULATIONS = (
    'Clips sold in May: 48 / 2 = 24.',
    'The pay per minute is 12 / 60 = 0.20 dollars.',
    'Betty has 100 / 2 = 50 dollars. Her grandparents give 2 * 15 = 30 dollars.')
FORMAT_TRANSFORM = 'Decode all non-EOS actions; exact substring before first U+000A; encode(add_special_tokens=False)'


def read(path):
    return json.loads(Path(path).read_text())


def sources():
    root = Path(__file__).resolve().parents[1]
    return {**evaluation_dependencies(), 'train/calculation_self_notes.py': sha256_file(__file__)}


def check_deadline(deadline):
    if datetime.now(timezone.utc) >= require_utc_deadline(deadline):
        raise RuntimeError('Self-note deadline reached; no retry')


def question_rows(path):
    values = [json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()]
    fields = {'item_id', 'document_id', 'task', 'split', 'question'}
    if (len(values) != 32 or any(set(r) != fields or r['task'] != 'gsm8k' or r['split'] != 'train'
            or not isinstance(r['question'], str) or not r['question'].strip()
            or not isinstance(r['item_id'], str) or not r['item_id'].startswith('gsm8k:train:')
            or r['document_id'] != r['item_id'] for r in values)
            or len({r['item_id'] for r in values}) != 32):
        raise ValueError('Require32 unique canonical TRAIN question-only records; no extra fields')
    if {r['item_id'] for r in values} & {f'gsm8k:train:{i:05d}' for i in range(3)}:
        raise ValueError('Current questions and three fixed demonstrations must be disjoint')
    return values


def canonical_demonstrations(source):
    """Only preparation opens the pinned TRAIN question column, never answers."""
    if sha256_file(source) != TRAIN_SHA:
        raise ValueError('Demonstrations require the pinned canonical TRAIN source')
    import pyarrow.parquet as pq
    questions = pq.read_table(source, columns=['question']).column('question')
    return [dict(item_id=f'gsm8k:train:{i:05d}', split='train', question=questions[i].as_py(),
                 calculation=calculation) for i, calculation in enumerate(DEMO_CALCULATIONS)]


def writer_prompt(question, demonstrations):
    """This API accepts one question and fixed demo calculations, no current target."""
    return '\n\n'.join([INSTRUCTION, *[f"Question: {d['question']}\nCalculation: {d['calculation']}"
        for d in demonstrations], f'Question: {question}\nCalculation:'])


def sample_seed(seed, question_id, sample_index):
    if type(sample_index) is not int or not 0 <= sample_index < SAMPLES:
        raise ValueError('Sample index must be0..3')
    return int(canonical_hash(dict(schema=SCHEMA, seed=seed, question_id=question_id,
                                   sample_index=sample_index))[:15], 16)


def base_assets(model_dir):
    assets = asset_provenance(Path(model_dir))
    if assets.get('model_id') != MODEL_ID or assets.get('revision') != REVISION:
        raise ValueError('Require the actual released B0 snapshot')
    return assets


def tokenizer_for(p):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(p['model_dir'], local_files_only=True, trust_remote_code=False)
    defined = set(tok.get_vocab().values())
    if (tok.eos_token_id != EOS or canonical_hash(tok.get_vocab()) != p['tokenizer_vocab_sha256']
            or p['banned_ids'] != sorted(set(range(p['vocab_size'])) - defined)):
        raise ValueError('Prepared B0 tokenizer/support changed')
    return tok


def prepare(output, questions, train_source, model_dir, deadline, seed=20261018):
    from transformers import AutoTokenizer
    check_deadline(deadline)
    values = question_rows(questions); demonstrations = canonical_demonstrations(train_source)
    assets = base_assets(model_dir)
    tok = AutoTokenizer.from_pretrained(model_dir, local_files_only=True, trust_remote_code=False)
    vocab_size = read(Path(model_dir)/'config.json')['vocab_size']; defined = set(tok.get_vocab().values())
    if tok.eos_token_id != EOS or not defined or min(defined) < 0 or max(defined) >= vocab_size:
        raise ValueError('Released EOS/defined support differs')
    prompts = []
    for row in values:
        rendered = writer_prompt(row['question'], demonstrations)
        prompts.append(dict(item_id=row['item_id'], question_id=row['item_id'], document_id=row['document_id'],
            question_sha256=canonical_hash(row['question']), rendered=rendered,
            prompt_ids=tok.encode(rendered, add_special_tokens=False)))
    if any(not x['prompt_ids'] for x in prompts):
        raise ValueError('Empty writer prompt')
    seeds = [sample_seed(seed, r['item_id'], n) for r in values for n in range(SAMPLES)]
    if len(set(seeds)) != 128:
        raise ValueError('Self-note seeds collide')
    inputs = dict(rows=values, demonstrations=demonstrations, prompts=prompts,
        question_only=True, current_gold_teacher_or_manual_notes_exposed=False,
        demo_source=dict(source_sha256=TRAIN_SHA, canonical_columns_read=['question'], positions=[0,1,2]))
    check_deadline(deadline); output.mkdir(parents=True, exist_ok=False)
    atomic_json(output/'inputs.json', inputs)
    p = dict(schema=SCHEMA, experiment_id=output.name, created_utc=utc_now(), source_files=sources(),
        questions_file=str(Path(questions).resolve()), questions_sha256=sha256_file(questions),
        inputs_sha256=sha256_file(output/'inputs.json'), item_ids=[r['item_id'] for r in values],
        item_ids_sha256=canonical_hash([r['item_id'] for r in values]),
        demonstrations_sha256=canonical_hash(demonstrations), demo_source_sha256=TRAIN_SHA,
        model_dir=str(Path(model_dir).resolve()), assets=assets, tokenizer_vocab_sha256=canonical_hash(tok.get_vocab()),
        vocab_size=vocab_size, banned_ids=sorted(set(range(vocab_size))-defined), compute='fp32',
        items=32, samples_per_question=SAMPLES, expected_notes=128, shards=SHARDS,
        items_per_shard=8, notes_per_shard=32, cap=CAP, temperature=TEMPERATURE, eos_id=EOS,
        seed=seed, seed_rule='SHA256(schema,seed,question_id,sample_index), first15 hex digits',
        prefill_width=max(len(x['prompt_ids']) for x in prompts),
        prompt_lengths=[len(x['prompt_ids']) for x in prompts],
        sampling='four independently seeded sequential B1 Gumbel-max paths/question; no top-k/top-p',
        stopping='First decoded newline, EOS151643, or64 actions; no resampling or utility filtering',
        format_transform=FORMAT_TRANSFORM, forward_atol=.002, replay_atol=.001,
        cpu_reference='first4 global prompt next-logp vectors, native unpadded HF/B0/T.7/defined support',
        replay='All actual actions of sample0 for the first question of each shard, including terminating action',
        limitations=['Reuses the previous32 TRAIN questions after their pilot outcomes: exploratory, not blinded.',
            'Three fixed question-only manual demonstration calculations are external prompt assistance.',
            'No current answer/gold, teacher, verified current note, utility selection, or training dependency.',
            'Decoded first-line re-encoding may differ from sampled token boundaries; both originals are retained.',
            'Keep empty lines, caps, final-answer violations, errors and all four samples; no best-of4.'])
    atomic_json(output/'protocol.json', p)
    return p


def read_protocol(output):
    p = read(output/'protocol.json')
    if (p['schema'] != SCHEMA or p['source_files'] != sources() or p['compute'] != 'fp32'
            or (p['items'],p['samples_per_question'],p['expected_notes'],p['shards'],p['cap'],p['temperature'])
                != (32,SAMPLES,128,SHARDS,CAP,TEMPERATURE)
            or p['eos_id'] != EOS or p['format_transform'] != FORMAT_TRANSFORM
            or p['assets'].get('model_id') != MODEL_ID or p['assets'].get('revision') != REVISION
            or (p['forward_atol'],p['replay_atol']) != (.002,.001)):
        raise ValueError('Self-note source/policy/format changed')
    for file, h in [(output/'inputs.json',p['inputs_sha256']), (p['questions_file'],p['questions_sha256']),
                   (Path(p['model_dir'])/'ASSET_MANIFEST.json',p['assets']['manifest_sha256'])]:
        if sha256_file(file) != h:
            raise ValueError(f'Changed self-note input: {file}')
    inputs = read(output/'inputs.json'); values = question_rows(p['questions_file']); demos=inputs['demonstrations']
    if (inputs['rows'] != values or p['item_ids'] != [r['item_id'] for r in values]
            or p['item_ids_sha256'] != canonical_hash(p['item_ids'])
            or canonical_hash(demos) != p['demonstrations_sha256'] or len(demos) != 3
            or [d['item_id'] for d in demos] != [f'gsm8k:train:{i:05d}' for i in range(3)]
            or [d['calculation'] for d in demos] != list(DEMO_CALCULATIONS)
            or inputs['demo_source'] != dict(source_sha256=TRAIN_SHA, canonical_columns_read=['question'], positions=[0,1,2])
            or p['demo_source_sha256'] != TRAIN_SHA or inputs['question_only'] is not True
            or inputs['current_gold_teacher_or_manual_notes_exposed'] is not False):
        raise ValueError('Self-note question/demonstration boundary differs')
    if len(inputs['prompts']) != 32:
        raise ValueError('Require32 writer prompts')
    for row, item in zip(values, inputs['prompts']):
        if (item['item_id'] != row['item_id'] or item['question_id'] != row['item_id']
                or item['document_id'] != row['document_id'] or item['question_sha256'] != canonical_hash(row['question'])
                or item['rendered'] != writer_prompt(row['question'], demos) or not item['prompt_ids']
                or any(type(t) is not int or not 0 <= t < p['vocab_size'] or t in p['banned_ids'] for t in item['prompt_ids'])):
            raise ValueError('Question-only prompt differs')
    if ([len(x['prompt_ids']) for x in inputs['prompts']] != p['prompt_lengths']
            or max(p['prompt_lengths']) != p['prefill_width']):
        raise ValueError('Prompt width differs')
    return p, inputs


def formatted_actions(tok, actions, *, eos_id=EOS, cap=CAP):
    """Retain terminating bytes; only the reader substring is re-encoded."""
    if not actions or len(actions)>cap or eos_id in actions[:-1]:
        raise ValueError('Invalid sampled action/stop sequence')
    decode = lambda ids:tok.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    original_ids = actions[:-1] if actions[-1] == eos_id else list(actions)
    original = decode(original_ids); raw = decode(actions)
    for end in range(1,len(actions)):
        if '\n' in decode(actions[:end]):
            raise ValueError('Actions continued after the first decoded newline')
    stop = 'newline' if '\n' in original else f'token:{eos_id}' if actions[-1]==eos_id else 'length_cap'
    if stop == 'length_cap' and len(actions) != cap:
        raise ValueError('Short path without EOS/newline')
    content = original.split('\n',1)[0]
    ids = tok.encode(content, add_special_tokens=False)
    return dict(content_ids=ids, content_text=content, text=content,
        original_content_ids=original_ids, original_content_text=original, raw_text=raw,
        stop_type=stop, n_actions=len(actions), n_content_tokens=len(ids),
        n_original_content_tokens=len(original_ids), empty_first_line=content=='',
        final_answer_marker_in_raw='####' in raw, final_answer_marker_in_content='####' in content)


@torch.no_grad()
def padded_next(decoder, prompt, width, pad_id=EOS, cap=CAP):
    x=torch.full((1,width),pad_id,dtype=torch.long);x[0,:len(prompt)]=torch.tensor(prompt)
    cache=CausalCache(decoder.model.config.num_hidden_layers,width+cap)
    return decoder._next(x.to(decoder.device),torch.arange(width,device=decoder.device),cache,
                         torch.tensor(len(prompt)-1,device=decoder.device))


@torch.no_grad()
def sample_note(decoder, prompt, *, width, seed, set_seed, cap=CAP, eos_id=EOS):
    """Prefix-only stochastic path. Never accepts a row, answer, reward or future."""
    if not prompt or len(prompt)>width or cap<1:
        raise ValueError('Invalid prompt/decode bounds')
    x=torch.full((1,width),eos_id,dtype=torch.long);x[0,:len(prompt)]=torch.tensor(prompt)
    x=x.to(decoder.device);positions=torch.arange(width,device=decoder.device)
    position=torch.tensor([len(prompt)],device=decoder.device);cache=CausalCache(decoder.model.config.num_hidden_layers,width+cap)
    actions=[];cached=[];failure=None;started=time.monotonic()
    try:
        decoder.sync();set_seed(seed)  # Settle before reset; no extra sync before the first draw.
        logits=decoder._next(x,positions,cache,position-1)
        for step in range(cap):
            token,lp=sample_actions(logits,TEMPERATURE,decoder.banned_ids)
            decoder.sync();value=int(token.cpu()[0]);actions.append(value);cached.append(float(lp.cpu()[0]))
            if not math.isfinite(cached[-1]):
                raise RuntimeError('Nonfinite sampled log probability')
            raw=decoder.tokenizer.decode(actions,skip_special_tokens=False,clean_up_tokenization_spaces=False)
            if value==eos_id or '\n' in raw:
                break
            if step+1<cap:
                logits=decoder._next(token[:,None],position,cache);position=position+1
        decoder.sync()
        formatted=formatted_actions(decoder.tokenizer,actions,eos_id=eos_id,cap=cap)
    except Exception as error:
        failure=f'{type(error).__name__}: {error}'
        try:formatted=formatted_actions(decoder.tokenizer,actions,eos_id=eos_id,cap=cap)
        except Exception:
            try: raw=decoder.tokenizer.decode(actions,skip_special_tokens=False,clean_up_tokenization_spaces=False)
            except Exception: raw=''
            formatted=dict(raw_text=raw,stop_type='error')
    return dict(status='error' if failure else 'ok',actions=actions,
        cached_logps=[v if math.isfinite(v) else repr(v) for v in cached],
        action_mask=[True]*len(actions)+[False]*(cap-len(actions)),seed=seed,
        generation_wall_seconds=time.monotonic()-started,**formatted,**({'error':failure} if failure else {}))


@torch.no_grad()
def replay_logps(backend, prompt, actions, width, *, cap=CAP, pad_id=EOS):
    x=torch.full((1,width+cap),pad_id,dtype=torch.long);real=prompt+actions[:-1];x[0,:len(real)]=torch.tensor(real)
    hidden=backend.hidden(x.to(backend.device));positions=torch.arange(cap,device=backend.device)+len(prompt)-1
    selected=hidden[:,positions];labels=torch.full((cap,),pad_id,dtype=torch.long);labels[:len(actions)]=torch.tensor(actions)
    labels=labels.to(backend.device);out=[]
    for start in range(0,cap,32):
        lp=policy_log_probs(backend.model.lm_head(selected[:,start:start+32]).float(),TEMPERATURE,backend.banned_ids)
        out.append(lp.gather(-1,labels[start:start+32][None,:,None])[:,:,0])
    value=torch.cat(out,-1);backend.sync()
    return value.cpu()[0,:len(actions)]


@torch.no_grad()
def cpu_reference(output, threads, deadline):
    if type(threads) is not int or not 1<=threads<=16:
        raise ValueError('CPU reference requires1..16 threads')
    check_deadline(deadline);p,inputs=read_protocol(output);torch.set_num_threads(threads)
    assets=base_assets(p['model_dir'])
    if assets!=p['assets']:raise ValueError('Actual released B0 bytes changed')
    model,tok=load_base(p['model_dir']);model.eval()
    if (not model.config.tie_word_embeddings or model.get_input_embeddings().weight is not model.get_output_embeddings().weight
            or any(t.dtype!=torch.float32 for t in model.parameters())):
        raise ValueError('Require tied FP32 released B0')
    if canonical_hash(tok.get_vocab())!=p['tokenizer_vocab_sha256']:raise ValueError('Tokenizer changed')
    if any(tok.encode(x['rendered'],add_special_tokens=False)!=x['prompt_ids'] for x in inputs['prompts']):
        raise ValueError('Prepared question-only tokenization differs')
    d=output/'cpu-reference';d.mkdir(exist_ok=False);values=[];timings=[];started=time.monotonic()
    for item in inputs['prompts'][:4]:
        check_deadline(deadline);before=time.monotonic();x=torch.tensor([item['prompt_ids']])
        hidden=model.model(input_ids=x,use_cache=False).last_hidden_state
        lp=policy_log_probs(model.lm_head(hidden[:,-1]),TEMPERATURE,p['banned_ids']).cpu()
        if not torch.isfinite(lp).all():raise ValueError('Nonfinite native CPU logps')
        values.append(lp);timings.append(time.monotonic()-before)
    value=torch.cat(values);torch.save(value,d/'next-logps.pt')
    import transformers
    atomic_json(d/'reference.json',dict(status='complete',protocol_sha256=sha256_file(output/'protocol.json'),
        signature=sampled_fingerprint(model),assets_sha256=canonical_hash(assets),tied=True,dtype='float32',
        shape=list(value.shape),item_ids=p['item_ids'][:4],prompt_sha256=[canonical_hash(x['prompt_ids']) for x in inputs['prompts'][:4]],
        logps_sha256=sha256_file(d/'next-logps.pt'),forward_wall_seconds=timings,total_wall_seconds=time.monotonic()-started,
        torch_version=torch.__version__,transformers_version=transformers.__version__,finished_utc=utc_now(),
        method='Actual released B0/native HF eager/unpadded causal trunk, real-last-position head, T.7/defined support'))


def reference_receipt(output,p):
    r=read(output/'cpu-reference/reference.json')
    if (r['status']!='complete' or r['protocol_sha256']!=sha256_file(output/'protocol.json')
            or r['signature']['parameters']!=1720574976 or r['dtype']!='float32' or not r['tied']
            or r['assets_sha256']!=canonical_hash(p['assets']) or r['item_ids']!=p['item_ids'][:4]
            or r['shape']!=[4,p['vocab_size']] or r['logps_sha256']!=sha256_file(output/'cpu-reference/next-logps.pt')):
        raise ValueError('Complete actual-B0 native CPU reference required')
    return r


@torch.no_grad()
def generate(output, shard, deadline):
    if type(shard) is not int or shard not in range(SHARDS):raise ValueError('Shard must be0..3')
    check_deadline(deadline);p,inputs=read_protocol(output);reference=reference_receipt(output,p)
    if base_assets(p['model_dir'])!=p['assets']:raise ValueError('Actual B0 assets changed')
    selected=inputs['prompts'][shard::SHARDS];d=output/'generation'/f'shard-{shard}';d.mkdir(parents=True,exist_ok=False)
    r=dict(status='running',shard=shard,expected_notes=32,notes=0,item_ids=[x['item_id'] for x in selected],
        protocol_sha256=sha256_file(output/'protocol.json'),started_utc=utc_now(),deadline_utc=deadline)
    atomic_json(d/'run.json',r)
    try:
        backend,tok,loaded,signature,set_seed=setup_model(SimpleNamespace(model_dir=Path(p['model_dir']),checkpoint=None,device='xla'))
        if loaded is not None or signature!=reference['signature'] or list(backend.banned_ids)!=p['banned_ids']:
            raise ValueError('Actual B0/support differs from native reference')
        def sync():backend.sync();check_deadline(deadline)
        decoder=GreedyDecoder(backend.model,tok,sync=sync,banned_ids=backend.banned_ids);r['signature']=signature
        before=time.monotonic();logits=padded_next(decoder,selected[0]['prompt_ids'],p['prefill_width'])
        actual=policy_log_probs(logits,TEMPERATURE,backend.banned_ids);backend.sync();actual=actual.cpu()
        native=torch.load(output/'cpu-reference/next-logps.pt',weights_only=True)[shard:shard+1]
        defined=sorted(tok.get_vocab().values());error=(actual[:,defined]-native[:,defined]).abs().max().item()
        torch.save(actual,d/'forward-logps.pt')
        r['forward_check']=dict(max_error=error if math.isfinite(error) else repr(error),atol=.002,
            prompt_sha256=canonical_hash(selected[0]['prompt_ids']),logps_sha256=sha256_file(d/'forward-logps.pt'),
            wall_seconds=time.monotonic()-before,native_reference_sha256=sha256_file(output/'cpu-reference/reference.json'))
        if not math.isfinite(error) or error>.002:raise RuntimeError('Native CPU/device next-logp parity failed')
        with (d/'notes.jsonl').open('x') as stream:
            for index,item in enumerate(selected):
                for sample in range(SAMPLES):
                    row=sample_note(decoder,item['prompt_ids'],width=p['prefill_width'],
                        seed=sample_seed(p['seed'],item['item_id'],sample),set_seed=set_seed)
                    row.update(item_id=f"{item['item_id']}:sample:{sample}",question_id=item['item_id'],
                        document_id=item['document_id'],sample_index=sample,split='train',task='gsm8k',shard=shard,
                        temperature=TEMPERATURE,prompt_sha256=canonical_hash(item['prompt_ids']),
                        question_sha256=item['question_sha256'],current_gold_teacher_or_manual_notes_exposed=False)
                    if index==0 and sample==0 and row['status']=='ok':
                        before=time.monotonic()
                        audit=dict(atol=.001,question_id=item['item_id'],sample_index=0,actions=len(row['actions']))
                        try:
                            replay=replay_logps(backend,item['prompt_ids'],row['actions'],p['prefill_width']).tolist()
                            # Preserve even a malformed audit without emitting nonfinite JSON.
                            row['replayed_logps']=[v if math.isfinite(v) else repr(v) for v in replay]
                            if len(replay)!=len(row['actions']) or not all(math.isfinite(v) for v in replay):
                                raise ValueError('Replay must cover every sampled action with finite logps')
                            error=max(abs(a-b) for a,b in zip(row['cached_logps'],replay))
                            audit['max_error']=error
                            if error>.001:raise ValueError('Actual sampled cache/replay mismatch')
                        except Exception as error:
                            audit['error']=f'{type(error).__name__}: {error}'
                            row.update(status='error',error=audit['error'])
                        audit['wall_seconds']=time.monotonic()-before;r['replay_check']=audit
                    stream.write(json.dumps(row,allow_nan=False)+'\n');stream.flush();r['notes']+=1
                    atomic_json(d/'run.json',r,overwrite=True)
                    if row['status']!='ok':raise RuntimeError('Failed note retained; no resampling')
                print(json.dumps(dict(shard=shard,notes=r['notes'],utc=utc_now())),flush=True)
        r.update(status='complete',notes_sha256=sha256_file(d/'notes.jsonl'))
    except BaseException as error:r.update(status='failed',error=f'{type(error).__name__}: {error}');raise
    finally:r['finished_utc']=utc_now();atomic_json(d/'run.json',r,overwrite=True)


def load_complete(output):
    """No model, canonical TRAIN source, gold, teacher or manual-current-note read."""
    p,inputs=read_protocol(output);ref=reference_receipt(output,p);tok=tokenizer_for(p);notes={}
    receipts=[output/'protocol.json',output/'inputs.json',output/'cpu-reference/reference.json',output/'cpu-reference/next-logps.pt']
    native=torch.load(output/'cpu-reference/next-logps.pt',map_location='cpu',weights_only=True)
    if list(native.shape)!=ref['shape'] or not torch.isfinite(native).all():raise ValueError('Invalid native reference tensor')
    defined=set(tok.get_vocab().values())
    for shard in range(SHARDS):
        d=output/'generation'/f'shard-{shard}';r=read(d/'run.json');values=[json.loads(s) for s in (d/'notes.jsonl').read_text().splitlines()]
        ids=p['item_ids'][shard::SHARDS];expected=[(k,n) for k in ids for n in range(SAMPLES)]
        if (r['status']!='complete' or r['shard']!=shard or r['notes']!=32 or r['expected_notes']!=32 or r['item_ids']!=ids
                or r['protocol_sha256']!=sha256_file(output/'protocol.json') or r['signature']!=ref['signature']
                or r['notes_sha256']!=sha256_file(d/'notes.jsonl') or len(values)!=32
                or [(v['question_id'],v['sample_index']) for v in values]!=expected):
            raise ValueError('Require all128 complete self-note paths and exact shard order')
        f=r['forward_check'];actual=torch.load(d/'forward-logps.pt',map_location='cpu',weights_only=True)
        if list(actual.shape)!=[1,p['vocab_size']] or not torch.isfinite(actual).all():
            raise ValueError('Invalid padded first-prefix reference tensor')
        error=(actual[:,sorted(defined)]-native[shard:shard+1,sorted(defined)]).abs().max().item()
        if (f['logps_sha256']!=sha256_file(d/'forward-logps.pt') or f['atol']!=.002
                or f['prompt_sha256']!=canonical_hash(inputs['prompts'][shard]['prompt_ids'])
                or not math.isfinite(error) or error!=f['max_error'] or not 0<=error<=.002
                or f['native_reference_sha256']!=sha256_file(output/'cpu-reference/reference.json')):
            raise ValueError('Native CPU/device first-prefix logp evidence differs')
        for row in values:
            key=(row['question_id'],row['sample_index']);item=inputs['prompts'][p['item_ids'].index(key[0])];a=row['actions'];lp=row['cached_logps']
            if (row['status']!='ok' or not 1<=len(a)<=CAP or any(type(t) is not int or t not in defined for t in a)
                    or len(lp)!=len(a) or any(type(v) not in (float,int) or not math.isfinite(v) for v in lp)
                    or row['item_id']!=f'{key[0]}:sample:{key[1]}' or row['document_id']!=item['document_id']
                    or row['shard']!=shard or row['split']!='train' or row['task']!='gsm8k'
                    or row['seed']!=sample_seed(p['seed'],*key) or row['temperature']!=TEMPERATURE
                    or row['action_mask']!=[True]*len(a)+[False]*(CAP-len(a))
                    or row['current_gold_teacher_or_manual_notes_exposed'] is not False
                    or row['question_sha256']!=item['question_sha256'] or row['prompt_sha256']!=canonical_hash(item['prompt_ids'])
                    or not math.isfinite(row['generation_wall_seconds']) or row['generation_wall_seconds']<0):
                raise ValueError('Self-note token/logp/seed/question provenance differs')
            formatted=formatted_actions(tok,a)
            if any(row.get(k)!=v for k,v in formatted.items()):raise ValueError('Decoded first-line format transform differs')
            notes[key]=row
        first=values[0];replayed=first.get('replayed_logps',[]);cached=first['cached_logps'];audit=r['replay_check']
        if (len(replayed)!=len(cached) or any(not math.isfinite(v) for v in replayed)
                or audit['question_id']!=ids[0] or audit['sample_index']!=0 or audit['actions']!=len(cached) or audit['atol']!=.001
                or max(abs(a-b) for a,b in zip(cached,replayed))!=audit['max_error']
                or not 0<=audit['max_error']<=.001):
            raise ValueError('All-action sampled cache/replay evidence differs')
        receipts.extend((d/'run.json',d/'notes.jsonl',d/'forward-logps.pt'))
    if set(notes)!={(k,n) for k in p['item_ids'] for n in range(SAMPLES)}:raise ValueError('Require all128 notes')
    return p,inputs,notes,receipts


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('prepare','cpu-reference','generate'))
    parser.add_argument('--output',required=True,type=Path);parser.add_argument('--questions',type=Path)
    parser.add_argument('--train-source',type=Path);parser.add_argument('--model-dir',type=Path,default=Path('local/assets/qwen3-1.7b-base'))
    parser.add_argument('--shard-index',type=int,choices=range(4));parser.add_argument('--cpu-threads',type=int,default=16)
    parser.add_argument('--seed',type=int,default=20261018);parser.add_argument('--deadline',required=True)
    args=parser.parse_args()
    if args.command=='prepare':
        if args.questions is None or args.train_source is None:parser.error('prepare requires --questions and --train-source')
        prepare(args.output,args.questions,args.train_source,args.model_dir,args.deadline,args.seed)
    elif args.command=='cpu-reference':cpu_reference(args.output,args.cpu_threads,args.deadline)
    elif args.shard_index is None:parser.error('generate requires --shard-index0..3')
    else:generate(args.output,args.shard_index,args.deadline)


if __name__=='__main__':main()

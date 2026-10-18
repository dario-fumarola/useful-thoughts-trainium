"""Question-only greedy external-teacher TRAIN32 pilot; no answers or training."""
from __future__ import annotations

import argparse
from collections.abc import Mapping
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import time
from types import SimpleNamespace

import torch

from train.evaluate import GenerationFailure, GreedyDecoder, shard_questions
from train.evaluate_local import setup_model
from train.evaluation_integrity import evaluation_dependencies
from train.hindsight_teacher import native_vocab, verify_teacher_assets
from train.neuron_backend import CausalCache
from weekend_thinking.artifacts import atomic_json, canonical_hash, require_utc_deadline, sha256_file, utc_now
from weekend_thinking.core import content_from_actions, policy_log_probs
from weekend_thinking.pretrained import sampled_fingerprint

SCHEMA = 'calculation-teacher-train32-greedy-v1'
REVISION = '1cfa9a7208912126459214e8b04321603b3df60c'
CAP, STOPS = 256, (151645, 151643)
SYSTEM = ('Write compact, useful intermediate calculations for the complete math question below. '
          'Give only the calculation note. Omit the final answer, question restatement, '
          'greetings, boilerplate, and discussion of these instructions.')


def read(path):
    return json.loads(Path(path).read_text())


def sources():
    root = Path(__file__).resolve().parents[1]
    return {**evaluation_dependencies(), **{name: sha256_file(root/name) for name in
        ('train/calculation_teacher.py', 'train/hindsight_teacher.py')}}


def check_deadline(deadline):
    if datetime.now(timezone.utc) >= require_utc_deadline(deadline):
        raise RuntimeError('Teacher deadline reached; no retry')


def question_rows(path):
    values = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    fields = {'item_id', 'document_id', 'task', 'split', 'question'}
    if (len(values) != 32 or any(set(row) != fields or row['task'] != 'gsm8k'
            or row['split'] != 'train' or any(not isinstance(row[key], str) or not row[key].strip()
                for key in ('item_id', 'document_id', 'question')) for row in values)
            or len({r['item_id'] for r in values}) != 32
            or len({r['document_id'] for r in values}) != 32):
        raise ValueError('Require exactly32 unique TRAIN question-only records; no additional fields')
    return values


def native_prompt(tokenizer, question):
    messages = [dict(role='system', content=SYSTEM), dict(role='user', content=question)]
    rendered = tokenizer.apply_chat_template(messages, tokenize=False,
        add_generation_prompt=True, enable_thinking=False)
    ids = tokenizer.encode(rendered, add_special_tokens=False)
    native = tokenizer.apply_chat_template(messages, tokenize=True,
        add_generation_prompt=True, enable_thinking=False)
    if isinstance(native, Mapping):
        native = native['input_ids']
    if not ids or ids != native:
        raise ValueError('Rendered/native chat tokenization differs')
    return dict(messages=messages, rendered=rendered, prompt_ids=ids,
                question_sha256=canonical_hash(question))


def prepare(output, questions, teacher_dir, base_dir, deadline):
    from transformers import AutoTokenizer
    check_deadline(deadline)
    values = question_rows(questions)
    verify_teacher_assets(teacher_dir)
    tok = AutoTokenizer.from_pretrained(teacher_dir, local_files_only=True, trust_remote_code=False)
    base = AutoTokenizer.from_pretrained(base_dir, local_files_only=True, trust_remote_code=False)
    vocab = native_vocab(base, tok)
    prompts = [dict(item_id=row['item_id'], document_id=row['document_id'],
                    **native_prompt(tok, row['question'])) for row in values]
    width = max(len(p['prompt_ids']) for p in prompts)
    inputs = dict(rows=values, prompts=prompts, chat_template=tok.chat_template,
        template_arguments=dict(enable_thinking=False, add_generation_prompt=True),
        question_only=True, gold_or_verified_notes_read=False)
    check_deadline(deadline)
    output.mkdir(parents=True, exist_ok=False)
    atomic_json(output/'inputs.json', inputs)
    protocol = dict(schema=SCHEMA, experiment_id=output.name, created_utc=utc_now(),
        questions_file=str(questions.resolve()), questions_sha256=sha256_file(questions),
        inputs_sha256=sha256_file(output/'inputs.json'), item_ids=[r['item_id'] for r in values],
        item_ids_sha256=canonical_hash([r['item_id'] for r in values]),
        teacher_dir=str(teacher_dir.resolve()), base_dir=str(base_dir.resolve()),
        teacher_model_id='Qwen/Qwen3-4B', teacher_revision=REVISION,
        teacher_manifest_sha256=sha256_file(teacher_dir/'ASSET_MANIFEST.json'),
        base_manifest_sha256=sha256_file(base_dir/'ASSET_MANIFEST.json'), native_vocab=vocab,
        source_files=sources(), compute='fp32', items=32, shards=4, items_per_shard=8,
        sharding='selection_order[shard_index::4]', cap=CAP, stop_ids=list(STOPS), pad_id=151643,
        prefill_width=width, cache_width=width+CAP, prompt_lengths=[len(p['prompt_ids']) for p in prompts],
        generation='one greedy argmax path per item; no selection or retry', temperature=1.,
        top_k=None, top_p=None, seed=20261018, forward_atol=.002, replay_atol=.001,
        cpu_reference='first4 global prompts, individually unpadded; each is the first prompt of one shard',
        limitations=['TRAIN question-only external4B assistance, not student self-generated usefulness.',
            'Final-answer omission is an instruction, not a content filter; retain all capped/empty/error outputs.',
            'Greedy convention differs from the previous temperature0.7 hindsight teacher.',
            'No gold, source answer, verified note, utility scoring, or training dependency.'])
    atomic_json(output/'protocol.json', protocol)
    return protocol


def read_protocol(output):
    p = read(output/'protocol.json')
    if (p['schema'] != SCHEMA or p['source_files'] != sources() or p['compute'] != 'fp32'
            or (p['items'],p['shards'],p['cap'],p['temperature']) != (32,4,CAP,1.)
            or p['stop_ids'] != list(STOPS) or p['teacher_revision'] != REVISION):
        raise ValueError('Changed teacher source/policy/identity')
    for file, expected in ((output/'inputs.json',p['inputs_sha256']),
        (Path(p['questions_file']),p['questions_sha256']),
        (Path(p['teacher_dir'])/'ASSET_MANIFEST.json',p['teacher_manifest_sha256']),
        (Path(p['base_dir'])/'ASSET_MANIFEST.json',p['base_manifest_sha256'])):
        if sha256_file(file) != expected:
            raise ValueError(f'Changed preparation input: {file}')
    inputs = read(output/'inputs.json')
    values = question_rows(p['questions_file'])
    if (inputs['rows'] != values or p['item_ids'] != [r['item_id'] for r in values]
            or p['item_ids_sha256'] != canonical_hash(p['item_ids'])
            or [x['item_id'] for x in inputs['prompts']] != p['item_ids']
            or [len(x['prompt_ids']) for x in inputs['prompts']] != p['prompt_lengths']
            or max(p['prompt_lengths']) != p['prefill_width']):
        raise ValueError('Question/prompt selection order differs')
    return p, inputs


@torch.no_grad()
def cpu_reference(output, threads, deadline):
    from transformers import AutoModelForCausalLM, AutoTokenizer, __version__
    if type(threads) is not int or not 1 <= threads <= 16:
        raise ValueError('CPU reference requires1..16 threads')
    check_deadline(deadline); torch.set_num_threads(threads)
    p, inputs = read_protocol(output); verify_teacher_assets(Path(p['teacher_dir']))
    directory = output/'cpu-reference'; directory.mkdir(exist_ok=False); started = time.monotonic()
    model, info = AutoModelForCausalLM.from_pretrained(p['teacher_dir'], local_files_only=True,
        trust_remote_code=False, dtype=torch.float32, attn_implementation='eager', output_loading_info=True)
    if any(info.get(k) for k in ('missing_keys','unexpected_keys','mismatched_keys','error_msgs')):
        raise ValueError('Non-exact teacher load')
    model.eval(); model.tie_weights()
    if (model.get_input_embeddings().weight is not model.get_output_embeddings().weight
            or any(t.dtype != torch.float32 for t in model.parameters())):
        raise ValueError('Require tied FP32 teacher')
    tok = AutoTokenizer.from_pretrained(p['teacher_dir'],local_files_only=True,trust_remote_code=False)
    if canonical_hash(tok.get_vocab()) != p['native_vocab']['mapping_sha256']:
        raise ValueError('Native teacher tokenizer differs from prepared exact-vocab proof')
    banned = sorted(set(range(model.config.vocab_size))-set(tok.get_vocab().values()))
    values, timings = [], []
    for item in inputs['prompts'][:4]:
        check_deadline(deadline); before = time.monotonic()
        x = torch.tensor([item['prompt_ids']])
        hidden = model.model(input_ids=x,use_cache=False).last_hidden_state
        lp = policy_log_probs(model.lm_head(hidden[:, -1]),1.,banned).cpu()
        if not torch.isfinite(lp).all():
            raise ValueError('Nonfinite CPU teacher next logps')
        values.append(lp);timings.append(time.monotonic()-before)
        print(json.dumps(dict(phase='native_cpu_next_logps',item_id=item['item_id'],seconds=timings[-1],utc=utc_now())),flush=True)
    torch.save(torch.cat(values),directory/'next-logps.pt')
    torch.save(info,directory/'loading-info.pt')
    atomic_json(directory/'reference.json',dict(status='complete',finished_utc=utc_now(),
        protocol_sha256=sha256_file(output/'protocol.json'),signature=sampled_fingerprint(model),
        tied=True,parameter_tensors=len(list(model.parameters())),dtype='float32',
        item_ids=p['item_ids'][:4],prompt_sha256=[canonical_hash(x['prompt_ids']) for x in inputs['prompts'][:4]],
        logps_sha256=sha256_file(directory/'next-logps.pt'),shape=list(torch.cat(values).shape),
        loading_info_sha256=sha256_file(directory/'loading-info.pt'),
        loading_info={k:sorted(v) if isinstance(v,set) else v for k,v in info.items()},
        loading_info_types={k:type(v).__name__ for k,v in info.items()},
        banned_ids=banned,threads=threads,torch_version=torch.__version__,transformers_version=__version__,
        forward_wall_seconds=timings,total_wall_seconds=time.monotonic()-started,
        method='Native HF eager causal trunk on individual unpadded prompts; real-last-position head only; T1/defined support'))


class RecordingDecoder(GreedyDecoder):
    """Observe selected logps on one original greedy path; never resample."""
    record = False

    def _next(self, *args, **kwargs):
        logits = super()._next(*args, **kwargs)
        if self.record:
            self.cached_logps.append(policy_log_probs(logits,1.,self.banned_ids).max(-1).values[0])
        return logits


@torch.no_grad()
def padded_next(decoder, prompt, width, pad_id):
    x = torch.full((1,width),pad_id,dtype=torch.long)
    x[0,:len(prompt)] = torch.tensor(prompt)
    cache = CausalCache(decoder.model.config.num_hidden_layers,width+CAP)
    return decoder._next(x.to(decoder.device),torch.arange(width,device=decoder.device),cache,
                         torch.tensor(len(prompt)-1,device=decoder.device))


@torch.no_grad()
def replay_logps(backend, prompt, actions, prefill_width, pad_id):
    x = torch.full((1,prefill_width+CAP),pad_id,dtype=torch.long)
    real = prompt+actions[:-1]
    x[0,:len(real)] = torch.tensor(real)
    hidden = backend.hidden(x.to(backend.device))
    positions = torch.arange(CAP,device=backend.device)+len(prompt)-1
    selected = hidden[:,positions]
    labels = torch.full((CAP,),pad_id,dtype=torch.long)
    labels[:len(actions)] = torch.tensor(actions)
    labels = labels.to(backend.device); values=[]
    for first in range(0,CAP,32):
        lp = policy_log_probs(backend.model.lm_head(selected[:,first:first+32]).float(),1.,backend.banned_ids)
        values.append(lp.gather(-1,labels[first:first+32][None,:,None])[:,:,0])
    result = torch.cat(values,-1);backend.sync()
    return result.cpu()[0,:len(actions)]


def reference_receipt(output,p):
    reference = read(output/'cpu-reference/reference.json')
    if (reference['status'] != 'complete' or not reference['tied'] or reference['dtype'] != 'float32'
            or reference['protocol_sha256'] != sha256_file(output/'protocol.json')
            or reference['item_ids'] != p['item_ids'][:4]
            or reference['logps_sha256'] != sha256_file(output/'cpu-reference/next-logps.pt')
            or reference['shape'] != [4,151936]):
        raise ValueError('Require complete hash-bound CPU reference')
    return reference


@torch.no_grad()
def generate(output, shard, deadline):
    check_deadline(deadline);p,inputs=read_protocol(output);reference=reference_receipt(output,p)
    verify_teacher_assets(Path(p['teacher_dir']))
    directory=output/'generation'/f'shard-{shard}';directory.mkdir(parents=True,exist_ok=False)
    selected=shard_questions(inputs['prompts'],shard,4)
    report=dict(status='running',started_utc=utc_now(),shard=shard,notes=0,
        protocol_sha256=sha256_file(output/'protocol.json'),deadline_utc=deadline,
        item_ids=[x['item_id'] for x in selected],expected_notes=8)
    atomic_json(directory/'run.json',report)
    try:
        before=time.monotonic();backend,tok,loaded,signature,_=setup_model(SimpleNamespace(
            model_dir=Path(p['teacher_dir']),checkpoint=None,device='xla'))
        if loaded is not None or signature!=reference['signature'] or (
                backend.model.get_input_embeddings().weight is not backend.model.get_output_embeddings().weight):
            raise ValueError('Released tied teacher CPU/device identity differs')
        def sync():
            backend.sync();check_deadline(deadline)
        decoder=RecordingDecoder(backend.model,tok,sync=sync,banned_ids=backend.banned_ids)
        report.update(signature=signature,model_setup_seconds=time.monotonic()-before,
            teacher_model_id=p['teacher_model_id'],teacher_revision=p['teacher_revision'])
        check_deadline(deadline)
        before=time.monotonic()
        logits=padded_next(decoder,selected[0]['prompt_ids'],p['prefill_width'],p['pad_id'])
        actual=policy_log_probs(logits,1.,backend.banned_ids);backend.sync()
        native=torch.load(output/'cpu-reference/next-logps.pt',weights_only=True)[shard:shard+1]
        defined=sorted(tok.get_vocab().values())
        error=(actual.cpu()[:,defined]-native[:,defined]).abs().max().item()
        report['forward_check']=dict(max_error=error,atol=p['forward_atol'],wall_seconds=time.monotonic()-before)
        if not math.isfinite(error) or error>p['forward_atol']:
            raise RuntimeError('Native CPU/padded XLA next-logp parity failed before generation')
        with (directory/'notes.jsonl').open('x') as stream:
            for index,item in enumerate(selected):
                check_deadline(deadline);before=time.monotonic();decoder.record=index==0;decoder.cached_logps=[]
                failure=None
                try:
                    decoded=decoder.decode(item['prompt_ids'],max_new_tokens=CAP,prefill_width=p['prefill_width'],
                        pad_id=p['pad_id'],stop_ids=STOPS,answer_line_stop=False)
                except GenerationFailure as exc:
                    failure=f'{type(exc).__name__}: {exc}';decoded=exc.partial
                backend.sync();seconds=time.monotonic()-before
                row=dict(item_id=item['item_id'],document_id=item['document_id'],split='train',task='gsm8k',shard=shard,
                    question_sha256=item['question_sha256'],prompt_sha256=canonical_hash(item['prompt_ids']),
                    status='error' if failure else 'ok',actions=decoded.actions,content_ids=decoded.content_ids,
                    action_mask=[True]*len(decoded.actions)+[False]*(CAP-len(decoded.actions)),
                    text=decoded.text,raw_text=decoded.raw_text,stop_type=decoded.stop_type,
                    n_actions=len(decoded.actions),n_content_tokens=len(decoded.content_ids),
                    generation_wall_seconds=seconds,greedy=True,temperature=1.,gold_or_verified_notes_exposed=False)
                if failure:row['error']=failure
                if index==0 and failure is None:
                    decoder.record=False;cached=torch.stack(decoder.cached_logps).cpu()
                    before=time.monotonic();replayed=replay_logps(backend,item['prompt_ids'],decoded.actions,p['prefill_width'],p['pad_id'])
                    error=(cached-replayed).abs().max().item()
                    row['cached_logps']=cached.tolist();row['replayed_logps']=replayed.tolist()
                    report['replay_check']=dict(item_id=item['item_id'],actions=len(decoded.actions),
                        max_error=error,atol=p['replay_atol'],wall_seconds=time.monotonic()-before)
                    if not math.isfinite(error) or error>p['replay_atol']:row.update(status='error',error='Cached/teacher-forced replay failed')
                stream.write(json.dumps(row,allow_nan=False)+'\n');stream.flush();report['notes']+=1
                atomic_json(directory/'run.json',report,overwrite=True)
                if row['status']!='ok':raise RuntimeError('Failed note retained; no retry')
                print(json.dumps(dict(shard=shard,notes=report['notes'],utc=utc_now())),flush=True)
        report.update(status='complete',notes_sha256=sha256_file(directory/'notes.jsonl'))
    except BaseException as exc:
        report.update(status='failed',error=f'{type(exc).__name__}: {exc}');raise
    finally:
        report['finished_utc']=utc_now();atomic_json(directory/'run.json',report,overwrite=True)


def load_complete(output):
    """CPU-only reader boundary: no weights, answers, or verified notes opened."""
    p,inputs=read_protocol(output);reference=reference_receipt(output,p);notes={}
    receipts=[output/'protocol.json',output/'inputs.json',output/'cpu-reference/reference.json',
              output/'cpu-reference/next-logps.pt',output/'cpu-reference/loading-info.pt']
    for shard in range(4):
        d=output/'generation'/f'shard-{shard}';run=read(d/'run.json');path=d/'notes.jsonl'
        ids=p['item_ids'][shard::4]
        if (run['status']!='complete' or run['shard']!=shard or run['notes']!=8
                or run['expected_notes']!=8 or run['item_ids']!=ids
                or run['protocol_sha256']!=sha256_file(output/'protocol.json')
                or run['notes_sha256']!=sha256_file(path) or run['signature']!=reference['signature']
                or not 0<=run['forward_check']['max_error']<=p['forward_atol']
                or not 0<=run['replay_check']['max_error']<=p['replay_atol']):
            raise ValueError('All four exact complete teacher shards/parity receipts required')
        values=[json.loads(line) for line in path.read_text().splitlines()]
        if [r['item_id'] for r in values]!=ids:raise ValueError('Teacher note coverage/order differs')
        for row in values:
            item=inputs['prompts'][p['item_ids'].index(row['item_id'])];a=row['actions']
            expected_stop=f'token:{a[-1]}' if a and a[-1] in STOPS else 'length_cap'
            if (row['status']!='ok' or row['document_id']!=item['document_id']
                    or row['split']!='train' or row['task']!='gsm8k' or row['shard']!=shard or not 1<=len(a)<=CAP
                    or any(type(t) is not int or t not in range(151936) for t in a)
                    or row['content_ids']!=content_from_actions(a,STOPS[0],STOPS[1:])
                    or row['n_actions']!=len(a) or row['n_content_tokens']!=len(row['content_ids'])
                    or row['action_mask']!=[True]*len(a)+[False]*(CAP-len(a))
                    or row['stop_type']!=expected_stop or expected_stop=='length_cap' and len(a)!=CAP
                    or any(t in STOPS for t in a[:-1]) or row['greedy'] is not True or row['temperature']!=1.
                    or row['gold_or_verified_notes_exposed'] is not False
                    or row['prompt_sha256']!=canonical_hash(item['prompt_ids'])
                    or row['question_sha256']!=item['question_sha256']
                    or not math.isfinite(row['generation_wall_seconds']) or row['generation_wall_seconds']<0):
                raise ValueError('Teacher note token/stop/provenance metadata differs')
            notes[row['item_id']]=row
        first=values[0];cached=first.get('cached_logps',[]);replayed=first.get('replayed_logps',[])
        if (run['replay_check']['item_id']!=ids[0] or run['replay_check']['actions']!=len(first['actions'])
                or len(cached)!=len(first['actions']) or len(replayed)!=len(cached)
                or any(not math.isfinite(v) for v in cached+replayed)
                or max(abs(a-b) for a,b in zip(cached,replayed))!=run['replay_check']['max_error']):
            raise ValueError('First actual note cached/replay evidence differs')
        receipts.extend((d/'run.json',path))
    if set(notes)!=set(p['item_ids']):raise ValueError('Require all32 teacher notes')
    return p,inputs,notes,receipts


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('prepare','cpu-reference','generate'))
    parser.add_argument('--output',required=True,type=Path);parser.add_argument('--questions',type=Path)
    parser.add_argument('--teacher-dir',type=Path,default=Path('local/nvme/assets/qwen3-4b-teacher'))
    parser.add_argument('--base-dir',type=Path,default=Path('local/assets/qwen3-1.7b-base'))
    parser.add_argument('--shard-index',type=int,choices=range(4));parser.add_argument('--cpu-threads',type=int,default=16)
    parser.add_argument('--deadline',required=True)
    args=parser.parse_args()
    if args.command=='prepare':
        if args.questions is None:parser.error('prepare requires --questions')
        prepare(args.output,args.questions,args.teacher_dir,args.base_dir,args.deadline)
    elif args.command=='cpu-reference':cpu_reference(args.output,args.cpu_threads,args.deadline)
    elif args.shard_index is None:parser.error('generate requires --shard-index0..3')
    else:generate(args.output,args.shard_index,args.deadline)


if __name__=='__main__':main()

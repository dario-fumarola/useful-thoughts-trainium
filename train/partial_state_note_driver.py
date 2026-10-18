"""Prefix-only rollouts from the separately trained, frozen partial writer.

The model/native preparation is the frozen-model v2 protocol. This command
opens only its questions and native reference, never controls or gold. All
attempts survive, including failures, empty notes and budget truncation.
"""
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import time
from types import SimpleNamespace

from train.partial_state_evaluation import donor_permutation, prompt_tokens, score_partial_state, tree_from_question
from weekend_thinking.artifacts import atomic_json, canonical_hash, require_utc_deadline, sha256_file, utc_now
from weekend_thinking.runtime_policy import require_runtime_policy, runtime_policy

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = 'partial-state-prefix-only-writer-rollouts-v2'
LEGACY_SCHEMA = 'partial-state-TRAIN128-prefix-only-writer-rollouts-v1'
EOS, CAP, WIDTH = 151643, 64, 32


def read(path):
    return json.loads(Path(path).read_text())


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def sources():
    from train.partial_state_answer_driver import sources as common
    return {**common(), 'train/partial_state_note_driver.py': sha256_file(Path(__file__))}


def prepare(output, model_preparation, model_source_root, execution_device='xla'):
    """Bind the writer's own CPU native reference and the question projection."""
    from train.partial_state_checkpoint import verify
    if execution_device not in ('cpu','xla'):
        raise ValueError('Explicit CPU reference or XLA device required')
    model_preparation, model_source_root = Path(model_preparation).resolve(), Path(model_source_root).resolve()
    p = read(model_preparation/'protocol.json')
    if (p['schema'] not in ('partial-state-TRAIN128-complete-answers-v2','partial-state-complete-answers-v4')
            or p['reader'].get('arm') != 'writer' or p['items'] not in (128,512)):
        raise ValueError('Own frozen writer/native preparation on a fixed cohort required')
    verify(p['reader'])
    if any(sha256_file(model_source_root/k) != v for k,v in p['source_files'].items()):
        raise ValueError('Original frozen-model preparation source changed')
    inputs = [model_preparation/n for n in ('protocol.json','questions.jsonl','native-contexts.json','native-reference.pt')]
    for f in inputs[1:]:
        if sha256_file(f) != p['files'][f.name]:
            raise ValueError('Original questions or own-model native reference changed')
    questions = rows(model_preparation/'questions.jsonl')
    if [q['item_id'] for q in questions] != p['ordered_item_ids']:
        raise ValueError('Complete ordered questions required')
    for q in questions:
        tree_from_question(q)
    output.mkdir(parents=True, exist_ok=False)
    if len(questions)!=p['items'] or len({q['panel'] for q in questions})!=1:
        raise ValueError('One entire fixed cohort required')
    protocol = dict(schema=SCHEMA, created_utc=utc_now(), items=p['items'], panel=questions[0]['panel'], shards=4,
        writer=p['reader'], model_preparation=str(model_preparation),
        model_source_root=str(model_source_root), input_pins={str(f):sha256_file(f) for f in inputs},
        source_files=sources(), runtime_policy=p['runtime_policy'],
        execution_device=execution_device,
        max_actions=CAP, prefill_width=WIDTH, selection='single_greedy_EOS_only',
        writer_inputs='Question projection only; neither controls nor gold is opened.',
        limits=['TRAIN is fitting only; each DEV panel is a separate computation diagnostic, never official TEST.',
                'Capped notes remain intact. Failed notes and failed donors remain unavailable, never replaced.',
                'No utility filtering, target-aware repair or best-of selection.'])
    atomic_json(output/'protocol.json',protocol)
    return protocol


def load(output):
    p = read(output/'protocol.json')
    if p['schema'] != SCHEMA or p['source_files'] != sources():
        raise ValueError('Frozen note-generation code changed')
    for name,digest in p['input_pins'].items():
        if sha256_file(name) != digest:
            raise ValueError('Frozen model preparation or question bytes changed')
    model = read(Path(p['model_preparation'])/'protocol.json')
    if any(sha256_file(Path(p['model_source_root'])/k) != v for k,v in model['source_files'].items()):
        raise ValueError('Frozen native/model preparation source changed')
    questions = rows(Path(p['model_preparation'])/'questions.jsonl')
    for q in questions:
        tree_from_question(q)
    return p, model, questions


def generate(output, shard, until):
    import torch
    from train.evaluate import GenerationFailure, GreedyDecoder, asset_provenance
    from train.evaluate_local import setup_model
    from train.partial_state_answer_driver import cache_prefill_check
    from train.partial_state_checkpoint import verify
    from train.useful_note_precision import compare_distributions
    p, model, questions = load(output)
    if asset_provenance(Path(model['model_dir'])) != model['assets']:
        raise ValueError('Actual writer base configuration/tokenizer assets changed')
    if not 0 <= shard < 4:
        raise ValueError('Exactly four disjoint shards')
    require_runtime_policy(p['runtime_policy'],runtime_policy()); verify(p['writer'])
    deadline=require_utc_deadline(until)
    def check():
        if datetime.now(timezone.utc)>=deadline:
            raise TimeoutError('Finite note-generation deadline; no retry')
    check(); d=output/'notes'/f'shard-{shard}';d.mkdir(parents=True,exist_ok=False)
    report=dict(status='running',started_utc=utc_now(),shard=shard,rows=0,
                writer=p['writer'],execution_device=p['execution_device'],protocol_sha256=sha256_file(output/'protocol.json'))
    atomic_json(d/'run.json',report);started=time.monotonic()
    try:
        writer=p['writer'];backend,tok,loaded,signature,_=setup_model(SimpleNamespace(
            model_dir=Path(model['model_dir']),checkpoint=Path(writer['checkpoint']),device=p['execution_device']))
        if backend.device.type != p['execution_device']:
            raise ValueError('Actual note-generation device differs from the declared reference/runtime')
        if (signature!=model['evaluation_signature'] or loaded['sha256']!=writer['checkpoint_sha256']
                or any(loaded.get(k)!=writer[k] for k in ('step','run_id','spec_hash'))):
            raise ValueError('Actual writer differs from its frozen/native identity')
        prep=Path(p['model_preparation']);refs=read(prep/'native-contexts.json')['refs']
        reference=torch.load(prep/'native-reference.pt',map_location='cpu',weights_only=True)
        context=refs[shard]['context_ids'];x=torch.full((1,1024),EOS,dtype=torch.long);x[0,:len(context)]=torch.tensor(context)
        with torch.no_grad():
            hidden=backend.hidden(x.to(backend.device));lp=backend.model.lm_head(hidden[:,len(context)-1]).float().log_softmax(-1);backend.sync();actual=lp.cpu()
        gate=compare_distributions(reference[shard:shard+1],actual)[0]
        torch.save(actual,d/'native-logps.pt');atomic_json(d/'native.json',gate)
        if not gate['passed']:
            raise RuntimeError('Actual frozen writer/native parity failed')
        del hidden,lp,x,actual
        decoder=GreedyDecoder(backend.model,tok,sync=backend.sync,banned_ids=backend.banned_ids)
        prompt=prompt_tokens(questions[shard],tok,cot=True)
        audit,vectors=cache_prefill_check(backend,decoder,prompt,width=WIDTH,cap=CAP)
        torch.save(vectors,d/'cache-prefill-logps.pt');atomic_json(d/'cache-prefill.json',audit)
        if audit['max_logp_error']>1e-3:
            raise RuntimeError('Actual cached writer prefill disagrees with ordinary forward')
        with (d/'notes.jsonl').open('x') as stream:
            for index in range(shard,len(questions),4):
                check();q=questions[index];prompt=prompt_tokens(q,tok,cot=True);before=time.monotonic()
                value=dict(item_id=q['item_id'],panel=q['panel'],status='error',prompt_ids=prompt,
                    prompt_sha256=canonical_hash(prompt),actions=[],content_ids=[],text='',raw_text='',stop_type='error')
                try:
                    value.update(asdict(decoder.decode(prompt,max_new_tokens=CAP,prefill_width=WIDTH,
                        pad_id=EOS,stop_ids=(EOS,),answer_line_stop=False)),status='ok')
                except Exception as error:
                    if isinstance(error,GenerationFailure):value.update(asdict(error.partial))
                    value['error']=f'{type(error).__name__}: {error}'
                value.update(wall_seconds=time.monotonic()-before,n_actions=len(value['actions']))
                stream.write(json.dumps(value,allow_nan=False)+'\n');stream.flush();report['rows']+=1
                print(json.dumps(dict(shard=shard,rows=report['rows'],utc=utc_now())),flush=True)
        report.update(status='complete_all_attempts',notes_sha256=sha256_file(d/'notes.jsonl'))
    except BaseException as error:
        report.update(status='failed',error=f'{type(error).__name__}: {error}');raise
    finally:
        report.update(finished_utc=utc_now(),stage_wall_seconds=time.monotonic()-started)
        atomic_json(d/'run.json',report,overwrite=True)


def note_controls(questions, notes):
    """Preserve every intact note exactly once as a donor, including failures."""
    ids=[q['item_id'] for q in questions]
    if len(notes)!=len(ids) or set(notes)!=set(ids):
        raise ValueError('Every original writer attempt required')
    donors=donor_permutation([len(notes[k]['content_ids']) for k in ids],ids)
    return [dict(item_id=k,own=notes[k],other=notes[ids[j]],other_item_id=ids[j],
                 other_length_matched=len(notes[k]['content_ids'])==len(notes[ids[j]]['content_ids']))
            for k,j in zip(ids,donors)]


def collect(output):
    from transformers import AutoTokenizer
    p, model, questions=load(output)
    closure=output/'generation-closure.json'
    if read(closure)['status']!='ALL_FOUR_GENERATION_SESSIONS_AND_DEVICE_PHYSICALLY_CLOSED':
        raise ValueError('Actual entire note-generation closure required')
    tok=AutoTokenizer.from_pretrained(model['model_dir'],local_files_only=True,trust_remote_code=False)
    notes={};pins={str(closure):sha256_file(closure),str(output/'protocol.json'):sha256_file(output/'protocol.json')}
    wanted={q['item_id']:q for q in questions};defined=set(tok.get_vocab().values())
    for shard in range(4):
        d=output/'notes'/f'shard-{shard}';r=read(d/'run.json');file=d/'notes.jsonl'
        if r['status']!='complete_all_attempts' or r['rows']!=len(range(shard,p['items'],4)) or sha256_file(file)!=r['notes_sha256']:
            raise ValueError('All four complete original writer shards required')
        pins.update({str(file):sha256_file(file),str(d/'run.json'):sha256_file(d/'run.json')})
        expected=[questions[i]['item_id'] for i in range(shard,len(questions),4)];values=rows(file)
        if [v['item_id'] for v in values]!=expected:
            raise ValueError('Original writer order changed')
        for v in values:
            q=wanted[v['item_id']];prompt=prompt_tokens(q,tok,cot=True);a=v['actions']
            if v['item_id'] in notes or v['prompt_ids']!=prompt or v['prompt_sha256']!=canonical_hash(prompt):
                raise ValueError('Writer input was not the exact prefix projection')
            if (len(a)>CAP or EOS in a[:-1] or any(type(x)is not int or x not in defined for x in a)
                    or v['content_ids']!=(a[:-1] if a and a[-1]==EOS else a)):
                raise ValueError('Native content/actions/stop inventory changed')
            if v['status']=='ok':
                if (v['stop_type'] not in ('token:151643','length_cap')
                        or (v['stop_type']=='length_cap' and len(a)!=CAP)
                        or (v['stop_type']=='token:151643' and (not a or a[-1]!=EOS))
                        or v['text']!=tok.decode(v['content_ids'],skip_special_tokens=False,clean_up_tokenization_spaces=False)):
                    raise ValueError('Successful native writer trace is inconsistent')
            elif v['status']!='error':raise ValueError('Unknown writer outcome')
            notes[v['item_id']]=v
    controls=note_controls(questions,notes)
    states={q['item_id']:score_partial_state(notes[q['item_id']]['text'],q['expression_tree'],
                stop_type=notes[q['item_id']]['stop_type'],status=notes[q['item_id']]['status']) for q in questions}
    bundle=dict(schema=SCHEMA,status='complete_all_attempts',finished_utc=utc_now(),
        writer=p['writer'],source_pins=pins,ordered_item_ids=[q['item_id'] for q in questions],
        execution_device=p['execution_device'],
        controls=controls,state_diagnostics=states,
        limits=['Canonical-state diagnostics are task-specific; different valid strategies may not match.',
                'Failed own/other notes remain failures; no imputation or filtering.'])
    atomic_json(output/'notes-bundle.json',bundle)
    return bundle


def load_bundle(path, ordered_ids):
    value=read(path)
    if (value['schema'] not in (SCHEMA,LEGACY_SCHEMA) or value['status']!='complete_all_attempts'
            or value['ordered_item_ids']!=ordered_ids):
        raise ValueError('Complete same-cohort prefix-only writer bundle required')
    for name,digest in value['source_pins'].items():
        if sha256_file(name)!=digest:raise ValueError('Original writer attempts changed')
    originals={v['item_id']:v for name in value['source_pins'] if Path(name).name=='notes.jsonl' for v in rows(name)}
    expected=note_controls([dict(item_id=k) for k in ordered_ids],originals)
    if value['controls']!=expected:
        raise ValueError('Reader controls differ from actual original writer outputs')
    protocols=[read(name) for name in value['source_pins'] if Path(name).name=='protocol.json']
    if len(protocols)!=1 or protocols[0]['schema']!=value['schema'] or protocols[0]['writer']!=value['writer']:
        raise ValueError('Actual producer writer identity differs')
    return value


def main():
    a=argparse.ArgumentParser(description=__doc__);a.add_argument('command',choices=('prepare','generate','collect'))
    a.add_argument('--output',type=Path,required=True);a.add_argument('--model-preparation',type=Path);a.add_argument('--model-source-root',type=Path)
    a.add_argument('--shard',type=int);a.add_argument('--deadline-utc');a.add_argument('--device',choices=('cpu','xla'),default='xla');v=a.parse_args()
    if v.command=='prepare':result=prepare(v.output,v.model_preparation,v.model_source_root,v.device)
    elif v.command=='generate':result=generate(v.output,v.shard,v.deadline_utc)
    else:result=collect(v.output)
    print(json.dumps(None if result is None else {k:x for k,x in result.items() if k not in ('controls','state_diagnostics')},allow_nan=False))


if __name__=='__main__':
    main()

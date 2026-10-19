"""Frozen single-model prefix-only calculation rollouts on fixed real DEV."""
import argparse,json,time
from pathlib import Path
from types import SimpleNamespace
from dataclasses import asdict
from datetime import datetime,timezone
from train.real_trace_contexts import writer_prompt,boundary_stop_ids
from weekend_thinking.artifacts import atomic_json,sha256_file,utc_now,require_utc_deadline

ROOT=Path(__file__).resolve().parents[1]
def read(p):return json.loads(Path(p).read_text())
def rows(p):return [json.loads(s) for s in Path(p).read_text().splitlines() if s.strip()]
def sources():
    from train.evaluation_integrity import evaluation_dependencies
    from train.partial_state_driver import source_files
    return {**evaluation_dependencies(),**source_files(),**{n:sha256_file(ROOT/n) for n in (
        'train/real_trace_notes.py','train/real_trace_contexts.py','train/partial_state_answer_driver.py',
        'train/partial_state_evaluation.py')}}


def prepare(output,inputs,panel,model_dir,base_protocol,binding=None):
    import torch
    from train.evaluate import asset_provenance
    from train.evaluate_local import setup_model
    from train.useful_note_sft_driver import load_native_reference
    torch.set_num_threads(16)
    manifest=read(inputs/'manifest.json')
    if panel not in ('DEV_AST256','DEV_legacy256') or not manifest['panels'][panel]['arithmetic']:
        raise ValueError('Declared arithmetic DEV panel required;neverTEST')
    questions=inputs/f'{panel}-questions.jsonl'
    if sha256_file(questions)!=manifest['files'][questions.name]['sha256']:
        raise ValueError('Original fixed prefix questions changed')
    qs=rows(questions)
    if len(qs)!=256 or len({q['item_id'] for q in qs})!=256:
        raise ValueError('All256unique prefix questions required')
    model=read(binding) if binding else dict(id='B0',checkpoint=None)
    if binding and (model['status']!='FROZEN_AUDITED_REAL_TRACE_TERMINAL320' or model['step']!=320
            or sha256_file(model['checkpoint'])!=model['checkpoint_sha256']
            or sha256_file(model['audit_file'])!=model['audit_sha256']):
        raise ValueError('Original independently audited terminal320 required')
    assets=asset_provenance(model_dir);base=read(base_protocol)
    refs,native=load_native_reference(base,assets);signature=base['base_signature']
    if binding:
        backend,tok,loaded,signature,_=setup_model(SimpleNamespace(model_dir=model_dir,checkpoint=Path(model['checkpoint']),device='cpu'))
        if loaded['sha256']!=model['checkpoint_sha256'] or any(loaded[k]!=model[k] for k in ('run_id','step','spec_hash')):
            raise ValueError('Actual CPU model differs from audited binding')
        values=[]
        with torch.no_grad():
            for ref in refs:
                x=torch.full((1,1024),151643,dtype=torch.long);x[0,:len(ref['context_ids'])]=torch.tensor(ref['context_ids'])
                hidden=backend.hidden(x);values.append(backend.model.lm_head(hidden[:,len(ref['context_ids'])-1]).float().log_softmax(-1))
        native=torch.cat(values).cpu()
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(model_dir,local_files_only=True)
    if any(len(writer_prompt(q,tok))>544 for q in qs):raise ValueError('Declared native prefill too short')
    output.mkdir(parents=True,exist_ok=False)
    torch.save(native,output/'native-reference.pt')
    atomic_json(output/'native-contexts.json',dict(refs=refs))
    # Explicit hardware policy;CPU reference thread count is a separate fact.
    runtime=dict(schema=1,environment=dict(MKL_NUM_THREADS='2',OMP_NUM_THREADS='2',PJRT_DEVICE='NEURON',
       NEURON_CC_FLAGS='--model-type=transformer',NEURON_NUM_RECENT_MODELS_TO_KEEP=None,
       XLA_COMPILATION_CACHE_SIZE='256',XLA_DOWNCAST_BF16=None,XLA_USE_BF16=None))
    p=dict(schema='real-trace-prefix-only-notes-v10',created_utc=utc_now(),items=256,shards=4,
      panel=panel,model=model,model_dir=str(model_dir),assets=assets,signature=signature,
      questions=str(questions),questions_sha256=sha256_file(questions),ordered_item_ids=[q['item_id'] for q in qs],
      native_files={n:sha256_file(output/n) for n in ('native-reference.pt','native-contexts.json')},
      runtime_policy=runtime,source_files=sources(),max_actions=128,prefill_width=544,
      stop_ids=boundary_stop_ids(tok),CPU_reference_threads=16,
      limits=['Generationopensonlyprefixquestionsandnativereference,neverscoringtargets.','FirstnativeAnswerboundarytokenorEOSstops;contentremainsexactincludingcaps/errors.'])
    atomic_json(output/'protocol.json',p);return p


def load(output):
    p=read(output/'protocol.json')
    if p.get('schema')!='real-trace-prefix-only-notes-v10' or p.get('items')!=256 or p.get('shards')!=4:
        raise ValueError('Require the fixed native real-document256generation schema')
    if p['source_files']!=sources():raise ValueError('Frozen generation implementation changed')
    if sha256_file(p['questions'])!=p['questions_sha256']:raise ValueError('Frozen prefix projection changed')
    for n,v in p['native_files'].items():
        if sha256_file(output/n)!=v:raise ValueError('Actual own-model native reference changed')
    return p,rows(p['questions'])


def generate(output,shard,until):
    import torch
    from train.evaluate import GreedyDecoder,GenerationFailure,asset_provenance
    from train.evaluate_local import setup_model
    from train.partial_state_answer_driver import cache_prefill_check
    from train.useful_note_precision import compare_distributions
    from weekend_thinking.runtime_policy import runtime_policy,require_runtime_policy
    p,qs=load(output);deadline=require_utc_deadline(until)
    def check():
        if datetime.now(timezone.utc)>=deadline:raise TimeoutError('Finite generation deadline;no retry')
    check()
    if not 0<=shard<4:raise ValueError('Explicit four-shard generation required')
    require_runtime_policy(p['runtime_policy'],runtime_policy())
    if asset_provenance(Path(p['model_dir']))!=p['assets']:raise ValueError('Released assets changed')
    model=p['model']
    if model['checkpoint'] and sha256_file(model['checkpoint'])!=model['checkpoint_sha256']:
        raise ValueError('Frozen terminal model changed')
    d=output/'notes'/f'shard-{shard}';d.mkdir(parents=True,exist_ok=False)
    report=dict(status='running',started_utc=utc_now(),rows=0,shard=shard,protocol_sha256=sha256_file(output/'protocol.json'))
    atomic_json(d/'run.json',report);started=time.monotonic()
    try:
        backend,tok,loaded,signature,_=setup_model(SimpleNamespace(model_dir=Path(p['model_dir']),
            checkpoint=Path(model['checkpoint']) if model['checkpoint'] else None,device='xla'))
        if signature!=p['signature']:raise ValueError('Transferred model differs from CPU reference')
        if model['checkpoint'] and (loaded['sha256']!=model['checkpoint_sha256'] or any(loaded[k]!=model[k] for k in ('run_id','step','spec_hash'))):
            raise ValueError('Actual generation checkpoint differs')
        refs=read(output/'native-contexts.json')['refs'];reference=torch.load(output/'native-reference.pt',map_location='cpu',weights_only=True)
        ctx=refs[shard]['context_ids'];x=torch.full((1,1024),151643,dtype=torch.long);x[0,:len(ctx)]=torch.tensor(ctx)
        with torch.no_grad():
            hidden=backend.hidden(x.to(backend.device));lp=backend.model.lm_head(hidden[:,len(ctx)-1]).float().log_softmax(-1);backend.sync();actual=lp.cpu()
        gate=compare_distributions(reference[shard:shard+1],actual)[0]
        torch.save(actual,d/'native-logps.pt');atomic_json(d/'native.json',gate)
        if not gate['passed']:raise RuntimeError('Actual own-model native parity failed')
        del x,hidden,lp,actual
        decoder=GreedyDecoder(backend.model,tok,sync=backend.sync,banned_ids=backend.banned_ids)
        audit,vectors=cache_prefill_check(backend,decoder,writer_prompt(qs[shard],tok),width=544,cap=128)
        torch.save(vectors,d/'cache-prefill-logps.pt');atomic_json(d/'cache-prefill.json',audit)
        if audit['max_logp_error']>1e-3:raise RuntimeError('Actual cached shape differs from ordinary causal forward')
        with (d/'notes.jsonl').open('x') as stream:
            for index in range(shard,256,4):
                check();q=qs[index];prompt=writer_prompt(q,tok);before=time.monotonic()
                value=dict(item_id=q['item_id'],document_id=q['document_id'],status='error',actions=[],content_ids=[],text='',raw_text='',stop_type='error',prompt_ids=prompt)
                try:
                    value.update(asdict(decoder.decode(prompt,max_new_tokens=128,prefill_width=544,pad_id=151643,
                      stop_ids=tuple(p['stop_ids']),answer_line_stop=False)),status='ok')
                except Exception as error:
                    if isinstance(error,GenerationFailure):value.update(asdict(error.partial))
                    value['error']=f'{type(error).__name__}: {error}'
                value.update(n_actions=len(value['actions']),wall_seconds=time.monotonic()-before)
                stream.write(json.dumps(value,allow_nan=False)+'\n');stream.flush();report['rows']+=1
                print(json.dumps(dict(utc=utc_now(),shard=shard,rows=report['rows'])),flush=True)
        report.update(status='complete_all_attempts',notes_sha256=sha256_file(d/'notes.jsonl'))
    except BaseException as error:
        report.update(status='failed',error=f'{type(error).__name__}: {error}');raise
    finally:
        report.update(finished_utc=utc_now(),wall_seconds=time.monotonic()-started)
        atomic_json(d/'run.json',report,overwrite=True)


def collect(output):
    from train.partial_state_evaluation import donor_permutation
    p,qs=load(output)
    closure=output/'generation-closure.json'
    if read(closure)['status']!='ALL_FOUR_GENERATION_SESSIONS_AND_DEVICE_PHYSICALLY_CLOSED':
        raise ValueError('Actual generation closure required')
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(p['model_dir'],local_files_only=True)
    defined=set(tok.get_vocab().values());by_id={q['item_id']:q for q in qs}
    originals={};pins={str(closure):sha256_file(closure),str(output/'protocol.json'):sha256_file(output/'protocol.json')}
    for shard in range(4):
        d=output/'notes'/f'shard-{shard}';run=read(d/'run.json');values=rows(d/'notes.jsonl')
        if run['status']!='complete_all_attempts' or run['rows']!=64 or run['protocol_sha256']!=sha256_file(output/'protocol.json') or sha256_file(d/'notes.jsonl')!=run['notes_sha256']:
            raise ValueError('Every original note attempt required')
        if [v['item_id'] for v in values]!=[q['item_id'] for q in qs[shard::4]]:
            raise ValueError('Original question order changed')
        for v in values:
            q=by_id[v['item_id']]
            if v['prompt_ids']!=writer_prompt(q,tok) or v['document_id']!=q['document_id']:
                raise ValueError('Actual rollout did not use its exact causal prefix')
            a=v['actions'];stopped=bool(a and a[-1] in p['stop_ids'])
            if any(type(t) is not int or t not in defined for t in a) or any(t in p['stop_ids'] for t in a[:-1]) or len(a)>128 or v['content_ids']!=(a[:-1] if stopped else a) or v['status'] not in ('ok','error'):
                raise ValueError('Native content or status changed')
            if v['status']=='ok' and (v['stop_type']!=(f'token:{a[-1]}' if stopped else 'length_cap') or not stopped and len(a)!=128):
                raise ValueError('Natural stop/cap inventory changed')
            if v['text']!=tok.decode(v['content_ids'],skip_special_tokens=False,clean_up_tokenization_spaces=False):
                raise ValueError('Saved text differs from intact native content')
            originals[v['item_id']]=v
        for name in ('run.json','notes.jsonl','native.json','native-logps.pt','cache-prefill.json','cache-prefill-logps.pt'):pins[str(d/name)]=sha256_file(d/name)
    ids=p['ordered_item_ids'];donors=donor_permutation([len(originals[k]['content_ids']) for k in ids],ids)
    controls=[dict(item_id=k,own=originals[k],other=originals[ids[j]],other_item_id=ids[j]) for k,j in zip(ids,donors)]
    bundle=dict(status='COMPLETE_REAL_PREFIX_ONLY_NOTE_BUNDLE',model_id=p['model']['id'],ordered_item_ids=ids,source_pins=pins,controls=controls,
      limits=['Intactdonorderangement;norepair,trim,filterorbest-ofselection.','Actualpartialerrorspreservedandmakeprimarygateineligible;naturalempty/cappednotesremainvalidobservations.'])
    atomic_json(output/'notes-bundle.json',bundle);return bundle


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=('prepare','generate','collect'))
    for n in ('output','inputs','model-dir','base-protocol','binding'):p.add_argument('--'+n,type=Path,required=n=='output')
    p.add_argument('--panel');p.add_argument('--shard',type=int);p.add_argument('--deadline-utc');a=p.parse_args()
    if a.command=='prepare':v=prepare(a.output,a.inputs,a.panel,a.model_dir,a.base_protocol,a.binding)
    elif a.command=='generate':v=generate(a.output,a.shard,a.deadline_utc)
    else:v=collect(a.output)
    print(json.dumps(None if v is None else {k:x for k,x in v.items() if k!='controls'},allow_nan=False))


if __name__=='__main__':main()

"""Frozen complete-answer QA generation; scoring is a separate, later action.

Generation opens only its per-mode prompt projections and native references.
The prepare command may use offline gold to build explicitly named oracle and
evidence-ablation conditions. Ordinary/prior prompts contain no gold annotations.
"""
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import time
from types import SimpleNamespace

from train.document_qa_contexts import context, question_only, omit_evidence, trace_from_output
from weekend_thinking.artifacts import atomic_json, sha256_file, utc_now, require_utc_deadline

ROOT=Path(__file__).resolve().parents[1]
SCHEMA='document-QA-complete-answers-v21'
MODES=('direct','prior','empty','reference_cot','question_only_direct',
       'question_only_prior','bridge_removed_prior','oracle')


def read(path):return json.loads(Path(path).read_text())
def rows(path):return [json.loads(v) for v in Path(path).read_text().splitlines()]


def sources():
    from train.evaluation_integrity import evaluation_dependencies
    names=('train/document_qa_evaluate.py','train/document_qa_contexts.py',
        'train/document_qa_decoder.py','train/document_qa_answers.py',
        'train/document_qa_data.py','train/document_qa_packing.py',
        'train/document_evidence_data.py','train/partial_state_answer_driver.py',
        'train/useful_note_precision.py','train/useful_note_sft_driver.py')
    return {**evaluation_dependencies(),**{n:sha256_file(ROOT/n) for n in names}}


def prompt_text(q,mode,gold=None):
    if mode=='question_only_direct':return context(question_only(q),'direct'),'direct'
    if mode=='question_only_prior':return context(question_only(q),'prior'),'prior'
    if mode=='bridge_removed_prior':
        import re
        first=re.search(r'\nEvidence \[(\d+)\]:',gold['trace'])
        if first is None:raise ValueError('Explicit annotated first-hop citation required')
        index=int(first[1])-1
        if index not in gold['support_indices']:raise ValueError('First-hop evidence differs')
        return context(omit_evidence(q,index),'prior'),'prior'
    if mode=='oracle':return context(q,'replay',gold['trace']),'direct'
    return context(q,mode),'prior' if mode in ('prior','reference_cot') else 'direct'


def prepare(output,data_dir,model_dir,base_protocol,*,qualification=False,modes=None,binding=None,release=None):
    import torch
    from transformers import AutoTokenizer
    from train.evaluate import asset_provenance
    from train.evaluate_local import setup_model
    from train.useful_note_sft_driver import load_native_reference
    output,data_dir,model_dir,base_protocol=(Path(v).resolve() for v in (output,data_dir,model_dir,base_protocol))
    binding=Path(binding).resolve() if binding else None
    release=Path(release).resolve() if release else None
    manifest=read(data_dir/'manifest.json')
    if manifest['status']!='PASS_ALL_6144_NATIVE_QA_VIEWS_AND_EVIDENCE_CLEAN_COHORTS_CPU_ONLY':
        raise ValueError('Frozen complete QA data required')
    if qualification:
        if binding is not None or release is not None:raise ValueError('Disposable unchanged-B0 qualification only')
        data_path=Path(manifest['files']['training']['path'])
        if sha256_file(data_path)!=manifest['files']['training']['sha256']:raise ValueError('TRAIN data changed')
        pool=rows(data_path)[:128]
        tok=AutoTokenizer.from_pretrained(model_dir,local_files_only=True,trust_remote_code=False)
        # Longest and shortest inputs in the disposable qualification TRAIN128.
        ranked=sorted((r['question'] for r in pool),key=lambda q:len(tok.encode(context(q,'direct'),add_special_tokens=False)))
        questions=[ranked[-1],ranked[0]];golds={}
        allowed=('direct','prior','reference_cot')
    else:
        if release is None:raise ValueError('Actual scientific release required before held model forwards')
        r=read(release)
        if r.get('status')!='RELEASED_DOCUMENT_QA_V21_SCIENCE_AND_FIXED_EVALUATION':
            raise ValueError('No implicit science release')
        if r['QA_data_manifest_sha256']!=sha256_file(data_dir/'manifest.json'):
            raise ValueError('Scientific data binding differs')
        f=manifest['files']['held_questions'];g=manifest['files']['held_gold']
        if sha256_file(f['path'])!=f['sha256'] or sha256_file(g['path'])!=g['sha256']:raise ValueError('Frozen held data changed')
        questions=rows(f['path']);golds={v['item_id']:v for v in rows(g['path'])}
        allowed=MODES
    modes=list(allowed if modes is None else modes)
    if not modes or len(set(modes))!=len(modes) or any(v not in allowed for v in modes):
        raise ValueError('Explicit unique permitted modes required')
    model=read(binding) if binding else dict(id='B0',checkpoint=None)
    if binding and (model.get('status')!='AUDITED_DOCUMENT_QA_V21_TERMINAL256' or model.get('step')!=256
            or sha256_file(model['checkpoint'])!=model['checkpoint_sha256']
            or sha256_file(model['audit_file'])!=model['audit_sha256']):
        raise ValueError('Original independently audited terminal256 required')
    assets=asset_provenance(model_dir);base=read(base_protocol)
    refs,native=load_native_reference(base,assets);signature=base['base_signature']
    torch.set_num_threads(16)
    if binding:
        backend,_,loaded,signature,_=setup_model(SimpleNamespace(model_dir=model_dir,
            checkpoint=Path(model['checkpoint']),device='cpu'))
        if any(loaded.get(k)!=model[k] for k in ('run_id','step','spec_hash')):raise ValueError('Loaded CPU model differs')
        values=[]
        with torch.no_grad():
            for ref in refs:
                x=torch.full((1,1024),151643,dtype=torch.long);x[0,:len(ref['context_ids'])]=torch.tensor(ref['context_ids'])
                h=backend.hidden(x);values.append(backend.model.lm_head(h[:,len(ref['context_ids'])-1]).float().log_softmax(-1))
        native=torch.cat(values).cpu()
        del backend,h,x,values
    tok=AutoTokenizer.from_pretrained(model_dir,local_files_only=True,trust_remote_code=False)
    output.mkdir(parents=True,exist_ok=False)
    torch.save(native,output/'native-reference.pt');atomic_json(output/'native-contexts.json',dict(refs=refs))
    files={n:sha256_file(output/n) for n in ('native-reference.pt','native-contexts.json')}
    maximum={}
    for mode in modes:
        path=output/f'prompts-{mode}.jsonl';lengths=[]
        with path.open('x') as f:
            for q in questions:
                text,parser_mode=prompt_text(q,mode,golds.get(q['item_id']))
                ids=tok.encode(text,add_special_tokens=False)
                if len(ids)>3584:raise ValueError('Complete prompt exceeds frozen prefill;never truncate')
                value=dict(item_id=q['item_id'],mode=mode,prompt_ids=ids,parser_mode=parser_mode)
                f.write(json.dumps(value)+'\n');lengths.append(len(ids))
        maximum[mode]=max(lengths);files[path.name]=sha256_file(path)
    runtime=dict(schema=1,environment=dict(MKL_NUM_THREADS='2',OMP_NUM_THREADS='2',PJRT_DEVICE='NEURON',
       NEURON_CC_FLAGS='--model-type=transformer',NEURON_NUM_RECENT_MODELS_TO_KEEP=None,
       XLA_COMPILATION_CACHE_SIZE='256',XLA_DOWNCAST_BF16=None,XLA_USE_BF16=None))
    p=dict(schema=SCHEMA,purpose='TRAIN_MECHANICS_ONLY' if qualification else 'FIXED_HELD_DOCUMENT_QA',
        created_utc=utc_now(),items=len(questions),modes=modes,shards=1 if qualification else 4,
        data_manifest_sha256=sha256_file(data_dir/'manifest.json'),model_dir=str(model_dir),model=model,
        assets=assets,signature=signature,source_files=sources(),files=files,runtime_policy=runtime,
        prefill_width=3584,max_actions=256,maximum_prompt_tokens=maximum,compute='FP32',
        question_file=None if qualification else manifest['files']['held_questions']['path'],
        question_file_sha256=None if qualification else manifest['files']['held_questions']['sha256'],
        ordered_item_ids=[q['item_id'] for q in questions],stop_ids=[151643,151645],
        release_file=None if qualification else str(release),release_sha256=None if qualification else sha256_file(release),
        limits=['Generation opens only per-mode prompts/native references,not original labeled rows or held gold.',
                'Oracle and evidence-removal prompts are offline controls,not available generated thoughts.',
                'All original actions/text/failures/caps retained;scoring occurs separately after full study closure.'])
    atomic_json(output/'protocol.json',p);return p


def prepare_replays(original,output):
    """Fixed cyclic donor controls after all original prior outputs close.

    Original single-stream inference remains the primary observation. Replays
    retokenize the generated trace TEXT under one shared serialization; own
    replay is therefore an explicit control rather than silent response reuse.
    """
    from transformers import AutoTokenizer
    import shutil
    original,output=Path(original).resolve(),Path(output).resolve()
    p=read(original/'protocol.json')
    if (p['source_files']!=sources() or p['purpose']!='FIXED_HELD_DOCUMENT_QA'
            or 'prior' not in p['modes'] or p['items']!=512 or p['shards']!=4
            or sha256_file(p['question_file'])!=p['question_file_sha256']):
        raise ValueError('Complete original frozen held-prior execution required')
    notes={};origins={}
    for shard in range(4):
        d=original/'answers'/f'shard-{shard}';report=read(d/'run.json')
        if (report['status']!='complete_all_attempts'
                or report['protocol_sha256']!=sha256_file(original/'protocol.json')
                or report['attempts_sha256']!=sha256_file(d/'attempts.jsonl')):
            raise ValueError('All original generation shards must close before controls')
        attempts=rows(d/'attempts.jsonl');origins[str(d/'attempts.jsonl')]=sha256_file(d/'attempts.jsonl')
        for row in attempts:
            if row['mode']=='prior':
                if row['item_id'] in notes:raise ValueError('Duplicate prior observation')
                notes[row['item_id']]=row
    if set(notes)!=set(p['ordered_item_ids']):raise ValueError('All512 original prior attempts,including failures,required')
    qs=rows(p['question_file']);tok=AutoTokenizer.from_pretrained(p['model_dir'],local_files_only=True,trust_remote_code=False)
    if [q['item_id'] for q in qs]!=p['ordered_item_ids']:
        raise ValueError('Original ordered question cohort required')
    output.mkdir(parents=True,exist_ok=False);files={};maximum={}
    for name in ('native-reference.pt','native-contexts.json'):
        if sha256_file(original/name)!=p['files'][name]:raise ValueError('Native reference changed')
        shutil.copyfile(original/name,output/name);files[name]=sha256_file(output/name)
    for mode in ('own_replay','other_replay'):
        lengths=[];path=output/f'prompts-{mode}.jsonl'
        with path.open('x') as f:
            for i,q in enumerate(qs):
                donor=qs[i if mode=='own_replay' else (i+1)%512]['item_id']
                attempt=notes[donor];trace,had_boundary=trace_from_output(attempt['text'])
                text=context(q,'replay',trace);ids=tok.encode(text,add_special_tokens=False)
                if len(ids)>3584:raise ValueError('Replay over frozen width;never trim a donor')
                value=dict(item_id=q['item_id'],mode=mode,prompt_ids=ids,parser_mode='direct',
                           donor_item_id=donor,donor_status=attempt['status'],donor_stop_type=attempt['stop_type'],
                           donor_had_final_boundary=had_boundary,trace_text=trace)
                f.write(json.dumps(value)+'\n');lengths.append(len(ids))
        files[path.name]=sha256_file(path);maximum[mode]=max(lengths)
    updated=dict(p,modes=['own_replay','other_replay'],files=files,maximum_prompt_tokens=maximum,
                 created_utc=utc_now(),original_protocol_sha256=sha256_file(original/'protocol.json'),
                 original_prior_attempts=origins,
                 replay_rule='Complete generated trace text;fixed next-item cyclic derangement;no trimming,repair,length matching or utility selection. Failed/capped donor traces retained. Original prior primary is distinct.')
    atomic_json(output/'protocol.json',updated);return updated


def generate(output,shard,until,*,schema=SCHEMA,expected_sources=None):
    import torch
    from train.document_qa_decoder import QADecoder
    from train.evaluate import GenerationFailure,asset_provenance
    from train.evaluate_local import setup_model
    from train.partial_state_answer_driver import cache_prefill_check
    from train.useful_note_precision import compare_distributions
    from weekend_thinking.runtime_policy import runtime_policy,require_runtime_policy
    output=Path(output).resolve()
    p=read(output/'protocol.json');deadline=require_utc_deadline(until)
    def check():
        if datetime.now(timezone.utc)>=deadline:raise TimeoutError('Finite evaluation deadline;no retry')
    check()
    if p['schema']!=schema or p['source_files']!=(sources() if expected_sources is None else expected_sources) or not 0<=shard<p['shards']:
        raise ValueError('Frozen generation source/shard required')
    if type(p.get('max_actions')) is not int or p['max_actions'] not in (128,256):
        raise ValueError('Prospectively fixed128/256-action bound required')
    for name,digest in p['files'].items():
        if sha256_file(output/name)!=digest:raise ValueError('Generation prompt or reference changed')
    if p['release_file'] is not None and sha256_file(p['release_file'])!=p['release_sha256']:
        raise ValueError('Scientific release changed')
    require_runtime_policy(p['runtime_policy'],runtime_policy())
    if asset_provenance(Path(p['model_dir']))!=p['assets']:raise ValueError('Released model assets changed')
    d=output/'answers'/f'shard-{shard}';d.mkdir(parents=True,exist_ok=False)
    report=dict(status='running',started_utc=utc_now(),rows=0,shard=shard,protocol_sha256=sha256_file(output/'protocol.json'))
    atomic_json(d/'run.json',report);started=time.monotonic()
    try:
        model=p['model'];path=model['checkpoint']
        if path and sha256_file(path)!=model['checkpoint_sha256']:raise ValueError('Checkpoint bytes changed')
        backend,tok,loaded,signature,_=setup_model(SimpleNamespace(model_dir=Path(p['model_dir']),checkpoint=Path(path) if path else None,device='xla'))
        if signature!=p['signature']:raise ValueError('Transferred model signature differs')
        if path and any(loaded.get(k)!=model[k] for k in ('run_id','step','spec_hash')):raise ValueError('Loaded model lineage differs')
        refs=read(output/'native-contexts.json')['refs'];reference=torch.load(output/'native-reference.pt',map_location='cpu',weights_only=True)
        i=shard%len(refs);ids=refs[i]['context_ids'];x=torch.full((1,1024),151643,dtype=torch.long);x[0,:len(ids)]=torch.tensor(ids)
        with torch.no_grad():
            h=backend.hidden(x.to(backend.device));lp=backend.model.lm_head(h[:,len(ids)-1]).float().log_softmax(-1);backend.sync();actual=lp.cpu()
        gate=compare_distributions(reference[i:i+1],actual)[0];torch.save(actual,d/'native-logps.pt');atomic_json(d/'native.json',gate)
        if not gate['passed']:raise RuntimeError('Actual own-model native parity failed')
        del h,lp,x,actual
        decoder=QADecoder(backend.model,tok,sync=backend.sync,banned_ids=backend.banned_ids)
        first=rows(output/f'prompts-{p["modes"][0]}.jsonl')[shard]
        audit,vectors=cache_prefill_check(backend,decoder,first['prompt_ids'],width=3584,cap=p['max_actions'])
        torch.save(vectors,d/'cache-prefill-logps.pt');atomic_json(d/'cache-prefill.json',audit)
        if audit['max_logp_error']>1e-3:raise RuntimeError('Long cached prefill differs from ordinary causal forward')
        with (d/'attempts.jsonl').open('x') as f:
            for mode in p['modes']:
                prompts=rows(output/f'prompts-{mode}.jsonl')
                if len(prompts)!=p['items'] or [v['item_id'] for v in prompts]!=p['ordered_item_ids']:raise ValueError('Complete ordered prompt cohort required')
                for index in range(shard,p['items'],p['shards']):
                    check();q=prompts[index];before=time.monotonic()
                    value=dict(item_id=q['item_id'],mode=mode,parser_mode=q['parser_mode'],status='error',actions=[],content_ids=[],text='',raw_text='',stop_type='error',prompt_tokens=len(q['prompt_ids']))
                    try:value.update(asdict(decoder.decode(q['prompt_ids'],mode=q['parser_mode'],max_new_tokens=p['max_actions'],prefill_width=3584)),status='ok')
                    except Exception as error:
                        if isinstance(error,GenerationFailure):value.update(asdict(error.partial))
                        value['error']=f'{type(error).__name__}: {error}'
                    value.update(n_actions=len(value['actions']),wall_seconds=time.monotonic()-before)
                    f.write(json.dumps(value,allow_nan=False)+'\n');f.flush();report['rows']+=1
                    print(json.dumps(dict(utc=utc_now(),mode=mode,rows=report['rows'])),flush=True)
        report.update(status='complete_all_attempts',attempts_sha256=sha256_file(d/'attempts.jsonl'))
    except BaseException as error:
        report.update(status='failed',error=f'{type(error).__name__}: {error}');raise
    finally:
        report.update(finished_utc=utc_now(),wall_seconds=time.monotonic()-started)
        atomic_json(d/'run.json',report,overwrite=True)


def main():
    a=argparse.ArgumentParser(description=__doc__);a.add_argument('command',choices=('prepare','prepare-replays','generate'))
    a.add_argument('--output',type=Path,required=True)
    for name in ('data-dir','model-dir','base-protocol','binding','release'):a.add_argument('--'+name,type=Path)
    a.add_argument('--qualification',action='store_true');a.add_argument('--modes',nargs='+',choices=MODES)
    a.add_argument('--shard',type=int);a.add_argument('--until');args=a.parse_args()
    if args.command=='prepare':
        if any(v is None for v in (args.data_dir,args.model_dir,args.base_protocol)):a.error('Explicit prepare data/model/base paths required')
        prepare(args.output,args.data_dir,args.model_dir,args.base_protocol,qualification=args.qualification,modes=args.modes,binding=args.binding,release=args.release)
    elif args.command=='prepare-replays':
        if args.data_dir is None:a.error('--data-dir is the complete original model execution directory for replay preparation')
        prepare_replays(args.data_dir,args.output)
    else:
        if args.shard is None or args.until is None:a.error('Explicit shard/deadline required')
        generate(args.output,args.shard,args.until)


if __name__=='__main__':main()

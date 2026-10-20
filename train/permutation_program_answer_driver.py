"""Bounded TRAIN-only program decoder qualification; no science release here.

Use the existing loader, native distribution checks and causal-cache decoder.
Generation reads frozen prompt IDs, not labels or interpreters. Learned-model
or held evaluation is deliberately absent until a prospective study is released.
"""
import argparse
from dataclasses import asdict
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import time
from types import SimpleNamespace

from train import document_qa_evaluate as engine
from train.permutation_program_evaluation import MODES,available_item,controls,prompt_ids
from weekend_thinking.artifacts import atomic_json,require_utc_deadline,sha256_file,utc_now

ROOT=Path(__file__).resolve().parents[1]
SCHEMA='permutation-program-decoder-qualification-v28'


def sources():
    names=('train/permutation_program_answer_driver.py','train/permutation_program_decoder.py',
           'train/permutation_program_answers.py','train/permutation_program_evaluation.py',
           'train/permutation_program_data.py','train/permutation_program_packing.py')
    return {**engine.sources(),**{n:sha256_file(ROOT/n) for n in names}}


def prepare(output,data_dir,model_dir,base_protocol):
    import torch
    from transformers import AutoTokenizer
    from train.evaluate import asset_provenance
    from train.useful_note_sft_driver import load_native_reference
    output,data_dir,model_dir,base_protocol=[Path(v).resolve() for v in (output,data_dir,model_dir,base_protocol)]
    manifest=engine.read(data_dir/'manifest.json');path=data_dir/'TRAIN-questions.jsonl'
    if (manifest.get('schema')!='permutation-program-v28-cpu-v1'
            or sha256_file(path)!=manifest['files'][path.name]):
        raise ValueError('Pinned questions-only TRAIN data required')
    pool=engine.rows(path)[:128]
    interventions=controls(pool);questions=[]
    for depth in (1,2,4,6):
        candidates=[v for v in pool if len(available_item(v).program)==depth]
        questions.append(min(candidates,key=lambda v:hashlib.sha256(v['item_id'].encode()).hexdigest()))
    tok=AutoTokenizer.from_pretrained(model_dir,local_files_only=True,trust_remote_code=False)
    assets=asset_provenance(model_dir);base=engine.read(base_protocol)
    refs,native=load_native_reference(base,assets)
    output.mkdir(parents=True,exist_ok=False)
    torch.save(native,output/'native-reference.pt');atomic_json(output/'native-contexts.json',dict(refs=refs))
    files={n:sha256_file(output/n) for n in ('native-reference.pt','native-contexts.json')};maximum={}
    for mode in MODES:
        path=output/f'prompts-{mode}.jsonl';lengths=[]
        with path.open('x') as stream:
            for item in questions:
                ids=prompt_ids(item,tok,mode,interventions[item['item_id']])
                if not ids or len(ids)>256 or any(v in (151643,151645) for v in ids):
                    raise ValueError('Complete native prompt within fixed prefill required')
                stream.write(json.dumps(dict(item_id=item['item_id'],family_id=item['family_id'],mode=mode,prompt_ids=ids))+'\n');lengths.append(len(ids))
        files[path.name]=sha256_file(path);maximum[mode]=max(lengths)
    atomic_json(output/'protocol.json',dict(schema=SCHEMA,purpose='TRAIN_PROGRAM_DECODER_MECHANICS_ONLY',created_utc=utc_now(),items=4,modes=list(MODES),shards=1,prefill_width=256,max_actions=128,compute='FP32',model_dir=str(model_dir),assets=assets,signature=base['base_signature'],model=dict(id='B0',checkpoint=None),base_protocol=str(base_protocol),base_protocol_sha256=sha256_file(base_protocol),data_manifest_sha256=sha256_file(data_dir/'manifest.json'),source_files=sources(),files=files,maximum_prompt_tokens=maximum,ordered_item_ids=[q['item_id'] for q in questions],runtime_policy=dict(schema=1,environment=dict(MKL_NUM_THREADS='2',OMP_NUM_THREADS='2',PJRT_DEVICE='NEURON',NEURON_CC_FLAGS='--model-type=transformer',NEURON_NUM_RECENT_MODELS_TO_KEEP=None,XLA_COMPILATION_CACHE_SIZE='256',XLA_DOWNCAST_BF16=None,XLA_USE_BF16=None)),limits=['Four fixed TRAIN questions at depths1,2,4,6; unchanged released B0, not learned capability.','No correctness/cap-rate gate for an untrained formatting path. All28 attempts retained.','Partial controls are computed only from input. Donors are intact, same-depth and outside recipient family.','No held/model-checkpoint generation or training release from this adapter.']))


def generate(output,until):
    import torch
    from train.evaluate import GenerationFailure,asset_provenance
    from train.evaluate_local import setup_model
    from train.partial_state_answer_driver import cache_prefill_check
    from train.permutation_program_decoder import ProgramDecoder
    from train.useful_note_precision import compare_distributions
    from weekend_thinking.runtime_policy import runtime_policy,require_runtime_policy
    output=Path(output).resolve();p=engine.read(output/'protocol.json');deadline=require_utc_deadline(until)
    def check():
        if datetime.now(timezone.utc)>=deadline:raise TimeoutError('Fixed qualification deadline;no retry')
    check()
    if (p['schema']!=SCHEMA or p['purpose']!='TRAIN_PROGRAM_DECODER_MECHANICS_ONLY'
            or p['model']!=dict(id='B0',checkpoint=None) or p['items']!=4 or p['shards']!=1
            or p['prefill_width']!=256 or p['max_actions']!=128 or p['source_files']!=sources()):
        raise ValueError('Only frozen unchanged-B0 TRAIN qualification is implemented')
    for name,digest in p['files'].items():
        if sha256_file(output/name)!=digest:raise ValueError('Qualification prompt/reference changed')
    require_runtime_policy(p['runtime_policy'],runtime_policy())
    if asset_provenance(Path(p['model_dir']))!=p['assets']:raise ValueError('Actual released assets changed')
    d=output/'answers';d.mkdir(exist_ok=False);report=dict(status='running',started_utc=utc_now(),rows=0,protocol_sha256=sha256_file(output/'protocol.json'));atomic_json(d/'run.json',report);start=time.monotonic()
    try:
        backend,tok,_,signature,_=setup_model(SimpleNamespace(model_dir=Path(p['model_dir']),checkpoint=None,device='xla'))
        if signature!=p['signature']:raise ValueError('Transferred B0 signature differs')
        refs=engine.read(output/'native-contexts.json')['refs'];reference=torch.load(output/'native-reference.pt',map_location='cpu',weights_only=True)
        x=torch.full((1,1024),151643,dtype=torch.long);ids=refs[0]['context_ids'];x[0,:len(ids)]=torch.tensor(ids)
        with torch.no_grad():
            h=backend.hidden(x.to(backend.device));lp=backend.model.lm_head(h[:,len(ids)-1]).float().log_softmax(-1);backend.sync();actual=lp.cpu()
        gate=compare_distributions(reference[:1],actual)[0];torch.save(actual,d/'native-logps.pt');atomic_json(d/'native.json',gate)
        if not gate['passed']:raise RuntimeError('Actual own-model native distribution check failed')
        del h,lp,x,actual
        decoder=ProgramDecoder(backend.model,tok,sync=backend.sync,banned_ids=backend.banned_ids)
        for mode in p['modes']:
            prompts=engine.rows(output/f'prompts-{mode}.jsonl');first=max(prompts,key=lambda v:len(v['prompt_ids']))
            audit,vectors=cache_prefill_check(backend,decoder,first['prompt_ids'],width=256,cap=128)
            torch.save(vectors,d/f'cache-prefill-{mode}.pt');atomic_json(d/f'cache-prefill-{mode}.json',audit)
            if audit['max_logp_error']>1e-3:raise RuntimeError('Program cached prefill differs from ordinary causal forward')
        with (d/'attempts.jsonl').open('x') as stream:
            for mode in p['modes']:
                prompts=engine.rows(output/f'prompts-{mode}.jsonl')
                if len(prompts)!=4 or [v['item_id'] for v in prompts]!=p['ordered_item_ids']:raise ValueError('Complete original TRAIN cohort required')
                for row in prompts:
                    check();before=time.monotonic();value=dict(item_id=row['item_id'],family_id=row['family_id'],mode=mode,status='error',actions=[],content_ids=[],text='',raw_text='',stop_type='error',prompt_tokens=len(row['prompt_ids']))
                    try:value.update(asdict(decoder.decode(row['prompt_ids'],mode=mode,max_new_tokens=128,prefill_width=256)),status='ok')
                    except Exception as error:
                        if isinstance(error,GenerationFailure):value.update(asdict(error.partial))
                        value['error']=f'{type(error).__name__}: {error}'
                    value.update(n_actions=len(value['actions']),wall_seconds=time.monotonic()-before)
                    stream.write(json.dumps(value,allow_nan=False)+'\n');stream.flush();report['rows']+=1
                    print(json.dumps(dict(utc=utc_now(),mode=mode,rows=report['rows'])),flush=True)
        report.update(status='complete_all_attempts',attempts_sha256=sha256_file(d/'attempts.jsonl'))
    except BaseException as error:
        report.update(status='failed',error=f'{type(error).__name__}: {error}');raise
    finally:
        report.update(finished_utc=utc_now(),wall_seconds=time.monotonic()-start);atomic_json(d/'run.json',report,overwrite=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=['prepare','generate']);p.add_argument('--output',type=Path,required=True)
    for name in ['data-dir','model-dir','base-protocol']:p.add_argument('--'+name,type=Path)
    p.add_argument('--until');a=p.parse_args()
    if a.command=='prepare':
        if any(v is None for v in (a.data_dir,a.model_dir,a.base_protocol)):p.error('All explicit input paths required')
        prepare(a.output,a.data_dir,a.model_dir,a.base_protocol)
    else:
        if a.until is None:p.error('A finite deadline is required')
        generate(a.output,a.until)


if __name__=='__main__':main()

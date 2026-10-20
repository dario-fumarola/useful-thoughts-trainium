"""Frozen complete-answer evaluation after all six V28 terminals are audited.

Only the existing decoder and numerical backend generate. This module chooses
cohorts, constructs prefix-only inputs and binds actual terminal model weights.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

from train import document_qa_evaluate as engine
from train.permutation_program_data import execute,render,trace
from train.permutation_program_evaluation import available_item,controls,prompt_ids
from train.permutation_program_packing import WORK,FINAL
from weekend_thinking.artifacts import atomic_json,sha256_file,utc_now

ROOT=Path(__file__).resolve().parents[1]
SCHEMA='permutation-program-fixed-answers-v28'
MANIFEST_SHA='41ddc050ebbaf8abd3e0fb1af2b0c383875313bd6ff13d34b77d7a2040b5aa41'
DESIGN_SHA='d01498ec6eec98fedad28bfa5b5426059ccceff196ef7522bb44ea27d98ebba6'
PANELS=('DEV_interpolation','DEV_composition','DEV_depth')
MODEL_IDS={f'program-{arm}-seed{seed}-v1' for arm in ('direct','trace','pause') for seed in (2026102028,2026102029)}


def sources():
    names=('scripts/release_permutation_program_evaluation.py','train/permutation_program_readout.py','train/permutation_program_science_evaluate.py','train/permutation_program_decoder.py',
           'train/permutation_program_answers.py','train/permutation_program_evaluation.py',
           'train/permutation_program_data.py','train/permutation_program_packing.py')
    return {**engine.sources(),**{n:sha256_file(ROOT/n) for n in names}}


def verify_release(path):
    r=engine.read(path)
    if (r.get('status')!='RELEASED_V28_FIXED_ALL_TERMINAL_EVALUATION'
            or r.get('data_manifest_sha256')!=MANIFEST_SHA
            or r.get('evaluation_source_files')!=sources()
            or r.get('prospective_design_sha256')!=DESIGN_SHA
            or r.get('model_specs',{}).get('B0')!=dict(id='B0',checkpoint=None)
            or set(r.get('model_specs',{}))!={'B0',*MODEL_IDS}):
        raise ValueError('All six audited terminal models and a fixed evaluation release are required')
    for key in MODEL_IDS:
        spec=r['model_specs'][key]
        if (spec.get('status')!='AUDITED_V28_TERMINAL4096' or spec.get('step')!=4096
                or spec.get('id')!=key or spec.get('run_id')!=key
                or spec.get('arm')!=key.split('-')[1]):
            raise ValueError('No development/partial model can enter the fixed evaluation')
    return r


def cohort(data_dir,panel,*,base=False,diagnostic=False):
    data_dir=Path(data_dir)
    if panel not in PANELS or sha256_file(data_dir/'manifest.json')!=MANIFEST_SHA:
        raise ValueError('Only the fixed synthetic DEV panels are permitted')
    m=engine.read(data_dir/'manifest.json');path=data_dir/(panel+'-questions.jsonl')
    if sha256_file(path)!=m['files'][path.name]:raise ValueError('Questions-only cohort changed')
    rows=engine.rows(path)
    if base or panel!='DEV_composition':rows=rows[:2048]
    if diagnostic:
        if panel!='DEV_composition' or base:raise ValueError('Fixed trace-model composition diagnostics only')
        families=sorted({v['family_id'] for v in rows},key=lambda f:hashlib.sha256(('v28-partial-diagnostic:'+f).encode()).hexdigest())[:512]
        rows=[v for v in rows if v['family_id'] in set(families)]
    counts=Counter(v['family_id'] for v in rows)
    expected=1024 if diagnostic else 2048 if base or panel!='DEV_composition' else 18432
    if len(rows)!=expected or set(counts.values())!={2} or len({v['item_id'] for v in rows})!=expected:
        raise ValueError('Complete original paired cohort required')
    for row in rows:available_item(row)
    return rows


def b0_demo(data_dir):
    m=engine.read(Path(data_dir)/'manifest.json');path=Path(data_dir)/'TRAIN-questions.jsonl'
    if sha256_file(path)!=m['files'][path.name]:raise ValueError('Fixed TRAIN demonstration source changed')
    pool=[v for v in engine.rows(path) if len(available_item(v).program)==1]
    row=min(pool,key=lambda v:hashlib.sha256(('v28-B0-fixed-depth1-demo:'+v['item_id']).encode()).hexdigest())
    q=available_item(row);text=render(q)+WORK+trace(q)+FINAL+'ABCDEFGHIJKLMNOP'[execute(q)[-1]]+'\n\n'
    return row['item_id'],text


def model_modes(spec,diagnostic=False):
    if diagnostic:
        if spec.get('arm')!='trace':raise ValueError('Partial controls belong only to trace models')
        return ['partial','corrupt','donor']
    if spec['id']=='B0':return ['direct','trace']
    if spec.get('arm') not in ('direct','trace','pause'):raise ValueError('Declared reference arm required')
    return ['direct','empty'] if spec['arm']=='direct' else ['direct',spec['arm'],'empty']


def prepare(output,data_dir,model_dir,base_protocol,release_file,model_id,panel,*,diagnostic=False,native_from=None):
    import torch
    from transformers import AutoTokenizer
    from train.evaluate import asset_provenance
    from train.evaluate_local import setup_model
    from train.useful_note_sft_driver import load_native_reference
    output,data_dir,model_dir,base_protocol,release_file=[Path(v).resolve() for v in (output,data_dir,model_dir,base_protocol,release_file)]
    release=verify_release(release_file)
    if model_id not in release['model_specs']:raise ValueError('Declared model required')
    spec=release['model_specs'][model_id];base=model_id=='B0'
    if not base and (sha256_file(spec['checkpoint'])!=spec['checkpoint_sha256']
                     or sha256_file(spec['audit_file'])!=spec['audit_sha256']):
        raise ValueError('Actual audited terminal weights and audit changed')
    if not base:
        audit=engine.read(spec['audit_file'])
        if (audit.get('status')!='PASS_V28_TERMINAL_ALL_STATE' or audit.get('step')!=4096
                or audit.get('model')!=model_id or audit.get('model_sha256')!=spec['checkpoint_sha256']):
            raise ValueError('Independent actual terminal model audit required')
    if diagnostic and spec.get('arm')!='trace':raise ValueError('Diagnostic intervention belongs only to trace models')
    items=cohort(data_dir,panel,base=base,diagnostic=diagnostic)
    modes=model_modes(spec,diagnostic)
    intervention=controls(items) if diagnostic else {}
    tok=AutoTokenizer.from_pretrained(model_dir,local_files_only=True,trust_remote_code=False)
    assets=asset_provenance(model_dir);original=engine.read(base_protocol);refs,native=load_native_reference(original,assets)
    signature=original['base_signature'];origin=None
    if native_from is not None:
        native_from=Path(native_from);prior=engine.read(native_from/'protocol.json')
        if prior['model']!=spec or prior['assets']!=assets or prior['source_files']!=sources():
            raise ValueError('Own-model native reference belongs to another model/source')
        for name in ('native-contexts.json','native-reference.pt'):
            if sha256_file(native_from/name)!=prior['files'][name]:raise ValueError('Own-model native reference changed')
        native=torch.load(native_from/'native-reference.pt',weights_only=True,map_location='cpu')
        signature=prior['signature'];origin=str(native_from)
    elif not base:
        torch.set_num_threads(16)
        backend,_,loaded,signature,_=setup_model(SimpleNamespace(model_dir=model_dir,checkpoint=Path(spec['checkpoint']),device='cpu'))
        if any(loaded.get(k)!=spec[k] for k in ('step','run_id','spec_hash')):raise ValueError('Loaded model lineage differs')
        values=[]
        with torch.no_grad():
            for ref in refs:
                x=torch.full((1,1024),151643,dtype=torch.long);ids=ref['context_ids'];x[0,:len(ids)]=torch.tensor(ids)
                h=backend.hidden(x);values.append(backend.model.lm_head(h[:,len(ids)-1]).float().log_softmax(-1))
        native=torch.cat(values).cpu();del backend,h,x,values
    output.mkdir(parents=True,exist_ok=False)
    torch.save(native,output/'native-reference.pt');atomic_json(output/'native-contexts.json',dict(refs=refs))
    files={n:sha256_file(output/n) for n in ('native-reference.pt','native-contexts.json')}
    demo_id,demo=b0_demo(data_dir) if base else (None,'');maximum={}
    for mode in modes:
        path=output/f'prompts-{mode}.jsonl';lengths=[]
        with path.open('x') as stream:
            for row in items:
                ids=prompt_ids(row,tok,mode,intervention.get(row['item_id']))
                if demo:
                    # Complete native tokenization, rather than assuming a splice boundary.
                    text=tok.decode(ids,skip_special_tokens=False,clean_up_tokenization_spaces=False)
                    ids=tok.encode(demo+text,add_special_tokens=False)
                if len(ids)>256 or any(v in (151643,151645) for v in ids):raise ValueError('No truncation or prompt policy stop allowed')
                stream.write(json.dumps(dict(item_id=row['item_id'],family_id=row['family_id'],mode=mode,parser_mode=mode,prompt_ids=ids))+'\n');lengths.append(len(ids))
        files[path.name]=sha256_file(path);maximum[mode]=max(lengths)
    gold=output/'gold.jsonl'
    with gold.open('x') as stream:
        for row in items:
            q=available_item(row);stream.write(json.dumps(dict(item_id=row['item_id'],family_id=row['family_id'],question=row['question'],answer='ABCDEFGHIJKLMNOP'[execute(q)[-1]]))+'\n')
    atomic_json(output/'protocol.json',dict(schema=SCHEMA,purpose='FIXED_V28_ALL_TERMINAL_EVALUATION',created_utc=utc_now(),panel=panel,diagnostic=diagnostic,model=spec,model_dir=str(model_dir),assets=assets,signature=signature,native_origin=origin,items=len(items),modes=modes,shards=4,prefill_width=256,max_actions=128,compute='FP32',source_files=sources(),files=files,gold_file_sha256=sha256_file(gold),data_manifest_sha256=MANIFEST_SHA,maximum_prompt_tokens=maximum,ordered_item_ids=[v['item_id'] for v in items],release_file=str(release_file),release_sha256=sha256_file(release_file),B0_fixed_TRAIN_demo_id=demo_id,runtime_policy=dict(schema=1,environment=dict(MKL_NUM_THREADS='2',OMP_NUM_THREADS='2',PJRT_DEVICE='NEURON',NEURON_CC_FLAGS='--model-type=transformer',NEURON_NUM_RECENT_MODELS_TO_KEEP=None,XLA_COMPILATION_CACHE_SIZE='256',XLA_DOWNCAST_BF16=None,XLA_USE_BF16=None)),limits=['Generation opens only frozen prompt IDs/native references; gold.jsonl is deliberately outside the generation files map.','B0 is one fixed demonstration reference on original subsets, not a primary matched-training comparison.','All failed/capped/unsupported answers remain in denominators; exact states are a separate diagnostic.','Fixed terminal models only, no performance-based checkpoint or prompt selection.']))


def generate(output,shard,until):
    from train.permutation_program_decoder import ProgramDecoder
    p=engine.read(Path(output)/'protocol.json')
    if (p.get('purpose')!='FIXED_V28_ALL_TERMINAL_EVALUATION' or p.get('prefill_width')!=256
            or p.get('max_actions')!=128 or p.get('shards')!=4 or p.get('panel') not in PANELS):
        raise ValueError('Explicit all-terminal fixed evaluation and128-action contract required')
    release=verify_release(p['release_file'])
    if p['model']!=release['model_specs'].get(p['model']['id']):
        raise ValueError('Generation model differs from the all-terminal release')
    expected_items=(1024 if p['diagnostic'] else 2048 if p['model']['id']=='B0' or p['panel']!='DEV_composition' else 18432)
    if p['modes']!=model_modes(p['model'],p['diagnostic']) or p['items']!=expected_items:
        raise ValueError('Fixed complete cohort and model-specific modes required')
    return engine.generate(output,shard,until,schema=SCHEMA,expected_sources=sources(),decoder_class=ProgramDecoder,prefill_width=256)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=['prepare','generate']);p.add_argument('--output',type=Path,required=True)
    for name in ['data-dir','model-dir','base-protocol','release-file','native-from']:p.add_argument('--'+name,type=Path)
    p.add_argument('--model-id');p.add_argument('--panel',choices=PANELS);p.add_argument('--diagnostic',action='store_true');p.add_argument('--shard',type=int);p.add_argument('--until');a=p.parse_args()
    if a.command=='prepare':
        if any(v is None for v in (a.data_dir,a.model_dir,a.base_protocol,a.release_file,a.model_id,a.panel)):p.error('All frozen model/data/release inputs are required')
        prepare(a.output,a.data_dir,a.model_dir,a.base_protocol,a.release_file,a.model_id,a.panel,diagnostic=a.diagnostic,native_from=a.native_from)
    else:
        if a.shard is None or a.until is None:p.error('A fixed shard and finite deadline are required')
        generate(a.output,a.shard,a.until)


if __name__=='__main__':main()

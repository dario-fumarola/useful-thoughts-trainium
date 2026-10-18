"""Complete answers on fixed TRAIN/DEV panels with a frozen partial-state reader.

Reuse the existing released-model loader and causal decoder. Generation reads
only projected questions and prefix-computable controls. Gold scoring is a
separate command after all four shards close. A trained reader requires a closed
full terminal128 publication and its own CPU native reference. The original B0
v1 baseline remains isolated on its immutable worktree.
"""
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import time
from types import SimpleNamespace

from train.partial_state_answers import NumberLineDecoder, score_answer
from train.partial_state_evaluation import prompt_tokens, question_projection, reader_tokens, teacher_controls
from weekend_thinking.runtime_policy import require_runtime_policy, runtime_policy
from weekend_thinking.artifacts import atomic_json, canonical_hash, require_utc_deadline, sha256_file, utc_now

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = 'partial-state-complete-answers-v4'
TRAIN_SHA = '020fdf19892217492866b38f48a9a2655c30310fa792207b188477f28a2c93cb'
MODES = ('direct', 'cot', 'empty', 'teacher', 'corrupt', 'other_teacher', 'restatement')
COHORTS = {
    'TRAIN': (TRAIN_SHA, 128),
    'DEV_interpolation': ('ef5af497e8a2a45c0ceede5e3a164f446c39a5cfc49d7d4f20347094cc63f018', 512),
    'DEV_structure': ('015528413c842defa168365cea4368f0c9f0a08f712fea02021d90cbc5a90662', 512),
    'DEV_range': ('b79c962fe3b6c0c3787363e5d5debb8a3f00dfdcdd2868a7481af4a1c9a74189', 512),
    'DEV_composition': ('cbe5d61ee280e12fd5152c0637f39f72e0394456df45d217a987e5db21132983', 512),
}


def read(path):
    return json.loads(Path(path).read_text())


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def sources():
    from train.evaluation_integrity import evaluation_dependencies
    names = ('train/partial_state_answer_driver.py', 'train/partial_state_answers.py',
             'train/partial_state_evaluation.py', 'train/partial_state_data.py',
             'train/partial_state_checkpoint.py', 'train/partial_state_driver.py',
             'train/partial_state_note_driver.py',
             'train/useful_note_precision.py', 'train/runtime_compat.py',
             'weekend_thinking/runtime_policy.py')
    return {**evaluation_dependencies(), **{n: sha256_file(ROOT/n) for n in names}}


def prepare(output, data_file, model_dir, base_protocol, reader_binding=None, notes_bundle=None,
            panel='TRAIN', requested_modes=None, native_from=None):
    from transformers import AutoTokenizer
    from train.evaluate import asset_provenance
    if panel not in COHORTS:
        raise ValueError('Only declared TRAIN/DEV panels; never TEST')
    data_sha, items = COHORTS[panel]
    if sha256_file(data_file) != data_sha:
        raise ValueError('Exact fixed panel bytes required')
    data = rows(data_file)[:items]
    if len(data) != items or any(r['panel'] != panel for r in data):
        raise ValueError('Complete declared panel required')
    questions = [question_projection(r) for r in data]
    tok = AutoTokenizer.from_pretrained(model_dir, local_files_only=True, trust_remote_code=False)
    assets = asset_provenance(model_dir); base = read(base_protocol)
    if assets['model_id'] != 'Qwen/Qwen3-1.7B-Base' or assets['manifest_sha256'] != base['model_manifest_sha256']:
        raise ValueError('Actual released B0 assets and canonical reference required')
    controls = teacher_controls(questions, tok)
    modes = list(MODES)
    bundle = None
    if notes_bundle is not None:
        from train.partial_state_note_driver import load_bundle
        bundle = load_bundle(notes_bundle, [q['item_id'] for q in questions])
        if [c['item_id'] for c in bundle['controls']] != [q['item_id'] for q in questions]:
            raise ValueError('Same ordered writer/reader question cohort required')
        for c, generated in zip(controls, bundle['controls']):
            c['note_generation'] = {}
            for mode in ('own', 'other'):
                note = generated[mode]
                c['notes'][mode] = note['content_ids']
                c['note_generation'][mode] = {k:note[k] for k in
                    ('item_id','status','stop_type','n_actions','wall_seconds')}
        modes += ['own', 'other']
    if requested_modes is not None:
        if not requested_modes or len(set(requested_modes))!=len(requested_modes) or any(m not in modes for m in requested_modes):
            raise ValueError('Unique declared conditions only; own/other require original note bundle')
        modes = list(requested_modes)
    reader = dict(id='B0', checkpoint=None)
    if reader_binding is not None:
        from train.partial_state_checkpoint import verify
        reader = verify(read(reader_binding))
    if max(len(prompt_tokens(q, tok, cot=True)) for q in questions) > 32:
        raise ValueError('Declared prefill width is too small')
    if any(len(reader_tokens(q, c['notes'][mode], tok)) > 128 for q,c in zip(questions,controls) for mode in modes if mode not in ('direct','cot')):
        raise ValueError('Declared reader prefill width is too small')
    output.mkdir(parents=True, exist_ok=False)
    from train.useful_note_sft_driver import load_native_reference
    refs, native = load_native_reference(base, assets)
    signature = base['base_signature']
    native_origin = None
    if native_from is not None:
        import torch
        native_from=Path(native_from);prior=read(native_from/'protocol.json');meta=read(native_from/'native-contexts.json')
        if (prior['reader']!=reader or prior['assets']!=assets or meta['reader']!=reader
                or meta['model_signature']!=prior['evaluation_signature']):
            raise ValueError('Native reference belongs to a different frozen model')
        for name in ('native-reference.pt','native-contexts.json'):
            if sha256_file(native_from/name)!=prior['files'][name]:
                raise ValueError('Original own-model native reference changed')
        native=torch.load(native_from/'native-reference.pt',map_location='cpu',weights_only=True)
        if native.shape!=(4,151936) or native.dtype!=torch.float32 or not torch.isfinite(native).all():
            raise ValueError('Four full-head own-model native vectors required')
        refs,signature=meta['refs'],meta['model_signature']
        native_origin={str(native_from/n):sha256_file(native_from/n) for n in ('protocol.json','native-reference.pt','native-contexts.json')}
    elif reader['checkpoint'] is not None:
        import torch
        from train.evaluate_local import setup_model
        backend, _, loaded, signature, _ = setup_model(SimpleNamespace(
            model_dir=model_dir, checkpoint=Path(reader['checkpoint']), device='cpu'))
        if any(loaded.get(k) != reader[k] for k in ('step','run_id','spec_hash')):
            raise ValueError('Actual CPU reference model differs from frozen reader')
        values = []
        with torch.no_grad():
            for ref in refs:
                x=torch.full((1,1024),151643,dtype=torch.long)
                x[0,:len(ref['context_ids'])]=torch.tensor(ref['context_ids'])
                hidden=backend.hidden(x)
                values.append(backend.model.lm_head(hidden[:,len(ref['context_ids'])-1]).float().log_softmax(-1))
        native=torch.cat(values).cpu()
        del backend,hidden,values,x
    import torch
    torch.save(native,output/'native-reference.pt')
    atomic_json(output/'native-contexts.json',dict(refs=refs,width=1024,model_signature=signature,reader=reader))
    for name, values in [('questions', questions), ('controls', controls),
                         ('gold', [dict(item_id=r['item_id'], answer=r['answer']) for r in data])]:
        with (output/f'{name}.jsonl').open('x') as stream:
            for value in values:
                stream.write(json.dumps(value, separators=(',', ':'))+'\n')
    protocol = dict(schema=SCHEMA, created_utc=utc_now(), panel=panel,
        purpose='TRAIN fitting' if panel=='TRAIN' else 'Prospectively fixed DEV computation diagnostic, not official TEST',
        model_dir=str(model_dir.resolve()), assets=assets, base_signature=base['base_signature'],
        base_protocol=str(base_protocol.resolve()), base_protocol_sha256=sha256_file(base_protocol),
        data_sha256=data_sha, source_files=sources(), items=items, shards=4, modes=modes,
        files={name: sha256_file(output/name) for name in ('questions.jsonl','controls.jsonl','gold.jsonl','native-contexts.json','native-reference.pt')},
        ordered_item_ids=[r['item_id'] for r in questions], model_identity=reader['id'],
        reader=reader, evaluation_signature=signature,
        native_origin=native_origin,
        writer=None if bundle is None else bundle['writer'],
        writer_execution_device=None if bundle is None else bundle['execution_device'],
        notes_bundle=None if bundle is None else str(Path(notes_bundle).resolve()),
        notes_bundle_sha256=None if bundle is None else sha256_file(notes_bundle),
        compute='fp32', selection='single_greedy_tokenizer_defined_support',
        writer_or_cot_prefill=32, reader_prefill=128, direct_and_cot_cap=80, reader_answer_cap=16,
        future_split='Generation never opens gold.jsonl or original supervised TRAIN rows. Supplied notes are computed only from prefix expressions.',
        endpoint='Complete numeric-answer accuracy; all unsupported/errors/caps retained. Own-minus-same-reader empty/direct/other, and ordinary compact-CoT at same80 total action cap.',
        controls='Intact global derangement, native-length matched when peers exist; no repeat, trim, repair or utility selection.',
        other_teacher_unmatched_lengths=sum(not c['other_teacher_length_matched'] for c in controls),
        runtime_policy=runtime_policy(), cache_prefill_max_logp_error=1e-3,
        limits=['TRAIN is fitting only. Separate512-item DEV panels test interpolation, unseen structures, range and composition; never pool them or call them official TEST.',
                'Trained readers, if present, are explicitly TRAIN128 overfit diagnostics, not scientific warm starts.',
                'Direct prompt asks for the answer; compact CoT starts at Work. Equal80 action caps do not equal measured FLOPs.',
                'Supplied-state reading is achievable-reader evidence, not student utility.',
                'Reader conditions use16 answer actions; direct/CoT80. Report actual actions and latency separately.'])
    atomic_json(output/'protocol.json', protocol)
    return protocol


def load(output):
    p = read(output/'protocol.json')
    if p['schema'] != SCHEMA or p['source_files'] != sources() or p['model_identity'] != p['reader']['id']:
        raise ValueError('Frozen baseline source/identity changed')
    # Intentionally do not open or hash gold here: generation uses no gold.
    for name in ('questions.jsonl', 'controls.jsonl', 'native-contexts.json', 'native-reference.pt'):
        if sha256_file(output/name) != p['files'][name]:
            raise ValueError('Frozen question/control bytes changed')
    if p['notes_bundle'] is not None:
        from train.partial_state_note_driver import load_bundle
        if sha256_file(p['notes_bundle']) != p['notes_bundle_sha256']:
            raise ValueError('Frozen writer bundle changed')
        load_bundle(p['notes_bundle'], p['ordered_item_ids'])
    qs, cs = rows(output/'questions.jsonl'), rows(output/'controls.jsonl')
    if (p.get('panel') not in COHORTS or (p['data_sha256'],p['items'])!=COHORTS[p['panel']]
            or len(qs)!=p['items'] or any(q['panel']!=p['panel'] for q in qs)
            or len(set(p['ordered_item_ids']))!=p['items']
            or [q['item_id'] for q in qs] != p['ordered_item_ids'] or [c['item_id'] for c in cs] != p['ordered_item_ids']):
        raise ValueError('Complete ordered cohort required')
    return p, qs, cs


def cache_prefill_check(backend, decoder, prompt, *, width, cap):
    """Compare this actual cached shape against the ordinary causal forward."""
    import torch
    from train.neuron_backend import CausalCache
    from train.useful_note_precision import compare_distributions
    with torch.no_grad():
        x=torch.full((1,width),151643,dtype=torch.long);x[0,:len(prompt)]=torch.tensor(prompt);x=x.to(backend.device)
        cache=CausalCache(backend.model.config.num_hidden_layers,width+cap)
        selected=torch.tensor([len(prompt)-1],device=backend.device)
        cached=decoder._next(x,torch.arange(width,device=backend.device),cache,selected).float().log_softmax(-1)
        # Materialize each forward separately. A combined lazy graph duplicates
        # the full trunk and makes this validation much more costly to compile.
        backend.sync();b=cached.cpu()
        hidden=backend.hidden(x)
        index=selected.reshape(1,1,1).expand(1,1,hidden.shape[-1])
        ordinary=backend.model.lm_head(hidden.gather(1,index)[:,0]).float().log_softmax(-1)
        backend.sync();a=ordinary.cpu()
    return compare_distributions(a,b)[0], dict(ordinary=a,cached=b)


def generate(output, shard, until):
    import torch
    from train.evaluate import GenerationFailure, asset_provenance
    from train.evaluate_local import setup_model
    from train.useful_note_precision import compare_distributions
    from weekend_thinking.pretrained import sampled_fingerprint
    deadline = require_utc_deadline(until)
    def check():
        if datetime.now(timezone.utc) >= deadline:
            raise TimeoutError('Finite baseline deadline; no automatic retry')
    p, questions, controls = load(output); check()
    require_runtime_policy(p['runtime_policy'],runtime_policy())
    if not 0 <= shard < 4 or sha256_file(p['base_protocol']) != p['base_protocol_sha256']:
        raise ValueError('Explicit four-shard native binding required')
    if asset_provenance(Path(p['model_dir'])) != p['assets']:
        raise ValueError('Released asset bytes changed')
    d = output/'answers'/f'shard-{shard}'; d.mkdir(parents=True, exist_ok=False)
    report = dict(status='running', started_utc=utc_now(), shard=shard, rows=0,
                  protocol_sha256=sha256_file(output/'protocol.json'), model_identity=p['model_identity'])
    atomic_json(d/'run.json',report); started=time.monotonic()
    try:
        reader=p['reader'];path=reader['checkpoint']
        if path is not None:
            from train.partial_state_checkpoint import verify
            verify(reader)
        backend, tok, checkpoint, signature, _ = setup_model(SimpleNamespace(model_dir=Path(p['model_dir']), checkpoint=Path(path) if path else None, device='xla'))
        if signature != p['evaluation_signature'] or sampled_fingerprint(backend.model) != signature:
            raise ValueError('Actual transferred model differs from its CPU reference')
        if path is None:
            if checkpoint is not None or signature != p['base_signature']:
                raise ValueError('Actual unchanged B0 required')
        elif any(checkpoint.get(k)!=reader[k] for k in ('step','run_id','spec_hash')):
            raise ValueError('Actual loaded terminal reader identity differs')
        refs=read(output/'native-contexts.json')['refs'];native=torch.load(output/'native-reference.pt',map_location='cpu',weights_only=True)
        context=refs[shard]['context_ids']; x=torch.full((1,1024),151643,dtype=torch.long);x[0,:len(context)]=torch.tensor(context)
        with torch.no_grad():
            hidden=backend.hidden(x.to(backend.device)); lp=backend.model.lm_head(hidden[:,len(context)-1]).float().log_softmax(-1);backend.sync(); actual=lp.cpu()
        native_check=compare_distributions(native[shard:shard+1],actual)[0]
        torch.save(actual,d/'native-logps.pt');atomic_json(d/'native.json',native_check)
        if not native_check['passed']:
            raise RuntimeError('Actual native canonical reference failed')
        del hidden,lp,x,actual
        decoder=NumberLineDecoder(backend.model,tok,sync=backend.sync,banned_ids=backend.banned_ids)
        q,c=questions[shard],controls[shard]
        for mode in ('direct','cot','teacher'):
            prompt=prompt_tokens(q,tok,cot=mode=='cot') if mode!='teacher' else reader_tokens(q,c['notes']['teacher'],tok)
            width=p['writer_or_cot_prefill'] if mode!='teacher' else p['reader_prefill']
            cap=p['direct_and_cot_cap'] if mode!='teacher' else p['reader_answer_cap']
            audit,vectors=cache_prefill_check(backend,decoder,prompt,width=width,cap=cap)
            torch.save(vectors,d/f'cache-{mode}-logps.pt');atomic_json(d/f'cache-{mode}.json',audit)
            if audit['max_logp_error']>p['cache_prefill_max_logp_error']:
                raise RuntimeError('Actual new cached prefill disagrees with ordinary forward')
        with (d/'answers.jsonl').open('x') as stream:
            for mode in p['modes']:
                for index in range(shard,len(questions),4):
                    q,c=questions[index],controls[index]
                    check(); before=time.monotonic()
                    prompt = (prompt_tokens(q,tok,cot=mode=='cot') if mode in ('direct','cot') else reader_tokens(q,c['notes'][mode],tok))
                    cap = p['direct_and_cot_cap'] if mode in ('direct','cot') else p['reader_answer_cap']
                    width = p['writer_or_cot_prefill'] if mode in ('direct','cot') else p['reader_prefill']
                    source_note=c.get('note_generation',{}).get(mode)
                    value=dict(item_id=q['item_id'],panel=q['panel'],mode=mode,status='error',actions=[],content_ids=[],text='',raw_text='',stop_type='error',prompt_ids=prompt,cap=cap,prefill_width=width,
                               source_note=source_note,n_writer_actions=0 if source_note is None else source_note['n_actions'])
                    try:
                        if source_note is not None and source_note['status']!='ok':
                            raise RuntimeError('Original writer/donor attempt failed; do not replace it with empty')
                        decoded=decoder.decode_number(prompt,max_new_tokens=cap,prefill_width=width,cot=mode=='cot')
                        value.update(asdict(decoded),status='ok')
                    except Exception as error:
                        if isinstance(error,GenerationFailure):
                            value.update(asdict(error.partial))
                        value['error']=f'{type(error).__name__}: {error}'
                    value.update(wall_seconds=time.monotonic()-before,n_actions=len(value['actions']))
                    stream.write(json.dumps(value,allow_nan=False)+'\n');stream.flush();report['rows']+=1
                    print(json.dumps(dict(shard=shard,mode=mode,items=(index-shard)//4+1,rows=report['rows'],utc=utc_now())),flush=True)
        if report['rows'] != len(range(shard,p['items'],4))*len(p['modes']):
            raise RuntimeError('All attempts must be retained')
        report.update(status='complete_all_attempts',answers_sha256=sha256_file(d/'answers.jsonl'))
    except BaseException as error:
        report.update(status='failed',error=f'{type(error).__name__}: {error}');raise
    finally:
        report.update(finished_utc=utc_now(),stage_wall_seconds=time.monotonic()-started)
        atomic_json(d/'run.json',report,overwrite=True)


def score(output):
    from collections import Counter
    p, questions, _ = load(output); allrows=[]
    for shard in range(4):
        d=output/'answers'/f'shard-{shard}';r=read(d/'run.json')
        if r['status']!='complete_all_attempts' or sha256_file(d/'answers.jsonl')!=r['answers_sha256']:
            raise ValueError('All original generation shards must close before gold scoring')
        allrows.extend(rows(d/'answers.jsonl'))
    if sha256_file(output/'gold.jsonl')!=p['files']['gold.jsonl']:
        raise ValueError('Offline gold changed')
    gold={r['item_id']:r['answer'] for r in rows(output/'gold.jsonl')}
    cells={}
    for value in allrows:
        key=(value['item_id'],value['mode'])
        if key in cells or value['item_id'] not in gold or value['mode'] not in p['modes']:
            raise ValueError('Unique complete declared cells required')
        cells[key]={**value,**score_answer(value,gold[value['item_id']],cot=value['mode']=='cot')}
    n=p['items']
    if len(cells)!=n*len(p['modes']):
        raise ValueError('Keep every answer attempt in its denominator')
    result={}
    for mode in p['modes']:
        values=[cells[q['item_id'],mode] for q in questions]
        result[mode]=dict(items=n,correct=sum(v['score'] for v in values),status_counts=dict(Counter(v['status'] for v in values)),unsupported=sum(v['status']!='ok' for v in values),generated_actions=sum(v['n_actions'] for v in values),writer_actions=sum(v['n_writer_actions'] for v in values),decode_wall_seconds=sum(v['wall_seconds'] for v in values),writer_wall_seconds=sum(0 if v['source_note'] is None else v['source_note']['wall_seconds'] for v in values))
    contrasts={}
    pairs=[('teacher','empty'),('teacher','direct'),('teacher','corrupt'),('teacher','other_teacher'),('cot','direct')]
    if 'own' in p['modes']:
        pairs += [('own','empty'),('own','direct'),('own','other'),('teacher','own')]
    for b,a in pairs:
        if b not in p['modes'] or a not in p['modes']:
            continue
        # Paired item bootstrap is computed below, rather than resampling tokens.
        import numpy as np
        delta=np.array([cells[q['item_id'],b]['score']-cells[q['item_id'],a]['score'] for q in questions],dtype=float)
        picks=np.random.default_rng(17).integers(0,n,size=(5000,n));ci=np.quantile(delta[picks].mean(1),[.025,.975])
        contrasts[b+'_minus_'+a]=dict(difference=float(delta.mean()),ci95=ci.tolist(),unit='accuracy_fraction',bootstrap_items=n,repeats=5000,seed=17)
    atomic_json(output/'summary.json',dict(schema=SCHEMA,status='complete_all_attempts',finished_utc=utc_now(),
        protocol_sha256=sha256_file(output/'protocol.json'),panel=p['panel'],reader=p['reader'],reader_execution_device='xla',
        writer=p['writer'],writer_execution_device=p['writer_execution_device'],
        results=result,contrasts=contrasts,limits=p['limits']))
    return result


def main():
    a=argparse.ArgumentParser(description=__doc__);a.add_argument('command',choices=('prepare','generate','score'))
    a.add_argument('--output',type=Path,required=True);a.add_argument('--data-file',type=Path);a.add_argument('--model-dir',type=Path);a.add_argument('--base-protocol',type=Path)
    a.add_argument('--shard',type=int);a.add_argument('--deadline-utc');a.add_argument('--reader-binding',type=Path);a.add_argument('--notes-bundle',type=Path)
    a.add_argument('--panel',choices=tuple(COHORTS),default='TRAIN');a.add_argument('--modes',nargs='+');a.add_argument('--native-from',type=Path)
    v=a.parse_args()
    if v.command=='prepare':
        result=prepare(v.output,v.data_file,v.model_dir,v.base_protocol,v.reader_binding,v.notes_bundle,v.panel,v.modes,v.native_from)
    elif v.command=='generate':
        result=generate(v.output,v.shard,v.deadline_utc)
    else:
        result=score(v.output)
    print(json.dumps(result,allow_nan=False))


if __name__=='__main__':
    main()

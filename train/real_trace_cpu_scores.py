"""FP32 full-span likelihood with a canonical CPU path and qualified XLA option.

This scoring command may open continuation targets. Prefix-only generation is
a separate job and receives only the projected questions. Every declared document
must finish; an incomplete panel cannot pass a scientific benefit gate.
"""
import argparse
from pathlib import Path
from types import SimpleNamespace
import json,time,os,gc,re

from train.real_trace_evaluation import span_logprobs
from train.real_trace_contexts import reader_context
from train.real_trace_scope import FRESH_PANEL,CONFIRM_PANEL,binding_statuses,scoring_items
from weekend_thinking.artifacts import atomic_json,sha256_file,utc_now


def read(path):return json.loads(Path(path).read_text())
def rows(path):return [json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()]


def span_parity(reference, actual):
    """Actual fixed-reference token and span checks, including finite values."""
    import torch
    if reference.shape != actual.shape or reference.ndim != 2 or reference.shape[1] != 16:
        raise ValueError('Matching full16 reference rows required')
    delta = actual.double()-reference.double()
    finite = bool(torch.isfinite(delta).all())
    maximum = float(delta.abs().max()) if finite else None
    drift = float(delta.sum(1).mean().abs()) if finite else None
    return dict(passed=finite and maximum<=.001 and drift<=.001,
                rows=reference.shape[0],max_token_error=maximum,absolute_mean_span_drift=drift,
                declared_max_token_error=.001,declared_mean_span_drift=.001)


def score(output,inputs,panel,model_dir,binding=None,notes=None,threads=16,device='cpu'):
    import torch
    from train.evaluate_local import setup_model
    from train.evaluate import asset_provenance
    manifest=read(inputs/'manifest.json')
    if (manifest['status']!='FIXED_PREFIX_ONLY_QUESTIONS_AND_SEPARATE_SCORING_TARGETS'
            or panel not in manifest['panels'] or threads not in (8,16)
            or device not in ('cpu','xla')):
        raise ValueError('Explicit fixed real-document panel and bounded CPU threads required')
    for name,value in manifest['files'].items():
        if sha256_file(inputs/name)!=value['sha256']:raise ValueError('Fixed evaluation bytes changed')
    qs,ys=rows(inputs/f'{panel}-questions.jsonl'),rows(inputs/f'{panel}-targets.jsonl')
    ids=[q['item_id'] for q in qs]
    items = scoring_items(panel)
    if len(ids)!=items or len(set(ids))!=items or [v['item_id'] for v in ys]!=ids:
        raise ValueError('All declared unique ordered documents and true targets required')
    if device == 'xla' and (panel not in (FRESH_PANEL,CONFIRM_PANEL,'DEV_ordinary256')
            or os.environ.get('NEURONCORE_NUM_DEVICES') != '1'
            or not re.fullmatch('[0-9]+',os.environ.get('NEURON_RT_VISIBLE_CORES',''))):
        raise ValueError('New study scoring requires one explicitly isolated logical core')
    from train.evaluation_integrity import evaluation_dependencies
    root=Path(__file__).resolve().parents[1]
    source_files={**evaluation_dependencies(root),**{name:sha256_file(root/name) for name in ('train/real_trace_cpu_scores.py','train/real_trace_evaluation.py','train/real_trace_contexts.py','train/real_trace_packing.py','train/partial_state_evaluation.py','train/real_trace_scope.py')}}
    model=read(binding) if binding else dict(id='B0',checkpoint=None)
    if binding:
        if model['status'] not in binding_statuses(panel) or model['step']!=320:
            raise ValueError('Explicit independently audited terminal320 required')
        if sha256_file(model['checkpoint'])!=model['checkpoint_sha256']:
            raise ValueError('Actual terminal model changed')
        if sha256_file(model['audit_file'])!=model['audit_sha256']:
            raise ValueError('Independent terminal audit changed')
    controls=rows(inputs/f'{panel}-controls.jsonl')
    arithmetic=manifest['panels'][panel]['arithmetic'];modes=['raw']
    if arithmetic:
        if [c['item_id'] for c in controls]!=ids:raise ValueError('All prefix-computed supplied controls required')
        modes+=['empty','oracle']
    bundle=None
    if notes:
        bundle=read(notes)
        if (not arithmetic or bundle['status']!='COMPLETE_REAL_PREFIX_ONLY_NOTE_BUNDLE'
                or bundle['ordered_item_ids']!=ids or bundle['model_id']!=model['id']):
            raise ValueError('All original same-model same-panel native note attempts required')
        for name,digest in bundle['source_pins'].items():
            if sha256_file(name)!=digest:raise ValueError('Original generation evidence changed')
        if len(bundle['controls'])!=items or [c['item_id'] for c in bundle['controls']]!=ids:
            raise ValueError('Complete ordered own/other note inventory required')
        from train.partial_state_evaluation import donor_permutation
        originals={v['item_id']:v for name in bundle['source_pins'] if Path(name).name=='notes.jsonl' for v in rows(name)}
        if set(originals)!=set(ids):raise ValueError('Every intact original writer attempt required')
        donors=donor_permutation([len(originals[k]['content_ids']) for k in ids],ids)
        for k,j,c in zip(ids,donors,bundle['controls']):
            if c['own']!=originals[k] or c['other']!=originals[ids[j]] or c['other_item_id']!=ids[j]:
                raise ValueError('Reader note differs from original attempt or declared intact donor')
        producer=[read(name) for name in bundle['source_pins'] if Path(name).name=='protocol.json']
        if len(producer)!=1 or producer[0]['model']!=model:
            raise ValueError('Actual note generator differs from scoring model')
        modes+=['own','other']
    output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(threads)
    started=time.monotonic()
    def context_for(mode,i,tokenizer):
        state,note='ok',None
        if mode=='empty':note=[]
        elif mode=='oracle':note=controls[i]['oracle_ids']
        elif mode in ('own','other'):
            value=bundle['controls'][i][mode];note=value['content_ids'];state=value['status']
        return reader_context(qs[i],tokenizer,note),state
    references=[];reference_contexts=[];reference_targets=[];reference_wall=None
    if device == 'xla':
        before=time.monotonic()
        cpu,cpu_tok,loaded,cpu_signature,_=setup_model(SimpleNamespace(
            model_dir=model_dir,checkpoint=Path(model['checkpoint']) if model['checkpoint'] else None,device='cpu'))
        if cpu.device.type!='cpu' or any(v.dtype!=torch.float32 for v in cpu.model.parameters()):
            raise ValueError('Native span reference must use actual FP32 CPU parameters')
        if binding and (loaded['sha256']!=model['checkpoint_sha256']
                or any(loaded[k]!=model[k] for k in ('run_id','step','spec_hash'))):
            raise ValueError('Actual CPU reference checkpoint changed')
        for mode in modes:
            contexts=[context_for(mode,i,cpu_tok)[0] for i in range(4)]
            targets=[ys[i]['target16'] for i in range(4)]
            references.append(span_logprobs(cpu,contexts,targets,width=672))
            reference_contexts.extend(contexts);reference_targets.extend(targets)
        reference=torch.cat(references).cpu()
        torch.save(reference,output/'CPU-fixed-first4-each-mode.pt')
        reference_wall=time.monotonic()-before
        del cpu,cpu_tok,references
        gc.collect()
    backend,tok,loaded,signature,_=setup_model(SimpleNamespace(
        model_dir=model_dir,checkpoint=Path(model['checkpoint']) if model['checkpoint'] else None,device=device))
    assets=asset_provenance(model_dir)
    if assets['model_id']!='Qwen/Qwen3-1.7B-Base' or any(p.dtype!=torch.float32 for p in backend.model.parameters()):
        raise ValueError('Require actual released1.7B architecture and FP32 parameters')
    if backend.device.type!=device:raise ValueError('Actual scoring backend differs from declaration')
    if device=='xla' and signature!=cpu_signature:
        raise ValueError('CPU and native model signatures differ')
    if binding and (loaded['sha256']!=model['checkpoint_sha256']
            or any(loaded[k]!=model[k] for k in ('run_id','step','spec_hash'))):
        raise ValueError('Loaded model differs from the audited terminal binding')
    result=dict(status='running',started_utc=utc_now(),panel=panel,model_id=model['id'],
      manifest_sha256=sha256_file(inputs/'manifest.json'),model_signature=signature,
      compute='CPU_FP32' if device=='cpu' else 'XLA_FP32',threads=threads,assets=assets,rows=0,
      notes_bundle_sha256=sha256_file(notes) if notes else None,
      source_files=source_files)
    atomic_json(output/'run.json',result)
    try:
        if device == 'xla':
            actual=[]
            for i in range(0,len(reference_contexts),4):
                actual.append(span_logprobs(backend,reference_contexts[i:i+4],reference_targets[i:i+4],width=672))
            actual=torch.cat(actual).cpu()
            torch.save(actual,output/'XLA-fixed-first4-each-mode.pt')
            parity=span_parity(reference,actual)
            atomic_json(output/'span-parity.json',parity)
            result['fixed_first4_per_mode_parity']=parity
            result['span_reference_files']={name:sha256_file(output/name) for name in
                ('CPU-fixed-first4-each-mode.pt','XLA-fixed-first4-each-mode.pt','span-parity.json')}
            result['span_reference_order']=[dict(item_id=ids[i],mode=mode) for mode in modes for i in range(4)]
            result['CPU_reference_including_load_seconds']=reference_wall
            result['reference_limits']='First4 frozen documents per mode;not all-panel CPU parity or independent capability samples. B0 all1280 prior profile remains separate.'
            if not parity['passed']:
                raise RuntimeError('Predeclared full16 CPU/native parity failed;preserve and use separately identified canonical CPU scoring')
            del actual,reference
        with (output/'logps.jsonl').open('x') as stream:
            for mode in modes:
                # Length-grouped widths affect padding only; true native inputs
                # and their sixteen causal target positions remain unchanged.
                ordered=sorted(range(items),key=lambda i:len(qs[i]['prefix_ids']))
                for start in range(0,items,4):
                    batch=ordered[start:start+4];contexts=[];states=[]
                    for i in batch:
                        context,state=context_for(mode,i,tok)
                        contexts.append(context);states.append(state)
                    width=672 if device=='xla' else ((max(map(len,contexts))+16+31)//32)*32
                    if width>672:raise ValueError('Native context exceeds declared width;never truncate')
                    lp=span_logprobs(backend,contexts,[ys[i]['target16'] for i in batch],width=width)
                    for i,values,state in zip(batch,lp.tolist(),states):
                        value=dict(item_id=ids[i],document_id=qs[i]['document_id'],mode=mode,
                           logps16=values,span16_logprob=sum(values),note_execution_status=state,
                           component_status=ys[i].get('component_status','not_applicable'))
                        if ys[i].get('component_status')=='exact_native_partition':
                            value['component_logprobs']={k:sum(values[a:b]) for k,(a,b) in ys[i]['component_ranges'].items()}
                        stream.write(json.dumps(value,allow_nan=False)+'\n');result['rows']+=1
                    stream.flush()
                print(json.dumps(dict(utc=utc_now(),mode=mode,rows=result['rows'])),flush=True)
        if any(sha256_file(root/name)!=digest for name,digest in source_files.items()):
            raise ValueError('Frozen scoring implementation changed while live')
        observed=rows(output/'logps.jsonl')
        assert len(observed)==items*len(modes)
        result.update(status='COMPLETE_ALL_FIXED_DOCUMENT_LIKELIHOODS',modes=modes,documents=items,
          logps_sha256=sha256_file(output/'logps.jsonl'),
          raw_likelihood_complete=True,
          mode_execution_eligible={m:all(v['note_execution_status']=='ok' for v in observed if v['mode']==m) for m in modes},
          inference_eligible_no_generation_errors=(all(v['note_execution_status']=='ok' for v in observed) if bundle is not None else None),
          limits=['Scoringtimeisnotacontrolledend-to-endservicelatency.','Allcapsandemptyvalidnotesremain;generationerrorsareflaggedandmakebenefitgateineligible.','Oracleincludesrootanswer;itisnotdeployedprefix-onlygeneration.'])
    except BaseException as error:
        result.update(status='FAILED_INCOMPLETE_PANEL_NO_SCIENTIFIC_PASS',error=f'{type(error).__name__}: {error}')
        raise
    finally:
        result.update(finished_utc=utc_now(),wall_seconds=time.monotonic()-started)
        atomic_json(output/'run.json',result,overwrite=True)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('output','inputs','model-dir'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--panel',required=True);p.add_argument('--binding',type=Path);p.add_argument('--notes',type=Path)
    p.add_argument('--threads',type=int,choices=(8,16),default=16)
    p.add_argument('--device',choices=('cpu','xla'),default='cpu');a=p.parse_args()
    print(json.dumps(score(a.output,a.inputs,a.panel,a.model_dir,a.binding,a.notes,a.threads,a.device),allow_nan=False))


if __name__=='__main__':main()

"""CPU FP32 full-span likelihood on fixed real-document evaluation inputs.

This scoring command may open continuation targets. Prefix-only generation is
a separate job and receives only the projected questions. All 256 documents
must finish; an incomplete panel cannot pass a scientific benefit gate.
"""
import argparse
from pathlib import Path
from types import SimpleNamespace
import json,time

from train.real_trace_evaluation import span_logprobs
from train.real_trace_contexts import reader_context
from weekend_thinking.artifacts import atomic_json,sha256_file,utc_now


def read(path):return json.loads(Path(path).read_text())
def rows(path):return [json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()]


def score(output,inputs,panel,model_dir,binding=None,notes=None,threads=16):
    import torch
    from train.evaluate_local import setup_model
    from train.evaluate import asset_provenance
    manifest=read(inputs/'manifest.json')
    if (manifest['status']!='FIXED_PREFIX_ONLY_QUESTIONS_AND_SEPARATE_SCORING_TARGETS'
            or panel not in manifest['panels'] or threads not in (8,16)):
        raise ValueError('Explicit fixed real-document panel and bounded CPU threads required')
    for name,value in manifest['files'].items():
        if sha256_file(inputs/name)!=value['sha256']:raise ValueError('Fixed evaluation bytes changed')
    qs,ys=rows(inputs/f'{panel}-questions.jsonl'),rows(inputs/f'{panel}-targets.jsonl')
    ids=[q['item_id'] for q in qs]
    if len(ids)!=256 or len(set(ids))!=256 or [v['item_id'] for v in ys]!=ids:
        raise ValueError('All256unique ordered documents and true targets required')
    from train.evaluation_integrity import evaluation_dependencies
    root=Path(__file__).resolve().parents[1]
    source_files={**evaluation_dependencies(root),**{name:sha256_file(root/name) for name in ('train/real_trace_cpu_scores.py','train/real_trace_evaluation.py','train/real_trace_contexts.py','train/real_trace_packing.py')}}
    model=read(binding) if binding else dict(id='B0',checkpoint=None)
    if binding:
        if model['status']!='FROZEN_AUDITED_REAL_TRACE_TERMINAL320' or model['step']!=320:
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
        if len(bundle['controls'])!=256 or [c['item_id'] for c in bundle['controls']]!=ids:
            raise ValueError('Complete ordered own/other note inventory required')
        modes+=['own','other']
    output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(threads)
    started=time.monotonic()
    backend,tok,loaded,signature,_=setup_model(SimpleNamespace(
        model_dir=model_dir,checkpoint=Path(model['checkpoint']) if model['checkpoint'] else None,device='cpu'))
    assets=asset_provenance(model_dir)
    if assets['model_id']!='Qwen/Qwen3-1.7B-Base' or any(p.dtype!=torch.float32 for p in backend.model.parameters()):
        raise ValueError('Require actual released1.7B architecture and FP32 CPU parameters')
    if backend.device.type!='cpu':raise ValueError('This declared reference is CPU FP32')
    if binding and (loaded['sha256']!=model['checkpoint_sha256']
            or any(loaded[k]!=model[k] for k in ('run_id','step','spec_hash'))):
        raise ValueError('Loaded model differs from the audited terminal binding')
    result=dict(status='running',started_utc=utc_now(),panel=panel,model_id=model['id'],
      manifest_sha256=sha256_file(inputs/'manifest.json'),model_signature=signature,
      compute='CPU_FP32',threads=threads,assets=assets,rows=0,
      notes_bundle_sha256=sha256_file(notes) if notes else None,
      source_files=source_files)
    atomic_json(output/'run.json',result)
    try:
        with (output/'logps.jsonl').open('x') as stream:
            for mode in modes:
                # Length-grouped widths affect padding only; true native inputs
                # and their sixteen causal target positions remain unchanged.
                ordered=sorted(range(256),key=lambda i:len(qs[i]['prefix_ids']))
                for start in range(0,256,4):
                    batch=ordered[start:start+4];contexts=[];states=[]
                    for i in batch:
                        state='ok';note=None
                        if mode=='empty':note=[]
                        elif mode=='oracle':note=controls[i]['oracle_ids']
                        elif mode in ('own','other'):
                            actual=bundle['controls'][i][mode];note=actual['content_ids'];state=actual['status']
                        contexts.append(reader_context(qs[i],tok,note));states.append(state)
                    width=((max(map(len,contexts))+16+31)//32)*32
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
        assert len(observed)==256*len(modes)
        result.update(status='COMPLETE_ALL_FIXED_DOCUMENT_LIKELIHOODS',modes=modes,documents=256,
          logps_sha256=sha256_file(output/'logps.jsonl'),
          primary_eligible_no_generation_errors=all(v['note_execution_status']=='ok' for v in observed),
          limits=['CPUtimingisnottargethardwareinferenceservicelatency.','Allcapsandemptyvalidnotesremain;generationerrorsareflaggedandmakebenefitgateineligible.','Oracleincludesrootanswer;itisnotdeployedprefix-onlygeneration.'])
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
    p.add_argument('--threads',type=int,choices=(8,16),default=16);a=p.parse_args()
    print(json.dumps(score(a.output,a.inputs,a.panel,a.model_dir,a.binding,a.notes,a.threads),allow_nan=False))


if __name__=='__main__':main()

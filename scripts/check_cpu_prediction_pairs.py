"""Compare sealed native paired gains with independent standard CPU forwards.

This is a post-result numerical diagnostic on a fixed subset, not confirmation.
The plan fixes identities, numerical thresholds, models and resource limits before
CPU predictions. Backbones stay frozen and original scientific decisions remain.
"""
from __future__ import annotations
import argparse,gc,json,time
from pathlib import Path
import numpy as np
import torch
from weekend_thinking.artifacts import sha256_file
from weekend_thinking.pretrained import load_base


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();plan=json.loads(args.plan.read_text());start=time.monotonic()
    if not plan['models'] or plan['models'][0]!='B0':
        raise ValueError('Score the untouched base before replacing its weights')
    if sha256_file(Path(__file__))!=plan['CPU_script_sha256']:
        raise ValueError('Preserve the CPU implementation fixed before its predictions')
    if args.out.exists():raise ValueError('Retain original CPU diagnostics; use a new output directory')
    args.out.mkdir();torch.set_num_threads(plan['CPU_threads']);torch.set_num_interop_threads(1)
    source=Path(plan['source_file'])
    if sha256_file(source)!=plan['source_sha256']:raise ValueError('Original source file changed')
    rows={v['item_id']:v for v in map(json.loads,source.open())}
    chosen=plan['item_ids']
    if len(chosen)!=len(set(chosen)) or not set(chosen)<=rows.keys():raise ValueError('Fixed unique subset required')
    readout_path=Path(plan['sealed_readout']);readout=json.loads(readout_path.read_text())
    inventory=readout['models'];evidence=Path(plan['original_scores'])
    if sha256_file(evidence)!=readout['public_scores_sha256']:raise ValueError('Original native score SHA differs')
    if set(inventory)|{'B0'}!=set(plan['models']):raise ValueError('Every prespecified frozen model must remain')
    prefix=plan['cohort']+'__'
    with np.load(evidence,allow_pickle=False) as archive:
        native_ids=archive[prefix+'item_ids'].tolist();native_docs=archive[prefix+'document_ids'].tolist()
        if len(set(native_ids))!=len(native_ids):raise ValueError('Unique native identities required')
        index={value:i for i,value in enumerate(native_ids)}
        if any(native_docs[index[i]]!=rows[i]['document_id'] for i in chosen):raise ValueError('Native/source pairing differs')
        native={m:archive[prefix+m][[index[i] for i in chosen]].astype(np.float64) for m in plan['models']}
    model,_=load_base(plan['base_model_directory']);model.eval();model.requires_grad_(False);cpu={};records=[];token_error=0.;forwards=0
    with torch.inference_mode():
        for name in plan['models']:
            if name!='B0':
                path=Path(inventory[name]['checkpoint'])
                if sha256_file(path)!=inventory[name]['checkpoint_sha256']:raise ValueError('Fixed model checkpoint changed')
                state=torch.load(path,map_location='cpu',weights_only=True,mmap=True);model.load_state_dict(state['model'],strict=True);del state;gc.collect()
            values=[]
            for item in chosen:
                if time.monotonic()-start>plan['maximum_wall_seconds']:raise TimeoutError('CPU diagnostic wall boundary')
                available=next(int(s.split()[1])*1024 for s in Path('/proc/meminfo').read_text().splitlines() if s.startswith('MemAvailable:'))
                if available<plan['minimum_host_available_GiB']*2**30:raise RuntimeError('Protect the available-host-memory floor')
                ids=torch.tensor([rows[item]['ids']],dtype=torch.long)
                if ids.shape!=(1,528):raise ValueError('Unmodified full original window required')
                hidden=model.model(input_ids=ids[:,:-1],use_cache=False).last_hidden_state;pieces=[]
                for first in range(0,527,64):
                    logits=model.lm_head(hidden[:,first:first+64])[0].double();target=ids[0,first+1:first+1+len(logits)]
                    pieces.append((logits.gather(1,target[:,None]).squeeze(1)-torch.logsumexp(logits,-1)).cpu().numpy())
                    del logits,target
                lp=np.concatenate(pieces);values.append(lp);forwards+=1
                records.append({'model':name,'item_id':item,'document_id':rows[item]['document_id'],'full_window_NLL':float(-lp.mean()),'suffix16_log_probability':float(lp[-16:].sum())})
                del ids,hidden,pieces
            cpu[name]=np.asarray(values);token_error=max(token_error,float(np.abs(cpu[name]-native[name]).max()))
            print(json.dumps({'model':name,'completed_documents':len(chosen),'CPU_forwards':forwards}),flush=True)
    comparisons=readout['metrics'][plan['cohort']]['contrasts'];contrasts={};max_mean=max_document=0.
    for key,v in comparisons.items():
        t,c=v['treatment'],v['control'];a=(cpu[t]-cpu[c]).mean(1);b=(native[t]-native[c]).mean(1);mean_error=float(abs(a.mean()-b.mean()));document_error=float(np.abs(a-b).max());max_mean=max(max_mean,mean_error);max_document=max(max_document,document_error)
        contrasts[key]={'treatment':t,'control':c,'CPU_mean_gain_nats_per_token':float(a.mean()),'native_mean_gain_same_documents':float(b.mean()),'absolute_mean_paired_gain_error':mean_error,'maximum_document_paired_gain_error':document_error}
    np.savez_compressed(args.out/'original-CPU-token-scores.npz',**cpu,item_ids=np.asarray(chosen),document_ids=np.asarray([rows[i]['document_id'] for i in chosen]))
    (args.out/'per-document-scores.jsonl').write_text(''.join(json.dumps(v)+'\n' for v in records))
    passed=max_mean<=plan['maximum_mean_paired_gain_error'] and max_document<=plan['maximum_document_paired_gain_error']
    result={'status':'PASS_STANDARD_CPU_PAIRED_GAIN_CHECK' if passed else 'FAIL_STANDARD_CPU_PAIRED_GAIN_CHECK','documents':len(chosen),'models':plan['models'],'actual_CPU_model_forwards':forwards,'new_accelerator_calls':0,'maximum_absolute_mean_paired_gain_error_nats_per_token':max_mean,'maximum_document_paired_gain_error_nats_per_token':max_document,'maximum_individual_token_log_probability_error':token_error,'contrasts':contrasts,'actual_wall_seconds':time.monotonic()-start,'CPU_original_scores_sha256':sha256_file(args.out/'original-CPU-token-scores.npz'),'sealed_native_readout_sha256':sha256_file(readout_path),'plan':plan,'scope':'Numerical diagnostic on a fixed inspected subset, not a fresh scientific confirmation, altered primary decision, tokenwise-equality claim or external replication.'}
    (args.out/'readout.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()

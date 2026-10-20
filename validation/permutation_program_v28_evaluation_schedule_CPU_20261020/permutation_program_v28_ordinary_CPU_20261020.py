"""Fixed ordinary-text retention after all six terminals; CPU FP32, no training.

Reuses the existing256 ordinary pages,256-context/32-target native tokens and
qualified score_spans primitive. It is a secondary specialization-cost diagnostic.
"""
from pathlib import Path
from types import SimpleNamespace
from datetime import datetime,timezone
import argparse,gc,json,time
import torch
from train.permutation_program_science_evaluate import verify_release,sources
from train.evaluate_local import setup_model
from train.document_qa_span_scores import score_spans
from weekend_thinking.artifacts import sha256_file,atomic_json
from weekend_thinking.stats import paired_cluster_bootstrap

R=Path('/home/ubuntu/dario')
INPUT=R/'local/nvme-cpt/evaluation/document-evidence-plan-v23-evidence_select-seed2026101923-likelihood-io-v2/ordinary-raw.jsonl'
INPUT_SHA='7580d34ad27c90627e63863e72386680aa5d5069c14e52bff7a592fd14c72dbd'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--release',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();release=verify_release(args.release)
    assert sha256_file(INPUT)==INPUT_SHA
    data=[json.loads(v) for v in INPUT.read_text().splitlines()]
    assert len(data)==len({v['item_id'] for v in data})==256
    assert all(v['panel']=='ordinary' and v['mode']=='raw' and len(v['context_ids'])==256 and len(v['target_ids'])==32 for v in data)
    args.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(16)
    atomic_json(args.output/'protocol.json',dict(scope='SECONDARY_ORDINARY_TEXT_RETENTION_CPU_NO_LEARNING',
        input_sha256=INPUT_SHA,release_sha256=sha256_file(args.release),source_files=sources(),
        span_scorer_sha256=sha256_file(Path(score_spans.__code__.co_filename)),checker_source_sha256=sha256_file(__file__),
        device='cpu',precision='FP32',threads=16,batch=4,width=288,context_tokens=256,target_tokens=32,
        limits=['Already-inspected ordinary pages,not a fresh general-pretraining confirmation.','No accuracy gate or checkpoint selection from this diagnostic.']))
    all_rows={};costs={}
    for key,spec in sorted(release['model_specs'].items()):
        started=time.monotonic();checkpoint=spec['checkpoint']
        if checkpoint:assert sha256_file(checkpoint)==spec['checkpoint_sha256']
        backend,_,loaded,_,_=setup_model(SimpleNamespace(model_dir=R/'local/assets/qwen3-1.7b-base',checkpoint=Path(checkpoint) if checkpoint else None,device='cpu'))
        assert all(v.device.type=='cpu' and v.dtype==torch.float32 for v in backend.model.parameters())
        if checkpoint:assert all(loaded[k]==spec[k] for k in ('run_id','step','spec_hash'))
        observed=[]
        with (args.output/(key+'-token-scores.jsonl')).open('x') as stream:
            for start in range(0,256,4):
                batch=data[start:start+4]
                values=score_spans(backend,[v['context_ids'] for v in batch],[v['target_ids'] for v in batch],width=288)
                for item,value in zip(batch,values):
                    row=dict(item_id=item['item_id'],document_id=item['item_id'],model=key,score=-value['full_logp']/32,
                        context_ids=item['context_ids'],target_ids=item['target_ids'],**value)
                    stream.write(json.dumps(row,allow_nan=False)+'\n');observed.append(row)
                stream.flush()
        assert len(observed)==256
        all_rows[key]=observed;costs[key]=time.monotonic()-started
        del backend;gc.collect()
        print(json.dumps(dict(model=key,complete=256,host_seconds=costs[key])),flush=True)
    contrasts={}
    for key in all_rows:
        if key!='B0':contrasts[key+'_minus_B0_NLL']=paired_cluster_bootstrap(all_rows['B0'],all_rows[key],repeats=5000,seed=17)
    for seed in (2026102028,2026102029):
        t=f'program-trace-seed{seed}-v1'
        for arm in ('direct','pause'):
            d=f'program-{arm}-seed{seed}-v1';contrasts[t+'_minus_'+d+'_NLL']=paired_cluster_bootstrap(all_rows[d],all_rows[t],repeats=5000,seed=17)
    atomic_json(args.output/'result.json',dict(status='COMPLETE_SECONDARY_ORDINARY256_CPU_DIAGNOSTIC',
        utc=datetime.now(timezone.utc).isoformat(),model_rows={k:len(v) for k,v in all_rows.items()},
        absolute_mean_token_NLL={k:sum(v['score'] for v in rows)/256 for k,rows in all_rows.items()},
        higher_is_worse_NLL_contrasts=contrasts,host_seconds=costs,reserved_accelerator_chip_hours=0,
        limits=['This supervised task recipe has no ordinary LM training term. Report specialization cost separately.','No claim of general pretraining benefit from this inspected panel.','CPU FP32 results are separate from historical accelerator scores; all seven models use the same scorer/inputs.']))


if __name__=='__main__':main()

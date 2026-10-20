"""All-row score reconstruction and fixed four-page independent HF forwards."""
from pathlib import Path
from types import SimpleNamespace
import gc, hashlib, json, time
import numpy as np
import torch
import torch.nn.functional as F
from train.evaluate_local import setup_model
from train.permutation_program_science_evaluate import verify_release
from weekend_thinking.artifacts import atomic_json,sha256_file

R=Path('/home/ubuntu/dario')
C=R/'local/diagnostics/permutation-program-v28-complete-readout-io-v2'
O=R/'local/nvme-cpt/evaluation/permutation-program-v28-ordinary-CPU-v1'
INPUT=R/'local/nvme-cpt/evaluation/document-evidence-plan-v23-evidence_select-seed2026101923-likelihood-io-v2/ordinary-raw.jsonl'
read=lambda p:json.loads(Path(p).read_text())
rows=lambda p:[json.loads(v) for v in Path(p).read_text().splitlines()]


def main():
    torch.set_num_threads(16)
    started=time.monotonic()
    assert sha256_file(INPUT)=='7580d34ad27c90627e63863e72386680aa5d5069c14e52bff7a592fd14c72dbd'
    source=rows(INPUT);gold={v['item_id']:v for v in source}
    assert len(source)==len(gold)==256
    fixed=sorted(gold,key=lambda key:hashlib.sha256(key.encode()).hexdigest())[:4]
    release=verify_release(R/'local/diagnostics/permutation-program-v28-fixed-evaluation-v1/release.json')
    result=read(O/'result.json');all_rows={}
    for model in release['model_specs']:
        values=rows(O/(model+'-token-scores.jsonl'));lookup={v['item_id']:v for v in values}
        assert len(values)==len(lookup)==256 and lookup.keys()==gold.keys()
        for key,v in lookup.items():
            g=gold[key]
            assert v['context_ids']==g['context_ids'] and v['target_ids']==g['target_ids']
            assert v['model']==model and v['document_id']==key and v['tokens']==len(v['token_logps'])==32
            assert all(np.isfinite(v['token_logps']))
            assert abs(sum(v['token_logps'])-v['full_logp'])<1e-10
            assert abs(v['score']+v['full_logp']/32)<1e-10 and v['K1_logp']==v['token_logps'][0]
        mean=sum(v['score'] for v in values)/256
        assert abs(mean-result['absolute_mean_token_NLL'][model])<1e-10
        all_rows[model]=lookup
    for name,expected in result['higher_is_worse_NLL_contrasts'].items():
        treatment,control=name.removesuffix('_NLL').split('_minus_')
        delta=np.array([all_rows[treatment][key]['score']-all_rows[control][key]['score'] for key in sorted(gold)])
        rng=np.random.default_rng(17);draws=[]
        for _ in range(50):
            draws.extend(delta[rng.integers(0,256,size=(100,256))].mean(1).tolist())
        assert abs(float(delta.mean())-expected['difference_b_minus_a'])<1e-10
        assert np.allclose(np.quantile(draws,[.025,.975]),expected['ci95'],atol=1e-10,rtol=0)
    audit={}
    # Threshold and identities fixed before any new forward; no scores changed.
    atomic_json(C/'ordinary-independent-forward-plan.json',dict(fixed_item_ids=fixed,
        selection='Four lowest SHA256 item IDs,shared by all7models; not selected by loss.',
        tolerance_max_token_logp_error=.001,CPU_forward_calls=28,scoring_rule_changes=False))
    for model,spec in sorted(release['model_specs'].items()):
        checkpoint=spec['checkpoint']
        if checkpoint:assert sha256_file(checkpoint)==spec['checkpoint_sha256']
        backend,_,loaded,_,_=setup_model(SimpleNamespace(model_dir=R/'local/assets/qwen3-1.7b-base',
            checkpoint=Path(checkpoint) if checkpoint else None,device='cpu'))
        backend.model.eval()
        if checkpoint:assert all(loaded[k]==spec[k] for k in ('run_id','step','spec_hash'))
        errors=[]
        with torch.no_grad():
            for key in fixed:
                v=gold[key];tokens=v['context_ids']+v['target_ids'][:-1]
                x=torch.tensor([tokens],dtype=torch.long)
                # Independent ordinary model forward,not score_spans/backend.hidden/cache.
                logits=backend.model(x).logits[0,-32:].float()
                lp=F.log_softmax(logits,-1).gather(-1,torch.tensor(v['target_ids'])[:,None])[:,0]
                saved=torch.tensor(all_rows[model][key]['token_logps'])
                error=float((lp-saved).abs().max())
                assert error<=.001
                errors.append(error)
        audit[model]=dict(calls=4,max_token_logp_error=max(errors),per_page_errors=errors)
        print(json.dumps(dict(model=model,independent_HF_pages=4,max_token_logp_error=max(errors))),flush=True)
        del backend;gc.collect()
    report=dict(status='PASS_ALL_ORDINARY_ROWS_MEANS_BOOTSTRAPS_AND28_INDEPENDENT_HF_FORWARDS',
        model_pages=7*256,stored_token_scores=7*256*32,independent_CPU_forward_calls=28,
        reserved_accelerator_chip_hours=0,fixed_item_ids=fixed,model_checks=audit,
        result_sha256=sha256_file(O/'result.json'),checker_sha256=sha256_file(__file__),
        host_seconds=time.monotonic()-started,
        limits='Numerical/source/count verification on existing inspected retention pages,not fresh capability confirmation. Original ordinary scores and all V28 scientific gates unchanged.')
    atomic_json(C/'ordinary-independent-check.json',report)
    print(report['status'],flush=True)


if __name__=='__main__':main()

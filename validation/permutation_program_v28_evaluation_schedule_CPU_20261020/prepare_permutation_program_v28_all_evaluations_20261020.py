"""Prepare exactly23 fixed inputs after the all-six-terminal release, no launch."""
from pathlib import Path
import argparse,json
from train.permutation_program_science_evaluate import prepare,verify_release,MODEL_IDS,PANELS,sources
from weekend_thinking.artifacts import atomic_json,sha256_file,utc_now

R=Path('/home/ubuntu/dario')
C=R/'local/diagnostics/permutation-program-v28-fixed-evaluation-v1'
O=R/'local/nvme-cpt/evaluation/permutation-program-v28-fixed-v1'
D=R/'local/nvme-study/data/permutation-program-v28-study-v1'
BASE=R/'local/diagnostics/postorder-reader-learning-compound5120-sft-v4r2/protocol.seed-20261018.json'


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--release',type=Path,required=True)
    args=parser.parse_args();release=verify_release(args.release)
    assert sha256_file(BASE)=='584ed0f9177d81a7c10ef3891b87b1ad7777169404cdb8b44b81b9897d6d2b62'
    assert C.is_dir() and not (C/'index.json').exists() and not O.exists()
    O.mkdir(parents=True,exist_ok=False);jobs=[];paths=[]
    for model in sorted({'B0',*MODEL_IDS}):
        native=None
        for panel,diagnostic in [(v,False) for v in PANELS]+([('DEV_composition',True)] if '-trace-' in model else []):
            key=model+'-'+panel+('-partial-controls' if diagnostic else '')
            out=O/key
            p=prepare(out,D,R/'local/assets/qwen3-1.7b-base',BASE,args.release,model,panel,
                diagnostic=diagnostic,native_from=native)
            if native is None:native=out
            paths.append(str(out));jobs.append(dict(key=key,model=model,panel=panel,diagnostic=diagnostic,
                items=p['items'],modes=p['modes'],attempts=p['items']*len(p['modes']),
                output=str(out),protocol_sha256=sha256_file(out/'protocol.json')))
            print(json.dumps(dict(key=key,prepared=p['items'],model_forwards='CPU own-model native references only; no held-task inference')),flush=True)
    assert len(paths)==23 and sum(v['attempts'] for v in jobs)==378880
    atomic_json(C/'index.json',paths)
    atomic_json(C/'registry.json',dict(status='PREPARED_FIXED23_INPUTS_NO_ACCELERATOR_EXECUTION',created_utc=utc_now(),
        release_file=str(args.release.resolve()),release_sha256=sha256_file(args.release),source_files=sources(),
        preparer_source_sha256=sha256_file(__file__),controller_source_sha256=sha256_file(R/'local/analysis/supervise_permutation_program_v28_generation_group_20261020.py'),
        index_sha256=sha256_file(C/'index.json'),jobs=jobs,attempts=378880,
        limits=['No held-task model generation or accuracy review during preparation.','CPU native references use only the original canonical contexts; later panels reuse the same model reference.','All fixed jobs must close, failures retained; no prompt/checkpoint/seed selection or automatic retry.']))


if __name__=='__main__':main()

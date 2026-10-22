"""Score original text after all six fixed learning-scaffold fits are frozen."""
import argparse
import json
from pathlib import Path

from train import partial_state_driver as driver
from train.uniform_lm_evaluate import worker
from weekend_thinking.artifacts import sha256_file


def validate(p, release, output):
    if (p['schema'] != 'scaffold-learning-likelihood-v1' or p['world_size'] != 8
            or p['micro_batch'] != 4 or p['source_files'] != driver.source_files()
            or p['evaluator_sha256'] != sha256_file(__file__)
            or sha256_file(p['data_file']) != p['data_sha256']
            or sha256_file(p['data_manifest_file']) != p['data_manifest_sha256']
            or str(output) not in release['jobs']):
        raise ValueError('Frozen original-text evaluator and named data are required')
    terminals=release['all_terminal_models']
    expected={f'scaffold_{arm}-seed{seed}' for arm in ('raw','format','trace') for seed in (11,12)}
    if not expected <= terminals.keys() or any(terminals[k]['step'] != 128 for k in expected):
        raise ValueError('All six prespecified terminal models must freeze before confirmation')
    if p['model_name'] == 'B0':
        if p['checkpoint'] is not None:raise ValueError('Original B0 must stay unchanged')
    elif (p['model_name'] not in terminals or p['checkpoint_sha256'] != terminals[
            p['model_name']]['checkpoint_sha256'] or sha256_file(p['checkpoint']) != p['checkpoint_sha256']):
        raise ValueError('Exact frozen trained-model identity required')
    if p['cohort'] == 'CONFIRM':
        m=json.loads(Path(p['data_manifest_file']).read_text())
        if m['files']['CONFIRM']['sha256'] != p['data_sha256'] or p['items'] != 4096:
            raise ValueError('Complete fresh natural confirmation sources required')
    elif p['cohort'] != 'CALCULATION':raise ValueError('Only named confirmation and old diagnostic cohorts')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('protocol','execution-release','output','stop-file'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--deadline-utc',required=True)
    args=parser.parse_args();p=json.loads(args.protocol.read_text());r=json.loads(args.execution_release.read_text())
    if sha256_file(args.protocol) != r['jobs'][str(args.output)]['protocol_sha256']:
        raise ValueError('Frozen evaluation protocol changed')
    validate(p,r,args.output)
    rows=[json.loads(line) for line in Path(p['data_file']).open()]
    if (args.output.exists() or len(rows) != p['items'] or len(rows)%32
            or len({x['document_id'] for x in rows}) != len(rows)
            or any(len(x['ids']) != 528 for x in rows)):
        raise ValueError('One fresh output and complete unique original windows required')
    import torch_xla.distributed.xla_multiprocessing as xmp
    xmp.spawn(worker,args=(args,p,rows),nprocs=None)


if __name__=='__main__':main()

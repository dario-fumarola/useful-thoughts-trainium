"""Package and verify exact trained BF16 weights relative to the pinned base."""
from pathlib import Path
import argparse,hashlib,json,shutil,time
import numpy as np
import torch
from weekend_thinking.model_delta import bits,decode,encode,qwen_base


def sha(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['base-dir','checkpoint','out']:p.add_argument('--'+name,type=Path,required=True)
    for name in ['checkpoint-sha256','name','training-source','report-url']:p.add_argument('--'+name,required=True)
    p.add_argument('--min-available-gib',type=float,default=0)
    p.add_argument('--min-free-gib',type=float,default=0)
    args=p.parse_args();start=time.monotonic();torch.set_num_threads(4)
    available=next(int(s.split()[1])*1024 for s in Path('/proc/meminfo').read_text().splitlines() if s.startswith('MemAvailable:'))
    if min(args.min_available_gib,args.min_free_gib)<0:raise ValueError('Nonnegative caller resource floors required')
    if available<args.min_available_gib*2**30:raise RuntimeError('Preserve the caller host-memory floor and packaging headroom')
    args.out.parent.mkdir(exist_ok=True,parents=True)
    if shutil.disk_usage(args.out.parent).free-args.checkpoint.stat().st_size<args.min_free_gib*2**30:
        raise RuntimeError('Preserve the filesystem floor even without compression')
    if sha(args.checkpoint)!=args.checkpoint_sha256:raise ValueError('Package the exact frozen native model')
    asset=json.loads((args.base_dir/'ASSET_MANIFEST.json').read_text());names=['model.safetensors','config.json','generation_config.json','tokenizer.json','tokenizer_config.json','vocab.json','merges.txt','LICENSE']
    hashes={name:sha(args.base_dir/name) for name in names}
    if any(hashes[n]!=asset['files'][n]['sha256'] for n in names):raise ValueError('Pinned upstream snapshot changed')
    native=torch.load(args.checkpoint,map_location='cpu',weights_only=True,mmap=True);base=qwen_base(args.base_dir)
    provenance={'model_name':args.name,'base_model_id':asset['model_id'],'base_revision':asset['resolved_revision'],'base_files_sha256':hashes,'base_safetensors_sha256':hashes['model.safetensors'],'native_checkpoint_sha256':args.checkpoint_sha256,'native_optimizer_update':native['step'],'native_spec_hash':native['spec_hash'],'training_source_commit':args.training_source,'report_url':args.report_url,'license':'Apache-2.0; upstream license retained in restored model'}
    encode(base,native['model'],args.out,provenance=provenance);restored,_=decode(base,args.out)
    if set(restored)!=set(native['model']) or any(not np.array_equal(bits(v),bits(native['model'][n])) for n,v in restored.items()):raise ValueError('Every restored tensor must equal original native bits')
    receipt={'status':'PASS_LOSSLESS_ACTUAL_MODEL_TENSOR_ROUNDTRIP','model':args.name,'delta':str(args.out),'delta_sha256':sha(args.out),'delta_bytes':args.out.stat().st_size,'native_model_bytes':args.checkpoint.stat().st_size,'all_native_tensor_bits_match':True,'native_model_tensor_names':len(restored),'wall_seconds':time.monotonic()-start,'new_accelerator_calls':0,'provenance':provenance};args.out.with_suffix('.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt,indent=2))


if __name__=='__main__':main()

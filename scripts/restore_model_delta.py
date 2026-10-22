"""Restore exact trained tensor values as a usable Hugging Face model directory."""
from pathlib import Path
import argparse,hashlib,json,shutil
from safetensors.torch import load_file,save_file
from weekend_thinking.model_delta import bits,decode,qwen_base,tensor_hash


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--base-dir',type=Path,required=True);p.add_argument('--delta',type=Path,required=True);p.add_argument('--out',type=Path,required=True);args=p.parse_args()
    if args.out.exists():raise FileExistsError(args.out)
    base=qwen_base(args.base_dir);restored,manifest=decode(base,args.delta);provenance=manifest['provenance']
    with (args.base_dir/'model.safetensors').open('rb') as f:base_sha=hashlib.file_digest(f,'sha256').hexdigest()
    if base_sha!=provenance['base_safetensors_sha256']:raise ValueError('Use the exact pinned upstream base file')
    expected={'model.safetensors','config.json','generation_config.json','tokenizer.json','tokenizer_config.json','vocab.json','merges.txt','LICENSE'}
    if set(provenance['base_files_sha256'])!=expected:raise ValueError('Complete fixed base-file hashes required')
    for name,digest in provenance['base_files_sha256'].items():
        with (args.base_dir/name).open('rb') as f:actual=hashlib.file_digest(f,'sha256').hexdigest()
        if actual!=digest:raise ValueError('Pinned base configuration/tokenizer differs: '+name)
    # Payload aliases also compress equal-valued, architecturally untied norms.
    # Keep their values exact without introducing new parameter sharing.
    args.out.mkdir();state={};seen=set()
    for name,tensor in restored.items():
        if name=='lm_head.weight':continue
        identity=(tensor.untyped_storage().data_ptr(),tensor.storage_offset())
        state[name]=tensor.clone() if identity in seen else tensor
        seen.add(identity)
    save_file(state,str(args.out/'model.safetensors'),metadata={'format':'pt'})
    check=load_file(str(args.out/'model.safetensors'),device='cpu')
    for name,tensor in state.items():
        if tensor_hash(bits(check[name]))!=tensor_hash(bits(tensor)):raise ValueError('Export must preserve every exact tensor')
    for name in ['config.json','generation_config.json','tokenizer.json','tokenizer_config.json','vocab.json','merges.txt','LICENSE']:
        shutil.copyfile(args.base_dir/name,args.out/name)
    (args.out/'model_delta_provenance.json').write_text(json.dumps(manifest,indent=2)+'\n')
    (args.out/'README.md').write_text('# '+provenance['model_name']+'\n\nContinued training of Qwen/Qwen3-1.7B-Base. Parameters are reconstructed losslessly; the upstream Apache 2.0 license is retained. This directory contains model weights and tokenizer, not optimizer state.\n\nOriginal checkpoint SHA-256: '+provenance['native_checkpoint_sha256']+'\n\nTraining source: '+provenance['training_source_commit']+'\n\nScientific results and limitations: '+provenance['report_url']+'\n')
    print(json.dumps({'status':'PASS_ALL_RESTORED_TENSOR_BITS_AND_HF_EXPORT','model':provenance['model_name'],'directory':str(args.out),'tensors':len(restored),'optimizer_state_included':False,'new_accelerator_calls':0}))


if __name__=='__main__':main()

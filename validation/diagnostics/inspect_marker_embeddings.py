"""Inspect released token-row equality without loading or modifying the model."""
import argparse
import json
from pathlib import Path

import torch
from safetensors import safe_open
from transformers import AutoTokenizer

from weekend_thinking.artifacts import sha256_file


def inspect(directory):
    tokenizer = AutoTokenizer.from_pretrained(directory, local_files_only=True)
    config = json.loads((directory/'config.json').read_text())
    manifest = json.loads((directory/'ASSET_MANIFEST.json').read_text())
    tokens = ('<think>', '</think>', '<|im_start|>', '<|im_end|>', '<|endoftext|>', '\n', '.')
    ids = {s:tokenizer.encode(s, add_special_tokens=False) for s in tokens}
    assert all(len(v) == 1 for v in ids.values())
    with safe_open(directory/'model.safetensors', framework='pt', device='cpu') as stream:
        array = stream.get_slice('model.embed_tokens.weight')
        reference = array[ids['<think>'][0]:ids['<think>'][0]+1]
        tail = array[151644:]
        equal = (tail == reference).all(dim=1)
        equal_ids = (torch.nonzero(equal).flatten()+151644).tolist()
        rows = {}
        for token, (index,) in ids.items():
            row = array[index:index+1].to(torch.float64)
            rows[token] = dict(token_id=index, exactly_equal_to_think=bool(torch.equal(row, reference.to(torch.float64))),
                norm=float(row.norm()), nonzero=int(torch.count_nonzero(row)),
                max_abs_difference_from_think=float((row-reference.to(torch.float64)).abs().max()))
        separate_head = 'lm_head.weight' in stream.keys()
    defined = set(tokenizer.get_vocab().values())
    marker_ids = list(range(151644,151669))
    assert all(i in equal_ids for i in marker_ids)
    return dict(model_id=manifest['model_id'], revision=manifest['resolved_revision'],
        asset_manifest_sha256=sha256_file(directory/'ASSET_MANIFEST.json'),
        released_weight_sha256=manifest['files']['model.safetensors']['sha256'],
        config_sha256=sha256_file(directory/'config.json'),
        source_sha256=sha256_file(__file__),
        released_dtype='BF16; scalar norms/differences computed in FP64',
        defined_token_count=len(defined), model_vocabulary=config['vocab_size'],
        tie_word_embeddings=config['tie_word_embeddings'], separate_head_tensor_present=separate_head,
        rows=rows, identical_defined_added_ids=marker_ids,
        all_ids_equal_to_think_in_inspected_tail=equal_ids,
        inspected_tail_start=151644,
        interpretation='These added token identities are indistinguishable through their input embedding at B0. This explains identical instruction scores when substituting think/end-think at otherwise identical positions. The qualified loader ties the output head, giving equal head rows too.',
        limitations=['No inference about training history or why these weights are equal.',
            'Surrounding text and positions can still distinguish boundary roles.',
            'This observation alone does not explain all format/content penalties or establish a remedy.'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = inspect(args.model_dir)
    with args.output.open('w') as stream:
        json.dump(report, stream, indent=2, allow_nan=False); stream.write('\n')

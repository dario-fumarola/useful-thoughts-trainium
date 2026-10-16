#!/usr/bin/env python3
"""Resolve an explicitly paraphrased RLP-like prompt on a local tokenizer.

The PDF prints a system prompt (page 21); its printed delimiter line appears
to use the closing delimiter twice. This script uses explicit opening/closing
markers and a NEW paraphrased instruction, not an exact reproduction.
No new vocabulary entries or randomly initialized embeddings are introduced.
"""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from weekend_thinking.core import Format

p=argparse.ArgumentParser();p.add_argument('--tokenizer-dir',required=True)
p.add_argument('--out',type=Path,required=True);a=p.parse_args()
from transformers import AutoTokenizer
tok=AutoTokenizer.from_pretrained(a.tokenizer_dir,local_files_only=True,trust_remote_code=False)
header=('Consider what logically follows from the supplied passage. Put your scratch work '
        'inside the marked section, then resume the passage with matching notation and tone. '
        'Do not replace the continuation with a general explanation.\n\nPassage:\n')
encode=lambda s:tuple(tok.encode(s,add_special_tokens=False))
stop=encode('</think>')
if len(stop)!=1:
    raise SystemExit('The current stop-mask implementation needs a single-token </think>. Do not guess an ID; implement and test multi-token stopping separately.')
if tok.eos_token_id is None:raise SystemExit('Tokenizer has no EOS/pad fallback; inspect explicitly.')
eos=int(tok.eos_token_id)
fmt=Format(encode('\n<think>'),encode('</think>\n'),stop[0],eos,encode(header),
           additional_stop_ids=(() if eos==stop[0] else (eos,)))
result={'format':asdict(fmt),'header_text':header,'open_text':'\n<think>',
        'close_text':'</think>\n','source':'Paraphrase informed by RLP v2 PDF page 21, not exact paper serialization.',
        'extra_stop_policy':'Both the explicit thought-end token and the base EOS terminate generation. The actually sampled first terminator is included in the policy loss, stripped from note content, and rendered with the reader close delimiter. Log which terminator occurred.',
        'tokenizer_dir':a.tokenizer_dir}
from weekend_thinking.artifacts import atomic_json
atomic_json(a.out,result)
print(json.dumps(result,indent=2))

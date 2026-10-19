"""CPU-only replay boundaries and all255 complete canonical spans; no model."""
from pathlib import Path
import hashlib
import json

from transformers import AutoTokenizer
from document_evidence_v23_likelihood_contexts import context_text, scoring_example, selected_ids

R=Path('/home/ubuntu/dario')
D=R/'local/nvme-cpt/data/document-evidence-plan-v23-TRAIN2048-held255-v2'
tok=AutoTokenizer.from_pretrained(R/'local/assets/qwen3-1.7b-base',local_files_only=True,trust_remote_code=False)
read=lambda p:[json.loads(line) for line in p.open()]
questions,gold=read(D/'held255-questions-only.jsonl'),read(D/'held255-gold-only.jsonl')
maximum=0
for q,g in zip(questions,gold):
    assert q['item_id']==g['item_id']
    examples=[scoring_example(q,g['answer'],mode,tok,indices)
        for mode,indices in [('direct',None),('empty',None),('oracle',g['support_indices'])]]
    assert examples[0]['target_ids']==examples[1]['target_ids']==examples[2]['target_ids']
    maximum=max(maximum,*[len(row['target_ids']) for row in examples])
q=questions[0]
for text in ['Evidence: [1] [1]\nFinal answer:\nWRONG\n',
             'Claim: WRONG\nEvidence: [1]\nFinal answer:\nWRONG\n',
             'Evidence: [999]\nFinal answer:\nWRONG\n',
             'Evidence: [1]\n']:
    ids,flag=selected_ids(dict(status='ok',raw_text=text),len(q['contexts']))
    assert ids==[] and flag['fallback']
    assert context_text(q,'selected_id_replay',ids)==context_text(q,'empty')
ids,flag=selected_ids(dict(status='ok',raw_text='Evidence: [2] [1]\nFinal answer:\nWRONG\n'),len(q['contexts']))
assert ids==[2,1] and not flag['fallback']
assert context_text(q,'selected_id_replay',ids).endswith('Evidence: [2] [1]\nFinal answer:\n')
a=scoring_example(q,'FUTURE-A','selected_id_replay',tok,ids)
b=scoring_example(q,'FUTURE-B','selected_id_replay',tok,ids)
assert a['context_ids']==b['context_ids'] and a['target_ids']!=b['target_ids']
for invalid in [[0],[True],[999],[1,1]]:
    try:context_text(q,'selected_id_replay',invalid)
    except ValueError:pass
    else:raise AssertionError('Invalid pointer silently repaired')
result=dict(status='PASS_CPU_POINTER_REPLAY_BOUNDARIES_NO_MODEL_FORWARD',
    questions=255,native_spans=765,max_target_plus_EOS_tokens=maximum,
    checks='All canonical native targets match acrossdirect/empty/oracle;invalid plans equal flagged empty replay;generated ID order preserved;gold answer changes only target IDs;invalid pointer values rejected.',
    helper_sha256=hashlib.sha256((R/'local/analysis/document_evidence_v23_likelihood_contexts.py').read_bytes()).hexdigest(),
    limits='Replay diagnostics are distinct from native generated-answer observations. No real model or accelerator called.')
p=R/'local/diagnostics/document-evidence-plan-v23-CPU-v1/likelihood-context-check.json';assert not p.exists();p.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))

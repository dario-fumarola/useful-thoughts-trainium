"""Measure explicit final-answer CE coefficients, not total useful gradients."""
from pathlib import Path
import hashlib
import json
import statistics

from transformers import AutoTokenizer
from train import document_qa_packing as packing

R = Path('/home/ubuntu/dario')
D = R / 'local/diagnostics/document-QA-v21-answer-coefficients-v1'
V = R / 'local/nvme-cpt/data/document-QA-v21-TRAIN2048-held512-v1/TRAIN2048.jsonl'
W = R / 'local/worktrees/document-qa-v21'
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert sha(Path(packing.__file__)) == sha(W / 'train/document_qa_packing.py')
assert sha(V) == 'a0a99d866b55c16d7e6f498ee881d87556dfa5eabcab3340feccbf11313476e9'
D.mkdir(exist_ok=False)
tokenizer = AutoTokenizer.from_pretrained(R / 'local/assets/qwen3-1.7b-base',
                                         local_files_only=True, trust_remote_code=False)
rows = [json.loads(line) for line in V.open()]
observations = []
for index, row in enumerate(rows):
    packs = {view: packing.training_pack(row, tokenizer, view)
             for view in ('direct', 'prior', 'post')}
    lengths = {view: int(pack['future_mask'].sum()) for view, pack in packs.items()}
    ordinary = packs['direct']['labels'][packs['direct']['future_mask']].tolist()
    prefix = packing.encode(tokenizer,
        packing.prompt(row['question'], 'prior') + row['trace'] + packing.FINAL_OPEN)
    prior = packs['prior']
    stop = int(prior['target_positions'][prior['future_mask']].max()) + 1
    native = prior['input_ids'][0, :stop].tolist()
    assert native[:len(prefix)] == prefix and native[len(prefix):] == ordinary
    observation = dict(item_id=row['item_id'], target_lengths=lengths,
        final_native_labels_identical_across_paths=True,
        augmented_final_token_coefficient_ratio_to_direct={
            view: lengths['direct']/lengths[view] for view in ('prior', 'post')},
        four_pass_average_final_token_coefficient_ratio_to_direct={
            view: .5 + .5*lengths['direct']/lengths[view] for view in ('prior', 'post')})
    observations.append(observation)
    if (index+1) % 512 == 0:
        print(json.dumps({'checked_training_rows': index+1}), flush=True)
counts = dict(qa_direct=4*sum(r['target_lengths']['direct'] for r in observations),
    qa_prior=2*sum(r['target_lengths']['direct']+r['target_lengths']['prior'] for r in observations),
    qa_post=2*sum(r['target_lengths']['direct']+r['target_lengths']['post'] for r in observations))
assert counts == {'qa_direct': 42832, 'qa_prior': 251440, 'qa_post': 259694}


def summarize(values):
    return dict(mean=statistics.mean(values), median=statistics.median(values),
                minimum=min(values), maximum=max(values))


(D/'per-question.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in observations))
result = dict(status='TRAIN_NATIVE_COEFFICIENT_INVENTORY_COMPLETE_NO_MODEL_FORWARD',
    training_questions=2048, four_pass_task_label_counts=counts,
    augmented_final_token_coefficient_ratio_to_direct={view: summarize([
        r['augmented_final_token_coefficient_ratio_to_direct'][view] for r in observations])
        for view in ('prior', 'post')},
    four_pass_average_final_token_coefficient_ratio_to_direct={view: summarize([
        r['four_pass_average_final_token_coefficient_ratio_to_direct'][view] for r in observations])
        for view in ('prior', 'post')},
    limits='Explicit CE coefficients on designated final answer+EOS only; NOT gradient magnitudes, all answer-bearing trace tokens, causal attribution of V21 failure or population generalization. Trace root answers add other answer-bearing supervision. Counts reproduce the published complete training inventories. Whole-pass scheduling puts augmented examples in the last64 updates.',
    training_sha256=sha(V), packing_sha256=sha(Path(packing.__file__)),
    per_question_sha256=sha(D/'per-question.jsonl'), source_sha256=sha(__file__))
(D/'result.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2),flush=True)

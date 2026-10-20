"""Bounded B0 TRAIN-only decoder-equivalence and forced-length speed check."""
from pathlib import Path
from types import SimpleNamespace
from dataclasses import asdict
import argparse
import json
import subprocess
import time
from statistics import median
import torch

from train.batched_causal_decoder import BatchedGreedyDecoder
from train.document_qa_decoder import QADecoder
from train.evaluate import asset_provenance
from train.evaluate_local import setup_model
from train.neuron_backend import CausalCache
from train.permutation_program_data import question_from_dict
from train.permutation_program_packing import inference_context
from train.useful_note_precision import compare_distributions
from weekend_thinking.artifacts import atomic_json, sha256_file

R = Path('/home/ubuntu/dario')
W = R / 'local/worktrees/causal-decoder-batching-v1'
C = R / 'local/diagnostics/causal-decoder-batching-v1'
ASSET = R / 'local/assets/qwen3-1.7b-base'
TRAIN = R / 'local/nvme-study/data/permutation-program-v28-study-v1/TRAIN-questions.jsonl'
read = lambda p: json.loads(Path(p).read_text())


def unchanged(plan):
    assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=W, text=True).strip() == plan['candidate_commit']
    assert not subprocess.check_output(['git', 'status', '--porcelain'], cwd=W, text=True).strip()
    for name, digest in plan['sources'].items():
        assert sha256_file(Path(name)) == digest
    assert asset_provenance(ASSET) == plan['assets']


def prepare():
    C.mkdir(exist_ok=False)
    torch.set_num_threads(16)
    backend, tokenizer, _, signature, _ = setup_model(SimpleNamespace(model_dir=ASSET, checkpoint=None, device='cpu'))
    assert sha256_file(TRAIN) == 'e099948826aaf7e262c06beabdd77e78c126cce8fdeed2714fa42db7c2d2d630'
    examples = []
    length = None
    for line in TRAIN.read_text().splitlines():
        row = json.loads(line)
        assert set(row) == {'item_id', 'family_id', 'question'}
        question = question_from_dict(row['question'])
        if len(question.program) != 6:
            continue
        ids = inference_context(question, tokenizer, 'trace')
        if length is None:
            length = len(ids)
        if len(ids) == length:
            examples.append(dict(item_id=row['item_id'], prompt_ids=ids))
        if len(examples) == 8:
            break
    assert len(examples) == 8 and length <= 256
    atomic_json(C / 'inputs.json', examples)
    vectors = []
    with torch.no_grad():
        for row in examples:
            ids = row['prompt_ids']
            x = torch.full((1, 256), 151643, dtype=torch.long)
            x[0, :len(ids)] = torch.tensor(ids)
            hidden = backend.hidden(x)
            vectors.append(backend.model.lm_head(hidden[:, len(ids)-1]).float().log_softmax(-1))
    torch.save(torch.cat(vectors), C / 'CPU-native.pt')
    names = [Path(__file__), W / 'train/batched_causal_decoder.py', W / 'train/evaluate.py',
             W / 'train/document_qa_decoder.py', W / 'train/neuron_backend.py',
             W / 'train/evaluate_local.py', W / 'weekend_thinking/core.py']
    plan = dict(status='PREPARED_ENGINEERING_TRAIN_ONLY_NOT_SCIENTIFIC_EVALUATION',
        candidate_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=W, text=True).strip(),
        sources={str(p): sha256_file(p) for p in names}, assets=asset_provenance(ASSET),
        base_signature=signature, inputs_sha256=sha256_file(C/'inputs.json'),
        cpu_native_sha256=sha256_file(C/'CPU-native.pt'), train_sha256=sha256_file(TRAIN),
        batch_sizes=[1, 4, 8], prompts=8, forced_actions=32, timed_repeats=3,
        physical_device=8, logical_core=32, maximum_reserved_chip_hours=.5,
        gate='All actual full-vocabulary native comparisons pass standard gate and max error<=1e-3; all forced greedy sequences exactly match original serial decoder. Speed is descriptive.',
        limits='No model update, gold labels, held task data, quality score, runtime-stack change, or V28 evaluator modification. One spare logical core; reserved time charges its whole physical chip. Forced-length microbenchmark is not an end-to-end serving result.')
    atomic_json(C / 'plan.json', plan)
    unchanged(plan)
    print('PREPARED_B0_TRAIN8_CPU_NATIVE_REFERENCE_NO_HELD_ROWS', flush=True)


def run():
    plan = read(C / 'plan.json')
    unchanged(plan)
    assert sha256_file(C/'inputs.json') == plan['inputs_sha256']
    assert sha256_file(C/'CPU-native.pt') == plan['cpu_native_sha256']
    rows = read(C / 'inputs.json')
    prompts = [row['prompt_ids'] for row in rows]
    backend, tokenizer, _, signature, _ = setup_model(SimpleNamespace(model_dir=ASSET, checkpoint=None, device='xla'))
    assert signature == plan['base_signature']
    cpu = torch.load(C/'CPU-native.pt', map_location='cpu', weights_only=True)
    old = QADecoder(backend.model, tokenizer, sync=backend.sync, banned_ids=backend.banned_ids)
    new = BatchedGreedyDecoder(backend.model, tokenizer, sync=backend.sync, banned_ids=backend.banned_ids)
    native = {}
    for size in plan['batch_sizes']:
        values = []
        with torch.no_grad():
            for offset in range(0, 8, size):
                prefix = prompts[offset:offset+size]
                x = torch.full((size, 256), 151643, dtype=torch.long)
                x[:, :len(prefix[0])] = torch.tensor(prefix)
                decoder = old if size == 1 else new
                cache = CausalCache(backend.model.config.num_hidden_layers, 288)
                logits = decoder._next(x.to(backend.device), torch.arange(256, device=backend.device),
                                      cache, torch.tensor([len(prefix[0])-1], device=backend.device))
                logps = logits.log_softmax(-1)
                backend.sync()
                values.append(logps.cpu())
        actual = torch.cat(values)
        torch.save(actual, C/f'XLA-native-B{size}.pt')
        checks = compare_distributions(cpu, actual)
        assert all(v['passed'] and v['max_logp_error'] <= .001 for v in checks)
        native[str(size)] = checks
        atomic_json(C/f'native-B{size}.json', checks)
        print(f'NATIVE_B{size}_ALL8_PASS', flush=True)

    def measure(size):
        backend.sync()
        started = time.monotonic()
        output = []
        for offset in range(0, 8, size):
            prefix = prompts[offset:offset+size]
            if size == 1:
                value = old.decode(prefix[0], mode='prior', max_new_tokens=32, prefill_width=256,
                    pad_id=151643, stop_ids=(), answer_parser=lambda text, mode: SimpleNamespace(status='incomplete'))
                output.append(asdict(value))
            else:
                output.extend(asdict(v) for v in new.decode_batch(prefix, max_new_tokens=32,
                    prefill_width=256, pad_id=151643, stop_ids=()))
        backend.sync()
        return output, time.monotonic()-started

    expected, warm_old = measure(1)
    assert all(len(v['actions']) == 32 and v['stop_type'] == 'length_cap' for v in expected)
    atomic_json(C/'original-serial-attempts.json', expected)
    warm = {'1': warm_old}
    for size in (4, 8):
        actual, seconds = measure(size)
        atomic_json(C/f'batched-B{size}-attempts.json', actual)
        assert actual == expected, 'Retain any mismatch; no tolerance or parser rescue'
        warm[str(size)] = seconds
    timings = {str(size): [] for size in (1, 4, 8)}
    for order in ((1, 4, 8), (4, 8, 1), (8, 1, 4)):
        for size in order:
            actual, seconds = measure(size)
            assert actual == expected
            timings[str(size)].append(seconds)
            atomic_json(C/'timings.json', timings)
            print(json.dumps(dict(batch=size, completed_repeats=len(timings[str(size)]), seconds=seconds)), flush=True)
    medians = {key: median(value) for key, value in timings.items()}
    atomic_json(C/'worker-result.json', dict(status='PASS_B0_TRAIN_NATIVE_AND_EXACT_FORCED_GREEDY_BATCH_PARITY',
        plan_sha256=sha256_file(C/'plan.json'), native=native, warm_seconds=warm,
        timed_seconds=timings, median_seconds=medians,
        forced_generated_actions_per_timing=256,
        speedup_vs_serial={key:medians['1']/value for key,value in medians.items()},
        limits=plan['limits']))
    print('BOUNDED_TRAIN_ONLY_ENGINEERING_CHECK_COMPLETE', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('prepare', 'run'))
    args = parser.parse_args()
    (prepare if args.stage == 'prepare' else run)()

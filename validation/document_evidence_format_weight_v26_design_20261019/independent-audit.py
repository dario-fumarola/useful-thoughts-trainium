"""Reread all actual long-context QA state after the two mechanics stages close.

Run once from the frozen evidence-plan worktree. This is a CPU audit, not another model
evaluation. The latest scientific full256 backup remains protected separately.
"""
from pathlib import Path
from datetime import datetime, timezone
import gc
import hashlib
import json
import math
import random
import os
import shutil
import torch
from train.partial_state_driver import ROOT, batch_at, checkpoint_identity, order_for
from train.useful_note_precision import compare_distributions
from train.useful_note_recovery import optimizer_digest
from weekend_thinking.artifacts import canonical_hash
from weekend_thinking.pretrained import load_base

R = Path('/home/ubuntu/dario')
W = R / 'local/worktrees/document-evidence-format-weight-v26'
assert ROOT == W
C = R / 'local/diagnostics/document-evidence-format-weight-v26-world8-qualification-v1'
O = R / 'local/nvme/diagnostics/document-evidence-format-weight-v26-world8-qualification-v1'


def read(path):
    return json.loads(Path(path).read_text())


def rows(path):
    return [json.loads(v) for v in Path(path).read_text().splitlines()]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def equal(a, b):
    return a.dtype == b.dtype and a.shape == b.shape and torch.equal(
        a.contiguous().reshape(-1).view(torch.uint8), b.contiguous().reshape(-1).view(torch.uint8))


def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


assert read(C / 'qualification-completion.json')['status'] == 'ALL_TWO_STAGES_CLOSED_PENDING_INDEPENDENT_AUDIT'
p = read(C / 'protocol.json')
protocol_sha = sha(C / 'protocol.json')
assert p['schema'] == 'document-evidence-plan-v23' and p['target_slots'] == 512 and p['accumulation'] == 4
assert p['source_loss_weight'] == p['answer_loss_weight'] == .5 and p['selection_loss_weight'] == 1.
assert all(sha(W / k) == v for k, v in p['source_files'].items())
assert sha(p['data_file']) == p['data_sha256']
for stage, output, step in [('direct-fresh4', O, 4), ('direct-resume8', O, 8)]:
    d = C / stage
    assert read(d / 'exit.json')['exit_code'] == 0 and read(d / 'exit.json')['external_stop'] is None
    closure = read(d / 'closure.json')
    assert closure['all_original_session_processes_absent'] and closure['assigned_physical12_13_empty']
    assert len(read(d / 'first-live.json')['actual8_on_physical12_13']) == 8
    completion = O / 'recoveries/actual-restart-001/previous-completion.json' if stage == 'direct-fresh4' else output / 'completion.json'
    assert sha(completion) == closure['completion_sha256']
    for k, v in closure['checkpoint_manifests'].items():
        assert sha(output / f'step-{int(k):07d}/COMPLETE.json') == v

model, tok = load_base(p['model_dir'])
params = list(model.named_parameters())
assert len(params) == 310
base = read(p['base_protocol'])
reference = torch.load(Path(base['native_reference_output']) / 'next-logps.pt', weights_only=True, map_location='cpu')
native = []
for directory in [O, O / 'recoveries/actual-restart-001/initialization']:
    for rank in range(8):
        receipt = read(directory / f'qualification-rank-{rank:05d}.json')
        i = receipt['reference_index']
        actual = torch.load(directory / f'forward-rank-{rank:05d}.pt', weights_only=True, map_location='cpu')
        check = compare_distributions(reference[i:i+1], actual)[0]
        assert check['passed'] and receipt['status'] == 'PASS'
        native.append(check)
        witness = torch.load(directory / f'initial-master-rank-{rank:05d}.pt', weights_only=True, map_location='cpu')
        assert equal(witness['expected_cpu'], witness['actual_fp32_masters'])
print('All16 actual native/master witnesses pass;four canonical contexts repeated.', flush=True)

data = rows(p['data_file'])[:128]
order = []
for epoch in range((p['steps']*32+p['items']-1)//p['items']):
    indices=list(range(p['items']));random.Random(p['seed']+epoch).shuffle(indices);order.extend(indices)
order=order[:p['steps']*32]
all_metrics, full_states, memory = {}, {}, {}
for arm, output, count, checkpoints in [('evidence_direct', O, 8, [2, 4, 6, 8])]:
    metrics = rows(output / 'metrics.jsonl')
    assert [v['step'] for v in metrics] == list(range(1, count+1))
    run = read(output / 'run.json')
    assert run['order_hash'] == canonical_hash(order) and run['protocol_sha256'] == protocol_sha
    assert run['arm'] == arm
    labels = 0
    for i, value in enumerate(metrics):
        assert value['ordered_item_ids'] == [data[k]['item_id'] for k in order[i*32:(i+1)*32]]
        indices=order[i*32:(i+1)*32]
        expected_source=32*256
        # Reconstruct targets independently;do not call either training packer.
        expected_answer,expected_selection=0,0
        for index in indices:
            row=data[index];q=row['question']
            text='Documents:\n'+'\n\n'.join(f'[{j}] {c["title"]}\n{c["text"]}' for j,c in enumerate(q['contexts'],1))+'\n\nQuestion: '+q['question']+'\n'
            instruction='Answer the question using the documents. Write Final answer: on its own line and the answer on the next line.\n'
            direct=(index+i*32//128)%2==0
            mode='Give the final answer directly.\nResponse:\n' if direct else 'You may first list relevant document numbers in an Evidence: field, then give the final answer.\nResponse:\n'
            opening=text+instruction+mode
            plan=''
            if not direct:
                selected=row['support_indices'] if arm=='evidence_select' else []
                plan='Evidence:'+''.join(f' [{j}]' for j in selected)+'\n'
            prefix=tok.encode(opening+plan,add_special_tokens=False)
            seq=tok.encode(opening+plan+'Final answer:\n'+row['answer'],add_special_tokens=False)+[151643]
            final=tok.encode('Final answer:\n'+row['answer'],add_special_tokens=False)+[151643]
            assert seq[:len(prefix)]==prefix and seq[len(prefix):]==final
            expected_answer+=len(final)
            expected_selection+=len(prefix)-len(tok.encode(opening,add_special_tokens=False))
        labels += expected_source+expected_answer+expected_selection
        component=value['loss_components']
        assert component['source_labels']==expected_source
        assert component['answer_labels']==expected_answer
        assert component['selection_labels']==expected_selection
        assert component['source_weight']==component['answer_weight']==.5
        assert component['selection_weight']==1.
        assert all(math.isfinite(component[k]) for k in ['source_mean_loss','answer_mean_loss','selection_mean_loss'])
        expected_loss=.5*component['source_mean_loss']+.5*component['answer_mean_loss']+1.*component['selection_mean_loss']
        assert abs(value['loss']-expected_loss)<1e-5
        assert value['counters'] == dict(expressions=(i+1)*32, supervised_tokens=labels)
        assert math.isfinite(value['loss']) and math.isfinite(value['gradient_norm']) and value['gradient_norm'] > 0
    all_metrics[arm], full_states[arm], memory[arm] = metrics, {}, {}
    for step in checkpoints:
        d = output / f'step-{step:07d}'
        manifest = read(d / 'COMPLETE.json')
        assert manifest['metadata']['counters'] == metrics[step-1]['counters']
        assert set(manifest['files']) == {'model.pt', *[f'rank-{i:05d}.pt' for i in range(8)]}
        for name, v in manifest['files'].items():
            assert (d / name).stat().st_size == v['bytes'] and sha(d / name) == v['sha256']
        weights = torch.load(d / 'model.pt', weights_only=True, map_location='cpu', mmap=True)
        active = weights['model']
        assert set(active) == set(model.state_dict())
        assert all(v.dtype == torch.bfloat16 and torch.isfinite(v).all() for v in active.values())
        assert active['lm_head.weight'].untyped_storage().data_ptr() == active['model.embed_tokens.weight'].untyped_storage().data_ptr()
        post = []
        for rank in range(8):
            saved = torch.load(d / f'rank-{rank:05d}.pt', weights_only=True, map_location='cpu', mmap=True)
            assert checkpoint_identity(manifest, weights, saved, p, arm, output.name, rank,
                                       canonical_hash(order), protocol_sha) == step
            opt = saved['optimizer']
            assert set(opt['sharded_master_weights']) == set(opt['base_state']) == set(range(310))
            for index, (name, param) in enumerate(params):
                chunk = (param.shape[0]+7)//8
                shape = (chunk, *param.shape[1:])
                master, state = opt['sharded_master_weights'][index], opt['base_state'][index]
                assert master.dtype == torch.float32 and tuple(master.shape) == shape and torch.isfinite(master).all()
                assert set(state) == {'exp_avg', 'exp_avg_sq', 'step', 'max_exp_avg_sq'}
                for key in ['exp_avg', 'exp_avg_sq']:
                    tensor = state[key]
                    assert tensor.dtype == torch.float32 and tuple(tensor.shape) == shape and torch.isfinite(tensor).all()
                assert (state['exp_avg_sq'] >= 0).all() and float(state['step']) == step and state['step'].dtype == torch.float32
                assert state['max_exp_avg_sq'].dtype == torch.float32 and state['max_exp_avg_sq'].numel() == 0
                start, end = rank*chunk, min((rank+1)*chunk, param.shape[0])
                used = max(0, end-start)
                if used:
                    assert equal(active[name][start:end], master[:used].bfloat16())
            obs = read(d / f'cpu-export-release-rank-{rank:05d}.json')
            assert obs['cleanup_enabled']
            post.append(obs['PSS_after_bytes'])
            if arm == 'evidence_direct' and step == 4:
                restored = read(O / f'recoveries/actual-restart-001/restored-rank-{rank:05d}.json')
                assert all(restored[k] for k in ['full_active_model_bytes_equal', 'full_optimizer_bytes_equal', 'tied_parameter_objects_preserved'])
                assert restored['optimizer_digest'] == optimizer_digest(opt)
            del saved, opt, state, master, tensor
        memory[arm][str(step)] = sum(post)
        full_states[arm][str(step)] = dict(COMPLETE_sha256=sha(d / 'COMPLETE.json'),
            all_model_master_moments_counters_and_cast_bytes_verified=True)
        del weights, active
        gc.collect()
        print('Verified all state bytes/shapes/counters', arm, step, flush=True)

growth=dict(format_fresh=(memory['evidence_direct']['4']-memory['evidence_direct']['2'])/2**30,
    format_resumed=(memory['evidence_direct']['8']-memory['evidence_direct']['6'])/2**30)
# Observe bounded memory;no post-hoc growth threshold was declared for this check.
A = R / 'local/backups/engineering' / C.name / 'POST_RUN'
A.mkdir(parents=True, exist_ok=False)
archived = {}
for label, folder in [('format-control', O), ('control', C)]:
    for src in folder.rglob('*'):
        if not src.is_file():
            continue
        # All original full states/models remain intact on NVMe. The
        # disposable engineering models do not consume the scientific EBS
        # reserve;small actual-state/native/recovery evidence is protected.
        if src.suffix == '.pt' and src.parent.name.startswith('step-'):
            continue
        dst = A / label / src.relative_to(folder)
        dst.parent.mkdir(parents=True, exist_ok=True)
        assert shutil.disk_usage(R).free > 35*2**30 + src.stat().st_size
        shutil.copyfile(src, dst)
        with dst.open('rb') as stream:
            os.fsync(stream.fileno())
        assert sha(src) == sha(dst)
        archived[str(dst.relative_to(A))] = dict(bytes=src.stat().st_size, sha256=sha(src))
write(A / 'ARCHIVE.json', dict(status='ALL_SMALL_ACTUAL_QUALIFICATION_EVIDENCE_READBACK_FSYNC_MODELS_RETAINED_NVME',
                              full_states_retained_on_original_NVMe=True, files=archived))
assert sum(v['actual_master_and_replica_audit'] for metrics in all_metrics.values() for v in metrics)==6
result = dict(status='PASS_ACTUAL_DOCUMENT_EVIDENCE_FORMAT_WEIGHT_FULL_RESTORE', utc=datetime.now(timezone.utc).isoformat(),
    actual_updates=dict(evidence_direct=8),qualified_arms=['evidence_direct'],
    qualified_training_shape={k:p[k] for k in ['world_size','micro_batch','global_batch','width','target_slots','head_chunk','accumulation']},
    full_states=full_states, actual_native_checks=native, all8_saved4_optimizer_digests_recomputed=True,
    grouped_post_save_PSS_bytes=memory, grouped_late_growth_GiB_per_save=growth, metrics=all_metrics,
    source_files=p['source_files'], protocol_sha256=protocol_sha, audit_source_sha256=sha(__file__),
    EBS_archive_receipt_sha256=sha(A / 'ARCHIVE.json'),
    limits=['Eight bounded real updates,not capability evidence or indefinite-memory guarantee.',
            'No uninterrupted/resumed bitwise-parity claim.16native checks repeat4 canonical contexts.',
            'Disposable qualification model excluded from science;the coefficient treatment must start fresh B0.',
            'Original four full states/models remain NVMe;EBS holds small engineering evidence and all scientific terminal models/latest full science state separately.',
            'RNG restoration runs in the verified driver;no independent immediate device-RNG readback or stochastic-RL qualification.'])
write(C / 'qualification-independent-audit.json', result)
print(json.dumps({k:v for k,v in result.items() if k not in ['metrics','source_files','actual_native_checks','full_states']}, indent=2), flush=True)

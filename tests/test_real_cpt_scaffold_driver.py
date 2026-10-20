"""Qualification and guarded fixed-arm study identities; no hardware claims."""
from collections import Counter
import json
from types import SimpleNamespace

import pytest

from train import partial_state_driver as driver
from test_real_cpt_scaffold import Tokenizer, example


def protocol():
    return dict(schema=driver.SCAFFOLD_SCHEMA, model_id='Qwen/Qwen3-1.7B-Base',
        experiment_id='real-cpt-scaffold-v29', purpose='real_cpt_scaffold_qualification',
        items=128, steps=8, save_every=4, seed=2026102029, lr=1e-5,
        world_size=8, micro_batch=4, global_batch=32, accumulation=1,
        auxiliary_micro_batch=1, width=640, target_slots=640, head_chunk=32,
        schedule='constant', parameter_dtype='bf16', optimizer_dtype='fp32',
        view_schedule='clean_plus_raw_format_computation_cycle',
        loss='clean_LM_plus_fraction_weighted_auxiliary_continuation_work',
        source_loss_weight=1., continuation_loss_weight=.5, work_loss_weight=1.,
        auxiliary_batch_fraction=.25, parent_data_sha256=driver.SCAFFOLD_PARENT_SHA,
        data_manifest_sha256=driver.SCAFFOLD_MANIFEST_SHA, data_sha256='a'*64,
        checkpoint_export_cleanup=True, checkpoint_memory_diagnostic=True,
        source_files=driver.source_files())


@pytest.mark.parametrize('change', [dict(purpose='real_cpt_scaffold_study'),
    dict(steps=1280), dict(items=20480), dict(world_size=4),
    dict(auxiliary_micro_batch=4), dict(auxiliary_batch_fraction=1),
    dict(continuation_loss_weight=True), dict(work_loss_weight=.1),
    dict(parent_data_sha256='b'*64), dict(data_manifest_sha256='b'*64),
    dict(checkpoint_export_cleanup=False)])
def test_no_science_or_changed_loss_shape_release(change):
    p = protocol(); driver.validate(p)
    with pytest.raises(ValueError): driver.validate({**p, **change})


def test_actual_two_pass_order_and_rank_batches_cover_all_three_conditions():
    p = protocol(); driver.validate(p)
    rows = [example(document=f'c{i}') for i in range(32)]
    rows += [example('ordinary', document=f'o{i}') for i in range(96)]
    order = driver.order_for(p, rows)
    assert Counter(order) == Counter({i:2 for i in range(128)})
    for step in range(8):
        for rank in range(8):
            passes = driver.scaffold_passes_at(rows, Tokenizer(), p, order,
                driver.SCAFFOLD_QUALIFICATION_ARM, step, rank)
            (name, source, sw), (aux_name, aux, aw) = passes
            assert name == 'clean' and aux_name == 'auxiliary'
            assert source['input_ids'].shape == (4,640)
            assert aux['input_ids'].shape == (1,640)
            assert sw == {'source':1.} and aw == {'work':.25,'continuation':.125}
            assert int(aux['component_masks']['continuation'].sum()) == 16
            work = int(aux['component_masks']['work'].sum())
            if step % 3 == 0: assert work == 0
            elif step % 3 == 1: assert work == len(Tokenizer().encode('Answer:\n'))
            else: assert work > len(Tokenizer().encode('Answer:\n'))
    with pytest.raises(ValueError):
        driver.batch_at(rows, Tokenizer(), p, order, driver.SCAFFOLD_QUALIFICATION_ARM, 0, 0)
    with pytest.raises(ValueError):
        driver.scaffold_passes_at(rows, Tokenizer(), p, order, 'computation', 0, 0)


def science_protocol(arm='computation', seed=2026102029):
    return {**protocol(), 'purpose': driver.SCAFFOLD_STUDY_PURPOSE,
        'items':20480, 'steps':1280, 'save_every':640, 'seed':seed,
        'scaffold_arm':arm, 'view_schedule':'clean_plus_fixed_arm_auxiliary',
        'data_sha256':driver.SCAFFOLD_PARENT_SHA,
        'hardware_qualification_file':'missing-actual-qualification.json',
        'hardware_qualification_sha256':'b'*64,
        'prospective_design_file':str(driver.ROOT/'configs/real_cpt_scaffold_v29_design.json'),
        'prospective_design_sha256':driver.sha256_file(driver.ROOT/'configs/real_cpt_scaffold_v29_design.json')}


@pytest.mark.parametrize('arm', ['raw','format','computation'])
@pytest.mark.parametrize('seed', [2026102029,2026102030])
def test_fixed_science_identity_and_dispatch_preserve_the_actual_packer(arm,seed):
    p=science_protocol(arm,seed);driver.validate(p)
    rows=[example(document=f'c{i}') for i in range(32)]
    rows += [example('ordinary',document=f'o{i}') for i in range(96)]
    order=driver.real_cpt_scaffold.grouped_order(rows,seed=seed,passes=2)
    for step in (0,2,7):
        passes=driver.scaffold_passes_at(rows,Tokenizer(),p,order,arm,step,0)
        aux=passes[1][1];n=int(aux['component_masks']['work'].sum())
        assert n==0 if arm=='raw' else n==len(Tokenizer().encode('Answer:\n')) if arm=='format' else n>len(Tokenizer().encode('Answer:\n'))
    with pytest.raises(ValueError):
        driver.scaffold_passes_at(rows,Tokenizer(),p,order,'raw' if arm!='raw' else 'format',0,0)


@pytest.mark.parametrize('change', [dict(items=5120),dict(steps=320),dict(seed=1),
    dict(save_every=64),dict(scaffold_arm='reader'),dict(data_sha256='a'*64),
    dict(work_loss_weight=.1),dict(hardware_qualification_file=None),
    dict(prospective_design_sha256='not-a-sha'),dict(view_schedule='clean_plus_raw_format_computation_cycle')])
def test_science_rejects_unfrozen_training_changes(change):
    with pytest.raises(ValueError):driver.validate({**science_protocol(),**change})


def test_execution_release_requires_exact_job_source_design_and_recovery(tmp_path):
    p=science_protocol();path=tmp_path/'protocol.json';path.write_text(json.dumps(p))
    local=driver.ROOT.parents[1] if driver.ROOT.parent.name=='worktrees' else driver.ROOT/'local'
    output=local/'science-runs/real-cpt-scaffold-v29/cpt-computation-seed2026102029-v1'
    args=SimpleNamespace(execution_release=None,arm='computation',output=output,
        protocol=path,resume=None,halt_after=64)
    with pytest.raises(ValueError):driver.validate_scaffold_execution(p,args)
    release=tmp_path/'release.json';args.execution_release=release
    body=dict(status='RELEASED_V29_SCIENCE_STAGE64_ONLY',source_files=driver.source_files(),
        prospective_design_sha256=p['prospective_design_sha256'],
        jobs={output.name:dict(protocol_sha256=driver.sha256_file(path))})
    release.write_text(json.dumps(body));driver.validate_scaffold_execution(p,args)
    args.halt_after=None
    with pytest.raises(ValueError):driver.validate_scaffold_execution(p,args)
    args.resume=tmp_path/'step-0000064';args.resume.mkdir();(args.resume/'COMPLETE.json').write_text('{}')
    body['status']='RELEASED_V29_SCIENCE_TERMINAL1280'
    body['jobs'][output.name]['resume_COMPLETE_sha256']=driver.sha256_file(args.resume/'COMPLETE.json')
    release.write_text(json.dumps(body));driver.validate_scaffold_execution(p,args)
    body['source_files']={};release.write_text(json.dumps(body))
    with pytest.raises(ValueError):driver.validate_scaffold_execution(p,args)


def test_old_or_missing_mechanics_receipt_cannot_release_changed_source(tmp_path):
    p=science_protocol();path=tmp_path/'receipt.json'
    receipt=dict(status='PASS_BOUNDED_TRAIN_MECHANICS_RECOVERY_AND_TERMINAL_PROTECTION',
        source_files={},actual_updates=8,actual_restore_ranks=8,
        full_master_shards_audited=4960,native_vectors_recomputed=16)
    path.write_text(json.dumps(receipt))
    p.update(hardware_qualification_file=str(path),hardware_qualification_sha256=driver.sha256_file(path))
    with pytest.raises(ValueError):driver.verify_scaffold_science_release(p)
    receipt['source_files']=driver.source_files();path.write_text(json.dumps(receipt))
    p['hardware_qualification_sha256']=driver.sha256_file(path)
    # A synthetic receipt only tests the CPU contract; actual hardware evidence is separate.
    driver.verify_scaffold_science_release(p)
    path.write_text('{}')
    with pytest.raises(ValueError):driver.verify_scaffold_science_release(p)

"""One causal SFT engine for versioned partial-state and real-document packs.

Fresh released B0, one ZeRO reducer, periodic full checkpoints and full recovery.
The v7 full-trace CoT arm includes the root operation; the writer omits it.
"""
import argparse
from datetime import datetime, timezone
import fcntl
import json
import math
import os
from pathlib import Path
import random
import re
import shutil
import time

from weekend_thinking.artifacts import atomic_json, canonical_hash, require_utc_deadline, sha256_file, utc_now
from train.partial_state_packing import batch_packs, training_pack
from train import real_trace_packing, real_target_packing, real_source_balanced_packing, document_qa_packing, document_evidence_plan_packing
from train import permutation_program_packing
from train import real_cpt_scaffold
from train import uniform_lm

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = 'partial-state-sft-v7'
DEPTH_SCHEMA = 'variable-depth-writer-v11'
ARMS = ('direct', 'cot', 'cot_full', 'writer', 'reader')
REAL_SCHEMA = 'real-trace-cpt-v10'
TARGET_SCHEMA = 'real-target-sft-v15'
BALANCED_SCHEMA = 'real-source-balanced-v17'
DOCUMENT_SCHEMA = 'document-qa-v21'
EVIDENCE_SCHEMA = 'document-evidence-plan-v23'
EVIDENCE_WEIGHT_ID = 'document-evidence-weight-v25'
EVIDENCE_WEIGHT_PURPOSES = ('document_evidence_weight_qualification',
                          'document_evidence_weight_ablation')
EVIDENCE_FORMAT_ID = 'document-evidence-format-weight-v26'
EVIDENCE_FORMAT_PURPOSES = ('document_evidence_format_weight_qualification',
                          'document_evidence_format_weight_control')
PROGRAM_SCHEMA = 'permutation-program-sft-v28'
PROGRAM_QUALIFICATION_ARM = 'program_qualification'
PROGRAM_STUDY_PURPOSE = 'permutation_program_reference'
PROGRAM_DATA_SHA = '251a4af175442ec2a113bf9e80d06b1c82fa94e14f56cf98a3234fa8b77c5128'
PROGRAM_MANIFEST_SHA = '41ddc050ebbaf8abd3e0fb1af2b0c383875313bd6ff13d34b77d7a2040b5aa41'
SCAFFOLD_SCHEMA = 'real-cpt-scaffold-v29'
SCAFFOLD_QUALIFICATION_ARM = 'scaffold_qualification'
SCAFFOLD_STUDY_PURPOSE = 'real_cpt_scaffold_study'
SCAFFOLD_PARENT_SHA = 'ef15ff29e4c972ed6162f05d293bdc1059967e2914a0e9973e7b0f95f71bd509'
SCAFFOLD_MANIFEST_SHA = '77c8774d3fdeb283728878c44522dab541ec61a213d3f89ece471b3dcc45fb3f'
UNIFORM_SCHEMA = uniform_lm.SCHEMA


def source_files():
    names = ('train/partial_state_driver.py', 'train/partial_state_data.py',
             'train/partial_state_packing.py', 'train/useful_note_sft_data.py',
             'train/masked_causal_ce.py', 'train/neuron_backend.py',
             'train/neuron_driver.py', 'train/runtime_compat.py',
             'train/useful_note_precision.py', 'train/useful_note_recovery.py',
             'train/useful_note_sft_driver.py', 'weekend_thinking/pretrained.py',
             'weekend_thinking/checkpoints.py', 'weekend_thinking/core.py',
             'weekend_thinking/runtime_policy.py', 'weekend_thinking/artifacts.py',
             'train/evaluate.py', 'train/variable_depth_data.py', 'train/real_trace_packing.py',
             'train/calculation_boundaries.py', 'train/symbolic_note_labels.py',
             'train/scalar_completion.py', 'train/real_target_packing.py',
             'train/real_source_balanced_packing.py', 'train/document_evidence_data.py',
             'train/document_qa_data.py', 'train/document_qa_packing.py',
             'train/document_evidence_plan_data.py', 'train/document_evidence_plan_packing.py',
             'train/permutation_program_data.py', 'train/permutation_program_packing.py',
             'train/permutation_program_retention.py', 'train/real_cpt_scaffold.py',
             'configs/real_cpt_scaffold_v29_design.json', 'train/uniform_lm.py')
    return {name: sha256_file(ROOT / name) for name in names}


def validate(p):
    depth = p.get('schema') == DEPTH_SCHEMA
    real = p.get('schema') == REAL_SCHEMA
    target = p.get('schema') == TARGET_SCHEMA
    balanced = p.get('schema') == BALANCED_SCHEMA
    document = p.get('schema') == DOCUMENT_SCHEMA
    evidence = p.get('schema') == EVIDENCE_SCHEMA
    program = p.get('schema') == PROGRAM_SCHEMA
    scaffold = p.get('schema') == SCAFFOLD_SCHEMA
    uniform = p.get('schema') == UNIFORM_SCHEMA
    program_science = program and p.get('purpose') == PROGRAM_STUDY_PURPOSE
    scaffold_science = scaffold and p.get('purpose') == SCAFFOLD_STUDY_PURPOSE
    fixed = dict(schema=SCHEMA, model_id='Qwen/Qwen3-1.7B-Base',
                 world_size=4, micro_batch=8, global_batch=32, accumulation=1,
                 width=128, target_slots=64, schedule='constant',
                 parameter_dtype='bf16', optimizer_dtype='fp32', head_chunk=32)
    if depth:
        fixed.update(schema=DEPTH_SCHEMA, width=192, target_slots=128)
    if program:
        fixed.update(schema=PROGRAM_SCHEMA, width=256, target_slots=64)
    if real:
        fixed.update(schema=REAL_SCHEMA, world_size=8, micro_batch=4,
                     width=640, target_slots=640)
    if target:
        fixed.update(schema=TARGET_SCHEMA, world_size=8, micro_batch=4,
                     width=640, target_slots=128)
    if balanced:
        fixed.update(schema=BALANCED_SCHEMA, world_size=8, micro_batch=4,
                     width=640, target_slots=640)
    if scaffold:
        fixed.update(schema=SCAFFOLD_SCHEMA, world_size=8, micro_batch=4,
                     width=640, target_slots=640)
    if uniform:
        fixed.update(schema=UNIFORM_SCHEMA, world_size=8, micro_batch=4,
                     width=528, target_slots=527, accumulation=p.get('accumulation'),
                     global_batch=p.get('global_batch'), schedule=p.get('schedule'))
    if document or evidence:
        fixed.update(schema=EVIDENCE_SCHEMA if evidence else DOCUMENT_SCHEMA, world_size=8, micro_batch=1,
                     accumulation=4, width=3328, target_slots=512)
    if any(p.get(k) != v or isinstance(p.get(k), bool) for k, v in fixed.items()):
        raise ValueError('Require the explicit versioned shape/topology identity')
    if type(p.get('items')) is not int:
        raise ValueError('Explicit integer cohort size required')
    science=p.get('purpose')=='TRAIN5120_partial_state_study'
    real_science = real and p.get('purpose') == 'real_trace_cpt_study'
    target_science = target and p.get('purpose') == 'real_target_sft_study'
    balanced_science = balanced and p.get('purpose') == 'real_source_balanced_study'
    document_science = document and p.get('purpose') == 'document_qa_study'
    evidence_weight = evidence and p.get('purpose') in EVIDENCE_WEIGHT_PURPOSES
    evidence_format = evidence and p.get('purpose') in EVIDENCE_FORMAT_PURPOSES
    evidence_science = evidence and p.get('purpose') in (
        'document_evidence_plan_pilot', 'document_evidence_weight_ablation',
        'document_evidence_format_weight_control')
    if uniform:
        uniform_lm.validate_recipe(p)
        if (p.get('purpose') not in ('uniform_lm_qualification', 'uniform_lm_development',
                                    'uniform_lm_confirmation')
                or not isinstance(p.get('data_manifest_file'), str)
                or not re.fullmatch('[a-f0-9]{64}', p.get('data_manifest_sha256', ''))
                or not re.fullmatch('[a-f0-9]{64}', p.get('data_sha256', ''))):
            raise ValueError('Explicit uniform-LM purpose and frozen source manifest required')
    elif scaffold:
        if scaffold_science:
            if (p['items'] != 20480 or p.get('steps') != 1280
                    or p.get('save_every') != 640
                    or p.get('seed') not in (2026102029, 2026102030)
                    or p.get('scaffold_arm') not in real_cpt_scaffold.ARMS
                    or p.get('view_schedule') != 'clean_plus_fixed_arm_auxiliary'
                    or p.get('data_sha256') != SCAFFOLD_PARENT_SHA
                    or not isinstance(p.get('hardware_qualification_file'), str)
                    or not re.fullmatch('[a-f0-9]{64}', p.get('hardware_qualification_sha256', ''))
                    or not isinstance(p.get('prospective_design_file'), str)
                    or not re.fullmatch('[a-f0-9]{64}', p.get('prospective_design_sha256', ''))):
                raise ValueError('Frozen fixed-arm two-pass V29 study and actual source-qualified receipt required')
        elif (p.get('purpose') != 'real_cpt_scaffold_qualification'
                or p['items'] != 128 or p.get('steps') != 8 or p.get('save_every') != 4
                or p.get('seed') != 2026102029
                or p.get('view_schedule') != 'clean_plus_raw_format_computation_cycle'):
            raise ValueError('Only the bounded V29 qualification or frozen two-pass study is implemented')
        if (p.get('experiment_id') != 'real-cpt-scaffold-v29' or p.get('lr') != 1e-5
                or p.get('loss') != 'clean_LM_plus_fraction_weighted_auxiliary_continuation_work'
                or type(p.get('auxiliary_micro_batch')) is not int or p['auxiliary_micro_batch'] != 1
                or any(type(p.get(k)) not in (int, float) or p[k] != v for k, v in (
                    ('source_loss_weight', 1.), ('continuation_loss_weight', .5),
                    ('work_loss_weight', 1.), ('auxiliary_batch_fraction', .25)))
                or p.get('parent_data_sha256') != SCAFFOLD_PARENT_SHA
                or p.get('data_manifest_sha256') != SCAFFOLD_MANIFEST_SHA
                or not re.fullmatch('[a-f0-9]{64}', p.get('data_sha256', ''))
                or p.get('checkpoint_export_cleanup') is not True
                or p.get('checkpoint_memory_diagnostic') is not True):
            raise ValueError('Exact V29 source/auxiliary loss, data and checkpoint policy required')
    elif program:
        if program_science:
            from train.permutation_program_retention import POLICY
            if (p['items'] != 32768 or p.get('steps') != 4096
                    or p.get('save_every') != 1024 or p.get('seed') not in (2026102028,2026102029)
                    or p.get('lr') != 1e-5 or p.get('program_arm') not in permutation_program_packing.ARMS
                    or p.get('view_schedule') != 'interleaved_fixed_item_plus_epoch_parity'
                    or p.get('checkpoint_retention') != POLICY
                    or p.get('data_sha256') != PROGRAM_DATA_SHA
                    or p.get('data_manifest_sha256') != PROGRAM_MANIFEST_SHA):
                raise ValueError('Only the prospective two-seed program reference is implemented')
        elif (p.get('purpose') != 'permutation_program_qualification'
                or p['items'] != 128 or p.get('steps') != 8
                or p.get('save_every') != 4 or p.get('seed') != 2026102028
                or p.get('lr') != 1e-5
                or p.get('view_schedule') != 'qualification_all_four_views_each_update'):
            raise ValueError('Only the bounded all-view program qualification is implemented')
        if (p.get('experiment_id') != 'controlled-computation-v28'
                or p.get('loss') != 'separate_mean_answer_work_per_example'
                or any(type(p.get(k)) not in (int, float) or p[k] != 1.
                       for k in ('answer_loss_weight', 'work_loss_weight'))
                or not re.fullmatch('[a-f0-9]{64}', p.get('data_sha256', ''))
                or not re.fullmatch('[a-f0-9]{64}', p.get('data_manifest_sha256', ''))):
            raise ValueError('Exact separately normalized program fields and data bindings required')
    elif depth:
        hashes = {'fixed': '3f829b7e9fdb493a72f7cf740952130b29eb2dd5ded0c88d69b9a5730031a3ad',
                  'variable': '1b8a328479cd05718753b89e2f99e0f696e2cbb62079d505cc372fbb927111d3'}
        qualification = p.get('purpose') == 'variable_depth_qualification'
        pilot = p.get('purpose') == 'variable_depth_pilot'
        if (not (qualification or pilot) or p.get('curriculum') not in hashes
                or p.get('data_sha256') != hashes[p['curriculum']]
                or p.get('items') != (128 if qualification else 640)
                or p.get('steps') != (8 if qualification else 160)
                or p.get('save_every') != (2 if qualification else 80)
                or p.get('lr') != 1e-5 or p.get('seed') != 2026101911):
            raise ValueError('Require the declared bounded variable-depth intervention')
    elif evidence:
        qualification = p.get('purpose') in (
            'document_evidence_plan_qualification', 'document_evidence_weight_qualification',
            'document_evidence_format_weight_qualification')
        if evidence_weight and (p.get('experiment_id') != EVIDENCE_WEIGHT_ID
                or not isinstance(p.get('control_protocol_file'), str)
                or not re.fullmatch('[a-f0-9]{64}', p.get('control_protocol_sha256', ''))):
            raise ValueError('New coefficient ablation needs its own identity and frozen control')
        if evidence_format and (p.get('experiment_id') != EVIDENCE_FORMAT_ID
                or any(not isinstance(p.get(key+'_file'), str)
                       or not re.fullmatch('[a-f0-9]{64}', p.get(key+'_sha256', ''))
                       for key in ('control_protocol', 'selector_protocol'))):
            raise ValueError('Format control needs its own identity and both frozen comparisons')
        if (not (qualification or evidence_science)
                or p.get('items') != (128 if qualification else 2048)
                or p.get('steps') != (8 if qualification else 256)
                or p.get('save_every') != (2 if qualification else 128)
                or p.get('seed') not in (2026101923, 2026101924)
                or p.get('lr') != 1e-5
                or p.get('view_schedule') != 'interleaved_fixed_item_plus_epoch_parity'
                or p.get('loss') != 'separate_mean_source_answer_selection_per_example'
                or p.get('source_sample_labels') != 256
                or any(type(p.get(key)) not in (int, float) or p[key] != value for key, value in
                       [('source_loss_weight', .5), ('answer_loss_weight', .5),
                        ('selection_loss_weight', 1. if evidence_weight or evidence_format else .1)])
                or not re.fullmatch('[a-f0-9]{64}', p.get('data_sha256', ''))
                or not re.fullmatch('[a-f0-9]{64}', p.get('data_manifest_sha256', ''))):
            raise ValueError('Require explicit pointer/final-answer components and interleaved exposure')
        if evidence_science and (not isinstance(p.get('hardware_qualification_file'), str)
                or not re.fullmatch('[a-f0-9]{64}', p.get('hardware_qualification_sha256', ''))):
            raise ValueError('Evidence pilot needs actual new-objective long-context qualification')
    elif document:
        qualification = p.get('purpose') == 'document_qa_qualification'
        if (not (qualification or document_science)
                or p.get('items') != (128 if qualification else 2048)
                or p.get('steps') != (8 if qualification else 256)
                or p.get('save_every') != (2 if qualification else 128)
                or p.get('lr') != 1e-5
                or p.get('view_schedule') != 'plain_even_pass_augmented_odd_pass'
                or p.get('loss') != 'half_mean_fixed_source256_plus_half_mean_task_per_example'
                or p.get('source_loss_weight') != .5 or p.get('task_loss_weight') != .5
                or isinstance(p.get('source_loss_weight'), bool)
                or isinstance(p.get('task_loss_weight'), bool)
                or p.get('source_sample_labels') != 256
                or not re.fullmatch('[a-f0-9]{64}', p.get('data_sha256', ''))
                or not re.fullmatch('[a-f0-9]{64}', p.get('data_manifest_sha256', ''))):
            raise ValueError('Require the declared complete-document QA objective and exposure')
        if document_science and (p.get('seed') not in (2026101921, 2026101922)
                or not isinstance(p.get('hardware_qualification_file'), str)
                or not re.fullmatch('[a-f0-9]{64}', p.get('hardware_qualification_sha256', ''))):
            raise ValueError('Document science needs actual same-source long-context qualification')
    elif balanced:
        qualification = p.get('purpose') == 'real_source_balanced_qualification'
        if (not (qualification or balanced_science)
                or p.get('items') != (128 if qualification else 5120)
                or p.get('steps') != (8 if qualification else 320)
                or p.get('save_every') != (2 if qualification else 160)
                or p.get('lr') != 1e-5
                or p.get('data_sha256') != '437c64bc9ce4b451f4d5da07fccca9e108ee9add75b3c955bc54d2a2eca6b915'
                or p.get('view_schedule') != 'same_two_component_view_every_occurrence'
                or p.get('loss') != 'half_mean_source_prefix_plus_half_mean_task_per_example'
                or p.get('source_loss_weight') != 0.5
                or p.get('task_loss_weight') != 0.5
                or isinstance(p.get('source_loss_weight'), bool)
                or isinstance(p.get('task_loss_weight'), bool)):
            raise ValueError('Require declared separately normalized source/task objective')
        if balanced_science and (p.get('seed') not in (2026101913, 2026101914)
                or not isinstance(p.get('hardware_qualification_file'), str)
                or not re.fullmatch('[a-f0-9]{64}', p.get('hardware_qualification_sha256', ''))):
            raise ValueError('Source-balanced science needs actual same-source shape qualification')
    elif target:
        qualification = p.get('purpose') == 'real_target_sft_qualification'
        if (not (qualification or target_science)
                or p.get('items') != (128 if qualification else 5120)
                or p.get('steps') != (8 if qualification else 320)
                or p.get('save_every') != (2 if qualification else 160)
                or p.get('lr') != 1e-5
                or p.get('data_sha256') != '437c64bc9ce4b451f4d5da07fccca9e108ee9add75b3c955bc54d2a2eca6b915'
                or p.get('view_schedule') != 'same_target_view_every_occurrence'
                or p.get('loss') != 'single_mean_target_tokens_prefix_unscored'):
            raise ValueError('Require the declared answer-focused mechanics or two-epoch study')
        if target_science and (p.get('seed') not in (2026101913, 2026101914)
                or not isinstance(p.get('hardware_qualification_file'), str)
                or not re.fullmatch('[a-f0-9]{64}', p.get('hardware_qualification_sha256', ''))):
            raise ValueError('Answer-focused science needs declared seeds and actual shape qualification')
    elif real:
        qualification = p.get('purpose') == 'real_trace_qualification'
        if (not (qualification or real_science)
                or p.get('items') != (128 if qualification else 5120)
                or p.get('steps') != (8 if qualification else 320)
                or p.get('save_every') != (2 if qualification else 160)
                or p.get('lr') != 1e-5
                or p.get('data_sha256') != '437c64bc9ce4b451f4d5da07fccca9e108ee9add75b3c955bc54d2a2eca6b915'
                or p.get('view_schedule') != 'one_plain_one_augmented_per_two_passes'
                or p.get('loss') != 'single_mean_all_active_tokens_per_example'):
            raise ValueError('Require the declared real-data mechanics or two-epoch study')
        if real_science and (p.get('seed') not in (2026101913,2026101914)
                or not isinstance(p.get('hardware_qualification_file'),str)
                or not re.fullmatch('[a-f0-9]{64}',p.get('hardware_qualification_sha256',''))):
            raise ValueError('Real-data science needs declared seeds and a pinned hardware qualification')
    elif science:
        if (p.get('items')!=5120 or p.get('steps') not in (320,640)
                or p.get('save_every')!=160 or p.get('lr')!=1e-5
                or p.get('reader_view_schedule')!='one_teacher_one_plain_per_two_passes'
                or p.get('data_sha256')!='020fdf19892217492866b38f48a9a2655c30310fa792207b188477f28a2c93cb'
                or p.get('seed') not in (2026101806,2026101807)):
            raise ValueError('Exact fixed TRAIN5120 study, two declared seeds and320/640 updates required')
    elif p.get('purpose') not in ('qualification', 'TRAIN_overfit_only') or p.get('items')!=128:
        raise ValueError('Explicit fitting or fixed TRAIN5120 study required')
    if (type(p.get('steps')) is not int or not 3 <= p['steps'] <= (8192 if uniform else 4096 if program_science else 1280 if scaffold_science else 640 if science else 320 if real_science or target_science or balanced_science else 256 if document_science or evidence_science else 160 if depth else 128)
            or type(p.get('seed')) is not int or not 0 <= p['seed'] < 2**63
            or type(p.get('save_every')) is not int or not 1 <= p['save_every'] <= (p['steps'] if uniform else 1024 if program_science else 640 if scaffold_science else 160 if science or real_science or target_science or balanced_science else 128 if document_science or evidence_science else 80 if depth else 32)
            or not 0 < p.get('lr', 0) <= 1e-4 or not math.isfinite(p['lr'])):
        raise ValueError('Finite mechanics steps, LR, seed and periodic full saves required')
    if p['source_files'] != source_files():
        raise ValueError('Frozen short-pack implementation changed')
    cleanup = p.get('checkpoint_export_cleanup', False)
    observe = p.get('checkpoint_memory_diagnostic', False)
    if type(cleanup) is not bool or type(observe) is not bool or science and (cleanup or observe):
        raise ValueError('CPU export cleanup remains a separately qualified engineering option')


def validate_program_execution(p, args):
    """A prepared protocol alone never launches a long learning job."""
    from train.permutation_program_retention import RUN_ROOT
    if (args.execution_release is None or args.arm != p['program_arm']
            or args.output != RUN_ROOT/f'program-{args.arm}-seed{p["seed"]}-v1'):
        raise ValueError('Explicit scoped program execution release required')
    release = json.loads(args.execution_release.read_text())
    job = release.get('jobs', {}).get(args.output.name)
    if (job is None or job.get('protocol_sha256') != sha256_file(args.protocol)
            or release.get('source_files') != source_files()
            or release.get('prospective_design_sha256') != p['prospective_design_sha256']):
        raise ValueError('Program job/source/design differs from its execution release')
    if release.get('status') == 'RELEASED_V28_SCIENCE_STAGE64_ONLY':
        if args.resume is not None or args.halt_after != 64:
            raise ValueError('The initial release permits only fresh first64 updates')
    elif release.get('status') == 'RELEASED_V28_SCIENCE_TERMINAL4096':
        if (args.resume is None or args.halt_after is not None
                or job.get('resume_COMPLETE_sha256') != sha256_file(args.resume/'COMPLETE.json')):
            raise ValueError('Audited planned/recovery checkpoint required for continuation')
    else:
        raise ValueError('A prepared study is not a released learning job')
    checks = (('hardware_qualification_file','hardware_qualification_sha256',
               'PASS_ACTUAL_PROGRAM_ALL_VIEW_FULL_RESTORE'),
              ('decoder_qualification_file','decoder_qualification_sha256',
               'PASS_ACTUAL_TRAIN_PROGRAM_DECODER_NATIVE_CACHE_RAW_TOKENS_AND_CPU_GREEDY'))
    for file_key, hash_key, status in checks:
        path = Path(p[file_key]); evidence = json.loads(path.read_text())
        if sha256_file(path) != p[hash_key] or evidence.get('status') != status:
            raise ValueError('Actual independently checked program execution evidence required')
        if file_key == 'hardware_qualification_file':
            excluded = {'train/partial_state_driver.py','train/permutation_program_retention.py'}
            current = {k:v for k,v in p['source_files'].items() if k not in excluded}
            previous = {k:v for k,v in evidence['source_files'].items() if k not in excluded}
            shape = {k:p[k] for k in ('world_size','micro_batch','global_batch','width','target_slots','head_chunk','accumulation')}
            if current != previous or evidence['qualified_training_shape'] != shape:
                raise ValueError('Numerical/packing source or actual qualified shape changed')
    design = Path(p['prospective_design_file'])
    if sha256_file(design) != p['prospective_design_sha256']:
        raise ValueError('Frozen study design changed')


def validate_evidence_weight_control(p, control):
    """The new trial changes one loss coefficient, using an existing control.

    Qualification uses a shorter prefix, but the full study must preserve the
    original data, exposures, order and numerical path. Device allocation and
    new execution/qualification receipts belong to the new attempt.
    """
    if p.get('purpose') not in EVIDENCE_WEIGHT_PURPOSES:
        raise ValueError('Explicit coefficient-ablation identity required')
    if (control.get('schema') != EVIDENCE_SCHEMA
            or control.get('purpose') != 'document_evidence_plan_pilot'
            or control.get('selection_loss_weight') != .1):
        raise ValueError('Original completed 0.1-selection protocol required')
    fields = ('model_id', 'model_dir', 'base_protocol', 'base_protocol_sha256',
              'base_signature', 'data_file', 'data_sha256', 'data_manifest_file',
              'data_manifest_sha256', 'seed', 'lr', 'world_size', 'micro_batch',
              'global_batch', 'accumulation', 'width', 'target_slots', 'head_chunk',
              'schedule', 'parameter_dtype', 'optimizer_dtype', 'view_schedule',
              'loss', 'source_sample_labels', 'source_loss_weight',
              'answer_loss_weight', 'runtime_policy', 'precision_contract',
              'checkpoint_export_cleanup', 'checkpoint_memory_diagnostic')
    if p['purpose'] == 'document_evidence_weight_ablation':
        fields += ('items', 'steps', 'save_every')
    if any(p.get(key) != control.get(key) for key in fields):
        raise ValueError('Coefficient comparison changed data, exposure or another training setting')
    # The executable change is confined to this driver's versioned validation.
    # Existing packing, loss kernels and optimizer implementation stay identical.
    expected = {k: v for k, v in control['source_files'].items()
                if k != 'train/partial_state_driver.py'}
    if expected != {k: v for k, v in p['source_files'].items()
                    if k != 'train/partial_state_driver.py'}:
        raise ValueError('Coefficient comparison changed an existing numerical/packing source')


def validate_evidence_format_control(p, control, selector):
    """Match the selected-field coefficient while keeping the format-only arm.

    The original direct protocol binds data/exposure; the completed selector
    binds the new coefficient. Neither comparison may change the numerical path.
    """
    if p.get('purpose') not in EVIDENCE_FORMAT_PURPOSES:
        raise ValueError('Explicit format-weight control identity required')
    if (control.get('schema') != EVIDENCE_SCHEMA
            or control.get('purpose') != 'document_evidence_plan_pilot'
            or control.get('selection_loss_weight') != .1
            or selector.get('schema') != EVIDENCE_SCHEMA
            or selector.get('purpose') != 'document_evidence_weight_ablation'
            or selector.get('experiment_id') != EVIDENCE_WEIGHT_ID
            or selector.get('selection_loss_weight') != 1.):
        raise ValueError('Original0.1 direct and completed1.0 selector protocols required')
    fields = ('model_id', 'model_dir', 'base_protocol', 'base_protocol_sha256',
              'base_signature', 'data_file', 'data_sha256', 'data_manifest_file',
              'data_manifest_sha256', 'seed', 'lr', 'world_size', 'micro_batch',
              'global_batch', 'accumulation', 'width', 'target_slots', 'head_chunk',
              'schedule', 'parameter_dtype', 'optimizer_dtype', 'view_schedule',
              'loss', 'source_sample_labels', 'source_loss_weight',
              'answer_loss_weight', 'runtime_policy', 'precision_contract',
              'checkpoint_export_cleanup', 'checkpoint_memory_diagnostic')
    if p['purpose'] == 'document_evidence_format_weight_control':
        fields += ('items', 'steps', 'save_every')
    if any(p.get(key) != reference.get(key)
           for reference in (control, selector) for key in fields):
        raise ValueError('Format comparison changed data, exposure or another training setting')
    numerical = {k: v for k, v in p['source_files'].items()
                 if k != 'train/partial_state_driver.py'}
    if any(numerical != {k: v for k, v in reference['source_files'].items()
                        if k != 'train/partial_state_driver.py'}
           for reference in (control, selector)):
        raise ValueError('Format comparison changed an existing numerical/packing source')


def order_for(p, rows=None):
    if p.get('schema') == SCAFFOLD_SCHEMA:
        if rows is None:
            rows = [json.loads(line) for line in Path(p['data_file']).read_text().splitlines()]
        if len(rows) != p['items'] or p['steps'] * p['global_batch'] % p['items']:
            raise ValueError('Complete frozen source passes required')
        return real_cpt_scaffold.grouped_order(rows, seed=p['seed'],
            passes=p['steps'] * p['global_batch'] // p['items'])
    order = []
    for epoch in range((p['steps'] * p['global_batch'] + p['items'] - 1) // p['items']):
        indices = list(range(p['items']))
        random.Random(p['seed'] + epoch).shuffle(indices)
        order.extend(indices)
    return order[:p['steps'] * p['global_batch']]


def role_for(arm, position, items, item_index):
    if arm not in ARMS:
        raise ValueError('Unknown short-pack arm')
    if arm != 'reader':
        return arm
    if items==5120:
        # Two-pass study: every expression gets one teacher and one plain view.
        # Interleave roles within each epoch, preserving50/25/25 aggregate mix.
        if (item_index+position//items)%2==0:return 'reader_teacher'
        return 'reader_empty' if (item_index//2)%2==0 else 'direct'
    # Over four epochs, every expression receives teacher/teacher/empty/direct.
    return ('reader_teacher', 'reader_teacher', 'reader_empty', 'direct')[(item_index + position // items) % 4]


def batch_at(rows, tokenizer, p, order, arm, step, rank, micro=0):
    if (not 0 <= step < p['steps'] or not 0 <= rank < p['world_size']
            or not 0 <= micro < p['accumulation']):
        raise ValueError('Actual bounded step/rank required')
    if p['schema'] == UNIFORM_SCHEMA:
        if arm != uniform_lm.ARM:
            raise ValueError('Uniform training has exactly one unaugmented arm')
        return uniform_lm.batch_at(rows, tokenizer, p, order, step, rank, micro)
    if p['schema'] == SCAFFOLD_SCHEMA:
        raise ValueError('V29 requires both independent forwards; use scaffold_passes_at')
    if p['schema'] == PROGRAM_SCHEMA:
        qualification = p['purpose'] == 'permutation_program_qualification'
        if (qualification and arm != PROGRAM_QUALIFICATION_ARM
                or not qualification and (p['purpose'] != PROGRAM_STUDY_PURPOSE or arm != p['program_arm'])):
            raise ValueError('Declared program qualification or frozen reference arm required')
        start = step*p['global_batch'] + rank*p['micro_batch']
        return batch_packs([permutation_program_packing.training_pack(
            rows[order[pos]], tokenizer,
            ('direct', 'empty', 'trace', 'pause')[pos % 4] if qualification else
                permutation_program_packing.view_for(arm, pos//p['items'], order[pos]),
            width=p['width'], target_slots=p['target_slots'])
            for pos in range(start, start+p['micro_batch'])])
    if p['schema'] in (DOCUMENT_SCHEMA, EVIDENCE_SCHEMA):
        start = (step*p['global_batch'] + micro*p['world_size']*p['micro_batch']
                 + rank*p['micro_batch'])
        packer = document_evidence_plan_packing if p['schema'] == EVIDENCE_SCHEMA else document_qa_packing
        def view(position):
            if p['schema'] == EVIDENCE_SCHEMA:
                return packer.view_for(arm, position//p['items'], order[position])
            return packer.view_for(arm, position//p['items'])
        return batch_packs([packer.training_pack(rows[order[pos]], tokenizer, view(pos),
            width=p['width'], target_slots=p['target_slots'])
            for pos in range(start, start+p['micro_batch'])])
    start = step * p['global_batch'] + rank * p['micro_batch']
    if p['schema'] == BALANCED_SCHEMA:
        packs = [real_source_balanced_packing.training_pack(rows[order[pos]], tokenizer, arm,
                     width=p['width'], target_slots=p['target_slots'])
                 for pos in range(start, start + p['micro_batch'])]
        return batch_packs(packs)
    if p['schema'] == TARGET_SCHEMA:
        packs = [real_target_packing.training_pack(rows[order[pos]], tokenizer, arm,
                     width=p['width'], target_slots=p['target_slots'])
                 for pos in range(start, start + p['micro_batch'])]
        return batch_packs(packs)
    if p['schema'] == REAL_SCHEMA:
        packs = [real_trace_packing.training_pack(rows[order[pos]], tokenizer,
                     real_trace_packing.view_for(arm, pos, p['items'], order[pos]),
                     width=p['width'], target_slots=p['target_slots'])
                 for pos in range(start, start + p['micro_batch'])]
        return batch_packs(packs)
    packs = [training_pack(rows[order[pos]], tokenizer, role_for(arm, pos, p['items'], order[pos]),
                           width=p['width'], target_slots=p['target_slots'])
             for pos in range(start, start + p['micro_batch'])]
    return batch_packs(packs)


def validate_scaffold_execution(p, args):
    """Scope the prepared protocol to one explicitly reviewed learning stage."""
    # ROOT can be an isolated worktree or the main repository checkout.
    local = ROOT.parents[1] if ROOT.parent.name == 'worktrees' else ROOT/'local'
    run_root = local/'science-runs/real-cpt-scaffold-v29'
    if (args.execution_release is None or args.arm != p['scaffold_arm']
            or args.output != run_root/f'cpt-{args.arm}-seed{p["seed"]}-v1'):
        raise ValueError('Explicit scoped V29 execution release and output required')
    release = json.loads(args.execution_release.read_text())
    job = release.get('jobs', {}).get(args.output.name)
    if (job is None or job.get('protocol_sha256') != sha256_file(args.protocol)
            or release.get('source_files') != source_files()
            or release.get('prospective_design_sha256') != p['prospective_design_sha256']):
        raise ValueError('V29 job, sources or design differ from the execution release')
    if release.get('status') == 'RELEASED_V29_SCIENCE_STAGE64_ONLY':
        if args.resume is not None or args.halt_after != 64:
            raise ValueError('Initial V29 release permits only fresh first64 updates')
    elif release.get('status') == 'RELEASED_V29_SCIENCE_TERMINAL1280':
        if (args.resume is None or args.halt_after is not None
                or job.get('resume_COMPLETE_sha256') != sha256_file(args.resume/'COMPLETE.json')):
            raise ValueError('Audited original-run full checkpoint required for continuation')
    else:
        raise ValueError('Prepared V29 data and CPU tests are not an execution release')


def verify_scaffold_science_release(p):
    """Require actual recovery qualification of these exact numerical sources.

    This is an execution preflight, not an authorization inferred from a passing
    CPU test. The outer supervisor separately owns resource/storage release.
    """
    if p['purpose'] != SCAFFOLD_STUDY_PURPOSE:
        raise ValueError('Only a frozen V29 study uses this release')
    receipt_path = Path(p['hardware_qualification_file'])
    design_path = Path(p['prospective_design_file'])
    if (sha256_file(receipt_path) != p['hardware_qualification_sha256']
            or sha256_file(design_path) != p['prospective_design_sha256']
            or design_path.resolve() != (ROOT/'configs/real_cpt_scaffold_v29_design.json').resolve()):
        raise ValueError('Exact hardware receipt and prospective design required')
    receipt, design = (json.loads(path.read_text()) for path in (receipt_path, design_path))
    if (receipt.get('status') != 'PASS_BOUNDED_TRAIN_MECHANICS_RECOVERY_AND_TERMINAL_PROTECTION'
            or receipt.get('source_files') != p['source_files']
            or receipt.get('actual_updates') != 8 or receipt.get('actual_restore_ranks') != 8
            or receipt.get('full_master_shards_audited') != 4960
            or receipt.get('native_vectors_recomputed') != 16
            or design.get('schema') != 'real-cpt-scaffold-v29-prospective-science-design-v1'
            or design.get('experiment_id') != p['experiment_id']
            or design.get('data_manifest_sha256') != p['data_manifest_sha256']
            or p['scaffold_arm'] not in design['arms'] or p['seed'] not in design['data_order_seeds']):
        raise ValueError('Actual source-qualified recovery and unchanged scientific identity required')


def scaffold_passes_at(rows, tokenizer, p, order, arm, step, rank):
    if p['schema'] != SCAFFOLD_SCHEMA or not 0 <= step < p['steps']:
        raise ValueError('Declared complete V29 source step required')
    if p['purpose'] == 'real_cpt_scaffold_qualification' and arm == SCAFFOLD_QUALIFICATION_ARM:
        actual_arm = real_cpt_scaffold.ARMS[step % len(real_cpt_scaffold.ARMS)]
    elif (p['purpose'] == SCAFFOLD_STUDY_PURPOSE and arm in real_cpt_scaffold.ARMS
          and arm == p.get('scaffold_arm')):
        actual_arm = arm
    else:
        raise ValueError('Explicit matching qualification or fixed science arm required')
    return real_cpt_scaffold.batches_at(rows, tokenizer, order, arm=actual_arm,
        step=step, rank=rank, width=p['width'], target_slots=p['target_slots'])


def checkpoint_identity(manifest, weights, saved, p, arm, run_id, rank, order_hash, protocol_sha):
    """Check lineage before supported loading and full actual byte verification."""
    step = manifest['step']
    expected = dict(spec_hash=canonical_hash(p), arm=arm, seed=p['seed'],
                    order_hash=order_hash, protocol_sha256=protocol_sha,
                    pipeline_schema=p['schema'], data_sha256=p['data_sha256'])
    if (manifest['run_id'] != run_id or manifest['world_size'] != p['world_size']
            or not 1 <= step <= p['steps'] or manifest['spec_hash'] != expected['spec_hash']
            or any(manifest['metadata'].get(k) != v or saved.get(k) != v for k, v in expected.items())
            or any(weights.get(k) != v for k, v in dict(step=step, run_id=run_id, spec_hash=expected['spec_hash']).items())
            or saved.get('rank') != rank or saved.get('world_size') != p['world_size']
            or saved.get('step') != step or saved.get('counters') != manifest['metadata']['counters']
            or saved['counters']['expressions'] != step * p['global_batch']):
        raise ValueError('Full short-pack checkpoint lineage/cursor differs')
    return step


def worker(_index, args, p, rows, tok, assets, refs, native, order, protocol_sha):
    import numpy as np
    import torch
    import torch.distributed as dist
    import torch_xla
    import torch_xla.core.xla_model as xm
    import torch_xla.runtime as xr
    import torch_xla.distributed.xla_backend  # noqa: F401
    from torch_xla.amp.syncfree import AdamW
    from torch_xla.distributed.zero_redundancy_optimizer import ZeroRedundancyOptimizer
    from train.masked_causal_ce import masked_causal_ce, masked_component_causal_ce
    from train.neuron_backend import NeuronCausalBackend
    from train.neuron_driver import restore_rng_after_sync, save_checkpoint
    from train.runtime_compat import configure_runtime
    from train.useful_note_precision import cast_parameters, master_samples, active_shard_samples, shard_expectations
    from train.useful_note_sft_driver import inspect_optimizer, native_gate_consensus, publish_native_gate, recovery_consensus, replicas_bitwise_equal
    from weekend_thinking.pretrained import load_base, sampled_fingerprint
    from weekend_thinking.runtime_policy import require_runtime_policy, runtime_policy

    configure_runtime()
    if not hasattr(torch, 'xla'):
        torch._register_device_module('xla', torch_xla)
    require_runtime_policy(p['runtime_policy'], runtime_policy())
    device = torch_xla.device(); rank, world = xr.global_ordinal(), xr.world_size()
    if world != p['world_size'] or os.environ.get('NEURON_RT_CONTIGUOUS_SCRATCHPAD') != '0':
        raise ValueError('Actual world/allocator differs')
    dist.init_process_group('xla', init_method='xla://')
    random.seed(p['seed'] + rank); np.random.seed((p['seed'] + rank) % 2**32); torch.manual_seed(p['seed'])
    model, _ = load_base(p['model_dir'])
    if sampled_fingerprint(model) != p['base_signature']:
        raise ValueError('Fresh actual released B0 required')
    conversion = cast_parameters(model); expected = shard_expectations(model, rank, world)
    model.to(device); model.tie_weights()
    if any(t.dtype != torch.bfloat16 for t in model.parameters()) or sampled_fingerprint(model) != p['base_signature']:
        raise ValueError('Actual released tied BF16 parameters required')
    for name, value in model.named_buffers():
        if value.is_floating_point() and (value.dtype != torch.float32 or canonical_hash(value.detach().cpu().reshape(-1).tolist()) != conversion['buffers'][name]['sha256']):
            raise ValueError('Original FP32 rotary buffer changed')
    sync = lambda: torch_xla.sync(wait=True)
    backend = NeuronCausalBackend(model, sync=sync, head_chunk=p['head_chunk'])
    evidence = args.output if args.resume is None else args.recovery_dir / 'initialization'
    if rank == 0:
        evidence.mkdir(parents=True, exist_ok=False)
    xm.rendezvous('partial-native-directory')
    with torch.no_grad():
        reference_index = rank % len(refs)
        context = refs[reference_index]['context_ids']; x = torch.full((1, 1024), 151643, dtype=torch.long)
        x[0, :len(context)] = torch.tensor(context)
        h = backend.hidden(x.to(device)); z = model.lm_head(h[:, len(context)-1]); lp = z.float().log_softmax(-1); sync()
        gate = publish_native_gate(evidence, rank, reference_index, lp.cpu(), native[reference_index:reference_index+1], str(h.dtype), str(z.dtype), p, protocol_sha, raise_on_failure=False)
        del x, h, z, lp
    native_gate_consensus(evidence, rank, world, gate, xm, device)
    optimizer = ZeroRedundancyOptimizer(model.parameters(), AdamW, optimizer_dtype=torch.float32,
        grad_clipping=True, max_norm=1., save_master_weights=True, lr=p['lr'], betas=(.9, .95),
        eps=1e-8, weight_decay=.1, use_grad_acc_hook=False, higher_cc_precision=False)
    reduction_calls = [0]
    if p['schema'] == UNIFORM_SCHEMA:
        original_reduce = optimizer._reduce_gradients
        def counted_reduce(*a, **kw):
            reduction_calls[0] += 1
            return original_reduce(*a, **kw)
        optimizer._reduce_gradients = counted_reduce
    initial = master_samples(optimizer, expected)
    if not torch.equal(initial.view(torch.uint8), expected['values'].contiguous().view(torch.uint8)):
        raise RuntimeError('FP32 masters differ from released CPU shard samples')
    atomic_json(evidence/f'initial-master-rank-{rank:05d}.json', dict(rank=rank, exact_released_FP32_samples=True))
    torch.save(dict(expected_cpu=expected['values'], actual_fp32_masters=initial), evidence/f'initial-master-rank-{rank:05d}.pt')
    found_inf = torch.zeros((), device=device); order_hash = canonical_hash(order)
    state = dict(pipeline_schema=p['schema'], spec_hash=canonical_hash(p), arm=args.arm, seed=p['seed'],
                 order_hash=order_hash, protocol_sha256=protocol_sha, data_sha256=p['data_sha256'])
    deadline = require_utc_deadline(args.deadline_utc)
    def stopped():
        return bool(xm.all_reduce(xm.REDUCE_MAX, torch.tensor(float(args.stop_file.exists() or datetime.now(timezone.utc) >= deadline), device=device)).cpu())
    counters = dict(expressions=0, supervised_tokens=0); begin = 0
    if args.resume is not None:
        error = None
        try:
            weights = torch.load(args.resume/'model.pt', map_location='cpu', weights_only=True)
            saved = torch.load(args.resume/f'rank-{rank:05d}.pt', map_location='cpu', weights_only=True)
            begin = checkpoint_identity(args.resume_manifest, weights, saved, p, args.arm, args.output.name, rank, order_hash, protocol_sha)
        except Exception as e:
            error = f'{type(e).__name__}: {e}'
        recovery_consensus(args.recovery_dir, rank, world, error, xm, device, 'payload')
        error = None
        try:
            from train.useful_note_recovery import restore_and_verify
            report = restore_and_verify(model, optimizer, weights, saved, xm=xm, device=device, sync=torch_xla.sync)
            inspect_optimizer(optimizer, begin, len(list(model.parameters())))
        except Exception as e:
            error = f'{type(e).__name__}: {e}'
        recovery_consensus(args.recovery_dir, rank, world, error, xm, device, 'restored')
        atomic_json(args.recovery_dir/f'restored-rank-{rank:05d}.json', report)
        counters = saved['counters']; run_updates = not stopped()
        restore_rng_after_sync(saved['rng'], xm, device, torch_xla.sync)
        del saved, weights
    elif rank == 0:
        atomic_json(args.output/'run.json', dict(state, purpose=p['purpose'], planned_steps=p['steps'],
            assets=assets, model='Qwen/Qwen3-1.7B-Base', global_batch=p['global_batch'],
            micro_batch=p['micro_batch'], started_utc=utc_now()))
    if args.resume is None:
        run_updates = not stopped()
    last = begin; checkpoint_walls = []; loop_started = time.monotonic()
    for step in (range(begin, p['steps']) if run_updates else ()):
        reductions_before = reduction_calls[0]
        rate = (uniform_lm.set_learning_rate(optimizer, p, step)
                if p['schema'] == UNIFORM_SCHEMA else p['lr'])
        started = time.monotonic(); optimizer.zero_grad(set_to_none=True)
        audited = (step < 3 or (step+1) % p['save_every'] == 0 or step+1 == p['steps']
                   or p.get('purpose') in (PROGRAM_STUDY_PURPOSE, SCAFFOLD_STUDY_PURPOSE)
                   and step+1 == args.halt_after)
        before = master_samples(optimizer, expected) if audited else None
        component_objective = p['schema'] in (BALANCED_SCHEMA, DOCUMENT_SCHEMA)
        scaffold_objective = p['schema'] == SCAFFOLD_SCHEMA
        pointer_objective = p['schema'] in (EVIDENCE_SCHEMA, PROGRAM_SCHEMA) or scaffold_objective
        field_names = (('source', 'work', 'continuation') if scaffold_objective else
                       ('answer', 'work') if p['schema'] == PROGRAM_SCHEMA
                       else document_evidence_plan_packing.COMPONENTS)
        weights = (dict(note_weight=p['source_loss_weight'], future_weight=p['task_loss_weight'])
                   if component_objective else {})
        loss = 0.; labels = 0; component_values = [0.] * (2*len(field_names) if pointer_objective else 4)
        passes = (scaffold_passes_at(rows, tok, p, order, args.arm, step, rank)
                  if scaffold_objective else ((None, batch_at(rows, tok, p, order,
                      args.arm, step, rank, micro), None) for micro in range(p['accumulation'])))
        for branch, batch, branch_weights in passes:
            if pointer_objective:
                names = tuple(batch['component_masks']) if scaffold_objective else field_names
                masks = (batch['component_masks'] if scaffold_objective
                         else {name: batch[name+'_mask'] for name in names})
                ce = masked_component_causal_ce(backend, batch['input_ids'],
                    batch['target_positions'], batch['labels'], masks,
                    branch_weights if scaffold_objective else
                    {name: p[name+'_loss_weight'] for name in names})
            else:
                ce = masked_causal_ce(backend, **batch, **weights)
            (ce.loss/p['accumulation']).backward(); sync()
            loss += float(ce.loss.detach().cpu())/p['accumulation']
            if pointer_objective:
                labels += sum(int(mask.sum()) for mask in masks.values())
                for name in names:
                    index = field_names.index(name)
                    component_values[index] += float(ce.losses[name].detach().cpu())/p['accumulation']
                    component_values[index+len(field_names)] += int(masks[name].sum())
            else:
                labels += int((batch['note_close_mask'] | batch['future_mask']).sum())
            if component_objective:
                component_values[0] += float(ce.note_close_loss.detach().cpu())/p['accumulation']
                component_values[1] += float(ce.future_loss.detach().cpu())/p['accumulation']
                component_values[2] += int(batch['note_close_mask'].sum())
                component_values[3] += int(batch['future_mask'].sum())
            del ce, batch
        if not math.isfinite(loss) or any(t.grad is not None and t.grad.dtype != torch.bfloat16 for t in model.parameters()):
            raise RuntimeError('Nonfinite loss or wrong gradient storage')
        finite = torch.stack([torch.isfinite(t.grad).all() for t in model.parameters() if t.grad is not None]).all()
        if bool(xm.all_reduce(xm.REDUCE_MAX, (~finite).float()).cpu()):
            raise RuntimeError('Nonfinite actual gradient')
        optimizer.step(found_inf=found_inf); sync(); norm = float(optimizer.grad_norm.cpu())
        actual_optimizer_lrs = None
        if p['schema'] == UNIFORM_SCHEMA:
            actual_optimizer_lrs = [float(g['lr']) for g in optimizer.base_optimizer.param_groups]
            if (reduction_calls[0]-reductions_before != 1
                    or not actual_optimizer_lrs
                    or any(abs(v-rate) > 1e-15 for v in actual_optimizer_lrs)):
                raise RuntimeError('Actual optimizer LR or single reduction differs')
        if not math.isfinite(norm) or norm <= 0:
            raise RuntimeError('Invalid global clipped norm')
        if audited:
            after = master_samples(optimizer, expected); active = active_shard_samples(model, expected)
            if not torch.isfinite(after).all() or not (after-before).ne(0).any() or not torch.equal(active.view(torch.uint8), after.bfloat16().view(torch.uint8)):
                raise RuntimeError('Actual FP32 master update/cast check failed')
            # Shard samples differ by rank; compare replicated MODEL samples.
            sampled = torch.cat([v.detach().reshape(-1)[:32] for v in model.parameters()]).cpu()
            replicas = xm.all_gather(sampled.to(device)).reshape(world, -1).cpu()
            if not replicas_bitwise_equal(replicas):
                raise RuntimeError('Replicated BF16 model samples disagree')
        if step == 2:
            inspection, _ = inspect_optimizer(optimizer, step+1, len(list(model.parameters())))
            atomic_json(evidence/f'optimizer-after3-rank-{rank:05d}.json', inspection)
        total_loss, total_labels = xm.all_reduce(xm.REDUCE_SUM, torch.tensor([loss, float(labels)], device=device)).cpu().tolist()
        components = None
        if pointer_objective:
            values = xm.all_reduce(xm.REDUCE_SUM,
                torch.tensor(component_values, device=device)).cpu().tolist()
            components = {}
            for index, name in enumerate(field_names):
                components[name+'_mean_loss'] = values[index]/world
                components[name+'_labels'] = int(values[index+len(field_names)])
                components[name+'_weight'] = p[name+'_loss_weight']
                if scaffold_objective:
                    components[name+'_batch_fraction'] = 1. if name == 'source' else p['auxiliary_batch_fraction']
        elif component_objective:
            source_loss, task_loss, source_labels, task_labels = xm.all_reduce(
                xm.REDUCE_SUM, torch.tensor(component_values, device=device)).cpu().tolist()
            components = dict(source_mean_loss=source_loss/world, task_mean_loss=task_loss/world,
                              source_labels=int(source_labels), task_labels=int(task_labels),
                              source_weight=p['source_loss_weight'], task_weight=p['task_loss_weight'])
        counters['expressions'] += p['global_batch']; counters['supervised_tokens'] += int(total_labels); last = step+1
        stop = stopped()
        halt = args.halt_after is not None and last == args.halt_after
        if rank == 0:
            metric = dict(step=last, time_utc=utc_now(), elapsed_seconds=time.monotonic()-started,
                          loss=total_loss/world, gradient_norm=norm, actual_master_and_replica_audit=audited,
                          learning_rate=rate, counters=dict(counters),
                          ordered_item_ids=[rows[k]['item_id'] for k in
                              order[step*p['global_batch']:(step+1)*p['global_batch']]])
            if components is not None:
                metric['loss_components'] = components
            if actual_optimizer_lrs is not None:
                metric['actual_base_optimizer_lrs'] = actual_optimizer_lrs
                metric['gradient_reductions'] = reduction_calls[0]-reductions_before
            with (args.output/'metrics.jsonl').open('a') as f:
                f.write(json.dumps(metric, allow_nan=False)+'\n'); f.flush(); os.fsync(f.fileno())
        if (last==64 and p['schema'] not in (REAL_SCHEMA, TARGET_SCHEMA, BALANCED_SCHEMA, DOCUMENT_SCHEMA, EVIDENCE_SCHEMA, UNIFORM_SCHEMA)) or last % p['save_every'] == 0 or last == p['steps'] or stop or halt:
            program_study = p.get('purpose') == PROGRAM_STUDY_PURPOSE
            lease = None
            if program_study:
                from train.permutation_program_retention import export_lease, rotate, publish_phase, await_phase
                if rank == 0:
                    lease = export_lease(deadline); lease.__enter__()
                    publish_phase(evidence, 'lease-'+str(last), p)
                else:
                    await_phase(evidence, 'lease-'+str(last), p, deadline)
            try:
                saving = time.monotonic()
                # Full state is approximately14 bytes/parameter; reserve16 plus
                # the existing35GiB floor before any rank starts writing.
                scarce = float(shutil.disk_usage(args.output).free < 35*2**30 + 16*p['base_signature']['parameters'])
                if bool(xm.all_reduce(xm.REDUCE_MAX, torch.tensor(scarce, device=device)).cpu()):
                    raise RuntimeError('Insufficient space for full state plus35GiB reserve; retain prior checkpoints')
                inspect_optimizer(optimizer, last, len(list(model.parameters())))
                save_checkpoint(args.output, model, optimizer, last, dict(state, counters=dict(counters)), rank, world, xm, device,
                                release_cpu_staging=p.get('checkpoint_export_cleanup', False),
                                observe_cpu_staging=p.get('checkpoint_memory_diagnostic', False))
                checkpoint_walls.append(dict(step=last, rank0_wall_seconds=time.monotonic()-saving))
                if program_study and rank == 0:
                    rotate(args.output, args.output/f'step-{last:07d}', p)
                    publish_phase(evidence, 'retained-'+str(last), p)
            finally:
                if lease is not None: lease.__exit__(None, None, None)
            if program_study and rank != 0:
                await_phase(evidence, 'retained-'+str(last), p, deadline)
        if stop or halt:
            break
    if rank == 0:
        atomic_json(args.output/'completion.json', dict(status='complete' if last == p['steps'] else 'stopped_at_boundary',
            actual_steps=last, planned_steps=p['steps'], purpose=p['purpose'], finished_utc=utc_now(),
            checkpoint_walls=checkpoint_walls, loop_including_saves_seconds=time.monotonic()-loop_started), overwrite=args.resume is not None)
    dist.destroy_process_group()


def main():
    a = argparse.ArgumentParser(description=__doc__)
    for name in ('protocol', 'output', 'stop-file'):
        a.add_argument('--'+name, type=Path, required=True)
    a.add_argument('--arm', choices=ARMS + real_trace_packing.ARMS + real_target_packing.ARMS + real_source_balanced_packing.ARMS + document_qa_packing.ARMS + document_evidence_plan_packing.ARMS + permutation_program_packing.ARMS + real_cpt_scaffold.ARMS + (PROGRAM_QUALIFICATION_ARM, SCAFFOLD_QUALIFICATION_ARM, uniform_lm.ARM), required=True); a.add_argument('--deadline-utc', required=True)
    a.add_argument('--execution-release', type=Path)
    a.add_argument('--resume', type=Path); a.add_argument('--attempt-id'); a.add_argument('--halt-after', type=int)
    args = a.parse_args(); p = json.loads(args.protocol.read_text()); validate(p)
    if p['schema'] == UNIFORM_SCHEMA:
        uniform_lm.validate_execution(p, args, source_files())
    if p.get('purpose') == SCAFFOLD_STUDY_PURPOSE:
        validate_scaffold_execution(p, args)
    if p.get('purpose') == PROGRAM_STUDY_PURPOSE:
        validate_program_execution(p, args)
    if p.get('purpose') in EVIDENCE_WEIGHT_PURPOSES:
        if args.arm != 'evidence_select':
            raise ValueError('Only the new selection-weight treatment is authorized')
        control_file = Path(p['control_protocol_file'])
        if sha256_file(control_file) != p['control_protocol_sha256']:
            raise ValueError('Frozen comparison control changed')
        validate_evidence_weight_control(p, json.loads(control_file.read_text()))
    if p.get('purpose') in EVIDENCE_FORMAT_PURPOSES:
        if args.arm != 'evidence_direct':
            raise ValueError('Only the coefficient-matched format control is authorized')
        comparisons = []
        for key in ('control_protocol', 'selector_protocol'):
            path = Path(p[key+'_file'])
            if sha256_file(path) != p[key+'_sha256']:
                raise ValueError('Frozen format comparison protocol changed')
            comparisons.append(json.loads(path.read_text()))
        validate_evidence_format_control(p, *comparisons)
    if p['schema'] == DEPTH_SCHEMA and args.arm != 'writer':
        raise ValueError('This intervention trains the writer only; the reader remains frozen')
    if p.get('purpose') in ('real_trace_cpt_study', 'real_target_sft_study', 'real_source_balanced_study', 'document_qa_study', 'document_evidence_plan_pilot', 'document_evidence_weight_ablation', 'document_evidence_format_weight_control'):
        qualification_file = Path(p['hardware_qualification_file'])
        qualification = json.loads(qualification_file.read_text())
        shape = {k:p[k] for k in ('world_size','micro_batch','global_batch','width','target_slots','head_chunk','accumulation')}
        expected_status = ('PASS_ACTUAL_DOCUMENT_EVIDENCE_FORMAT_WEIGHT_FULL_RESTORE' if p.get('purpose') == 'document_evidence_format_weight_control' else
                           'PASS_ACTUAL_DOCUMENT_EVIDENCE_WEIGHT_FULL_RESTORE' if p.get('purpose') == 'document_evidence_weight_ablation' else
                           'PASS_ACTUAL_DOCUMENT_EVIDENCE_PLAN_FULL_RESTORE' if p['schema'] == EVIDENCE_SCHEMA else
                           'PASS_ACTUAL_DOCUMENT_QA_FULL_RESTORE' if p['schema'] == DOCUMENT_SCHEMA else
                           'PASS_ACTUAL_REAL_SOURCE_BALANCED_FULL_RESTORE' if p['schema'] == BALANCED_SCHEMA else
                           'PASS_ACTUAL_REAL_TARGET_SFT_FULL_RESTORE' if p['schema'] == TARGET_SCHEMA
                           else 'PASS_ACTUAL_REAL_TRACE_SFT_FULL_RESTORE')
        if (sha256_file(qualification_file) != p['hardware_qualification_sha256']
                or qualification.get('status') != expected_status
                or qualification.get('source_files') != p['source_files']
                or qualification.get('qualified_training_shape') != shape):
            raise ValueError('Actual same-source and same-shape real-data recovery qualification required')
    allowed_arms = ((uniform_lm.ARM,) if p['schema'] == UNIFORM_SCHEMA else
                   (real_cpt_scaffold.ARMS if p.get('purpose') == SCAFFOLD_STUDY_PURPOSE
                    else (SCAFFOLD_QUALIFICATION_ARM,)) if p['schema'] == SCAFFOLD_SCHEMA else
                    (permutation_program_packing.ARMS if p.get('purpose') == PROGRAM_STUDY_PURPOSE
                     else (PROGRAM_QUALIFICATION_ARM,)) if p['schema'] == PROGRAM_SCHEMA else
                    document_evidence_plan_packing.ARMS if p['schema'] == EVIDENCE_SCHEMA else
                    document_qa_packing.ARMS if p['schema'] == DOCUMENT_SCHEMA else
                    real_source_balanced_packing.ARMS if p['schema'] == BALANCED_SCHEMA else
                    real_target_packing.ARMS if p['schema'] == TARGET_SCHEMA else
                    real_trace_packing.ARMS if p['schema'] == REAL_SCHEMA else ARMS)
    if args.arm not in allowed_arms:
        raise ValueError('Arm does not belong to the declared experiment schema')
    if args.halt_after is not None and not 1 <= args.halt_after < p['steps']:
        raise ValueError('Explicit unfinished complete-update boundary required')
    if (args.resume is None) != (args.attempt_id is None) or args.resume is None and args.output.exists():
        raise ValueError('Fresh output or explicit original-run recovery attempt required')
    if require_utc_deadline(args.deadline_utc) <= datetime.now(timezone.utc):
        raise ValueError('Future finite deadline required')
    if (os.environ.get('NEURONCORE_NUM_DEVICES') != str(p['world_size'])
            or os.environ.get('NEURON_RT_VISIBLE_CORES') != p['visible_cores']
            or os.environ.get('NEURON_LOGICAL_NC_CONFIG') != '2'
            or os.environ.get('NEURON_RT_CONTIGUOUS_SCRATCHPAD') != '0'
            or any(not os.environ.get(k) for k in ('MASTER_PORT', 'NEURON_RT_ROOT_COMM_ID', 'MASTER_ADDR'))):
        raise ValueError('Explicit declared topology and communication endpoints required')
    if args.attempt_id is not None and not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,63}', args.attempt_id):
        raise ValueError('One fresh simple recovery attempt ID required')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with (args.output.parent/('.'+args.output.name+'.writer.lock')).open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        from transformers import AutoTokenizer
        from train.evaluate import asset_provenance
        from train.useful_note_sft_driver import load_native_reference
        if sha256_file(p['data_file']) != p['data_sha256'] or sha256_file(p['base_protocol']) != p['base_protocol_sha256']:
            raise ValueError('Frozen data or native base protocol changed')
        rows = [json.loads(x) for x in Path(p['data_file']).read_text().splitlines()][:p['items']]
        if p['schema'] == SCAFFOLD_SCHEMA:
            parent_path, manifest_path = Path(p['parent_data_file']), Path(p['data_manifest_file'])
            if (sha256_file(parent_path) != SCAFFOLD_PARENT_SHA
                    or sha256_file(manifest_path) != SCAFFOLD_MANIFEST_SHA):
                raise ValueError('Frozen full TRAIN and CPU data manifest required')
            parent = [json.loads(line) for line in parent_path.read_text().splitlines()]
            if p['purpose'] == SCAFFOLD_STUDY_PURPOSE:
                if rows != parent or sha256_file(p['data_file']) != SCAFFOLD_PARENT_SHA:
                    raise ValueError('Science requires the entire frozen TRAIN20480 source cohort')
                verify_scaffold_science_release(p)
            else:
                selected = real_cpt_scaffold.grouped_order(parent, seed=p['seed'], passes=1)[:128]
                if rows != [parent[i] for i in selected]:
                    raise ValueError('Qualification must use the fixed first128 grouped TRAIN sources')
        if p['schema'] == PROGRAM_SCHEMA:
            manifest_file = Path(p['data_manifest_file'])
            manifest = json.loads(manifest_file.read_text())
            if (sha256_file(manifest_file) != p['data_manifest_sha256']
                    or manifest.get('schema') != 'permutation-program-v28-cpu-v1'
                    or manifest.get('status') != 'CPU_DATA_ONLY_NO_TRAINING_OR_EVALUATION_RELEASE'
                    or manifest['files']['TRAIN-labels.jsonl'] != p['data_sha256']
                    or manifest['inventories']['TRAIN']['questions'] != 32768
                    or any(row.get('schema') != manifest['schema'] or row.get('split') != 'TRAIN'
                           for row in rows)):
                raise ValueError('Pinned prefix-computable program TRAIN data required')
        if p['schema'] == DOCUMENT_SCHEMA:
            manifest_file = Path(p['data_manifest_file'])
            manifest = json.loads(manifest_file.read_text())
            if (sha256_file(manifest_file) != p['data_manifest_sha256']
                    or manifest.get('status') != 'PASS_ALL_6144_NATIVE_QA_VIEWS_AND_EVIDENCE_CLEAN_COHORTS_CPU_ONLY'
                    or manifest['files']['training']['sha256'] != p['data_sha256']
                    or manifest.get('items') != 2048 or manifest.get('held') != 512
                    or any(sha256_file(ROOT/name) != value for name,value in manifest['source_files'].items())):
                raise ValueError('Pinned complete native/evidence-clean QA data preparation required')
        if p['schema'] == EVIDENCE_SCHEMA:
            manifest_file = Path(p['data_manifest_file'])
            manifest = json.loads(manifest_file.read_text())
            if (sha256_file(manifest_file) != p['data_manifest_sha256']
                    or manifest.get('status') != 'PASS_ALL_6144_NATIVE_POINTER_VIEWS_AND_FRESH255_CPU_ONLY'
                    or manifest['files']['training']['sha256'] != p['data_sha256']
                    or manifest.get('items') != 2048 or manifest.get('held') != 255
                    or any(sha256_file(ROOT/name) != value for name, value in manifest['source_files'].items())
                    or any(set(row) != {'item_id', 'split', 'question', 'answer', 'support_indices'} for row in rows)):
                raise ValueError('Pinned pointer-only labels and fresh evidence-clean panel required')
        if p['schema'] == UNIFORM_SCHEMA:
            uniform_lm.validate_sources(p, rows)
        if (len(rows) != p['items'] or len({r['item_id'] for r in rows}) != p['items']
                or any((r.get('split') != 'TRAIN' if p['schema'] == PROGRAM_SCHEMA else
                        r.get('split') != 'train' if p['schema'] in (REAL_SCHEMA, TARGET_SCHEMA, BALANCED_SCHEMA, DOCUMENT_SCHEMA, EVIDENCE_SCHEMA, SCAFFOLD_SCHEMA, UNIFORM_SCHEMA) else r.get('panel') != 'TRAIN') for r in rows)):
            raise ValueError('Complete declared distinct TRAIN expressions required')
        assets = asset_provenance(Path(p['model_dir'])); base = json.loads(Path(p['base_protocol']).read_text())
        if (assets['model_id'] != p['model_id'] or assets['manifest_sha256'] != base['model_manifest_sha256']
                or p['base_signature'] != base['base_signature'] or p['precision_contract'] != base['precision_contract']):
            raise ValueError('Pinned real released model required')
        tok = AutoTokenizer.from_pretrained(p['model_dir'], local_files_only=True, trust_remote_code=False)
        refs, native = load_native_reference(base, assets); order = order_for(p, rows)
        if args.resume is not None:
            from train.neuron_driver import verify_checkpoint
            args.resume_manifest = verify_checkpoint(args.resume, canonical_hash(p), p['world_size'])
            if (args.resume.parent.resolve() != args.output.resolve() or args.resume.is_symlink()
                    or json.loads((args.output/'run.json').read_text())['arm'] != args.arm
                    or args.resume_manifest['step'] >= p['steps']):
                raise ValueError('Original run/arm full checkpoint required')
            metric_file = args.output/'metrics.jsonl'; lines = metric_file.read_bytes().splitlines(keepends=True)
            step = args.resume_manifest['step']
            if len(lines) < step or [json.loads(x)['step'] for x in lines[:step]] != list(range(1, step+1)):
                raise ValueError('Saved complete metric prefix required')
            for i, line in enumerate(lines[:step]):
                m = json.loads(line)
                if m['ordered_item_ids'] != [rows[k]['item_id'] for k in order[i*p['global_batch']:(i+1)*p['global_batch']]] or m['counters']['expressions'] != (i+1)*p['global_batch']:
                    raise ValueError('Saved completed order/counters differ')
            later = [q for q in args.output.glob('step-*') if int(q.name[5:]) > step]
            if any((q/'COMPLETE.json').exists() for q in later):
                raise ValueError('Use the latest complete resumable checkpoint')
            args.recovery_dir = args.output/'recoveries'/args.attempt_id; args.recovery_dir.mkdir(parents=True, exist_ok=False)
            (args.recovery_dir/'previous-metrics.jsonl').write_bytes(b''.join(lines))
            if (args.output/'completion.json').exists():
                (args.recovery_dir/'previous-completion.json').write_bytes((args.output/'completion.json').read_bytes())
            for q in later:
                dest = args.recovery_dir/'abandoned'/q.name; dest.parent.mkdir(exist_ok=True); q.rename(dest)
            from weekend_thinking.artifacts import atomic_bytes
            atomic_bytes(metric_file, b''.join(lines[:step]), overwrite=True)
        import torch_xla.distributed.xla_multiprocessing as xmp
        xmp.spawn(worker, args=(args, p, rows, tok, assets, refs, native, order, sha256_file(args.protocol)), nprocs=None)


if __name__ == '__main__':
    main()

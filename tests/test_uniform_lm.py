"""Uniform token coefficients, distributed accumulation and sealed source splits."""
import copy
import json
from types import SimpleNamespace

import pytest
import torch
from torch.nn import functional as F
from transformers import Qwen3Config, Qwen3ForCausalLM

from train import uniform_lm as lm
from train import partial_state_driver as driver
from train.masked_causal_ce import masked_causal_ce
from train.neuron_backend import NeuronCausalBackend
from train import uniform_lm_evaluate as scoring
from train.uniform_lm_evaluate import token_logps
from scripts.prepare_uniform_lm import select_sources
from weekend_thinking.artifacts import sha256_file


class Tokenizer:
    def __init__(self, size=31): self.size=size
    def __len__(self): return self.size


def test_full_window_evaluation_matches_dense_loss_and_keeps_real_special_targets():
    torch.manual_seed(48)
    model = Qwen3ForCausalLM(Qwen3Config(vocab_size=31, hidden_size=16,
        intermediate_size=24, num_hidden_layers=2, num_attention_heads=2,
        num_key_value_heads=1, head_dim=8, attention_dropout=0.,
        tie_word_embeddings=True, eos_token_id=30))
    model.eval()
    ids = torch.randint(0, 31, (3, 24))
    ids[0, 8] = 30
    backend = NeuronCausalBackend(model, checkpoint_trunk=False, head_chunk=7)
    with torch.no_grad():
        actual = token_logps(backend, ids)
        logits = model(input_ids=ids).logits[:, :-1].float()
        expected = logits.log_softmax(-1).gather(-1, ids[:, 1:, None]).squeeze(-1)
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)
    assert actual.shape == (3, 23)
    torch.testing.assert_close(actual[:, -16:].sum(1), expected[:, -16:].sum(1))


def test_scoring_requires_all_terminals_and_a_frozen_confirmation_choice(tmp_path):
    data = tmp_path/'data.jsonl'; data.write_text('{}\n')
    manifest = tmp_path/'manifest.json'; manifest.write_text(json.dumps(dict(files={
        k: dict(sha256=sha256_file(data), rows=32) for k in ('DEV', 'CONFIRM')})))
    output = tmp_path/'scores'
    p = dict(schema='uniform-lm-likelihood-v1', world_size=8, micro_batch=4,
        visible_cores='32-39', source_files=driver.source_files(),
        evaluator_sha256=sha256_file(scoring.__file__), data_file=str(data),
        data_sha256=sha256_file(data), data_manifest_file=str(manifest),
        data_manifest_sha256=sha256_file(manifest), cohort='DEV', items=32,
        model_name='B0', checkpoint=None)
    release = dict(jobs={str(output): {}}, all_terminal_models={})
    with pytest.raises(ValueError, match='all four terminal'):
        scoring.validate(p, release, output)
    release['all_terminal_models']={str(i): dict(step=128) for i in range(4)}
    scoring.validate(p, release, output)
    choice = tmp_path/'choice.json';choice.write_text(json.dumps(dict(recipe='recipe-b')))
    release['selection']=dict(file=str(choice), sha256=sha256_file(choice))
    p['cohort']='CONFIRM'
    p['model_name']='recipe-a'
    with pytest.raises(ValueError, match='Only B0 and the frozen choice'):
        scoring.validate(p, release, output)
    p['model_name']='B0';scoring.validate(p, release, output)
    choice.write_text(json.dumps(dict(recipe='recipe-a')))
    with pytest.raises(ValueError, match='Freeze development choice'):
        scoring.validate(p, release, output)


def row(i, width=8):
    return dict(schema=lm.SCHEMA, item_id=f'item-{i}', document_id=f'doc-{i}',
        split='train', cohort='TRAIN', ids=[(i+j)%31 for j in range(width)])


def recipe():
    return dict(schema=lm.SCHEMA, loss='uniform_next_token', items=12, steps=3,
        world_size=2, micro_batch=1, global_batch=4, accumulation=2,
        width=8, target_slots=7, schedule='warmup_cosine', lr=1e-5,
        minimum_lr_ratio=.1, warmup_steps=1)


def protocol():
    return dict(schema=lm.SCHEMA, loss='uniform_next_token',
        purpose='uniform_lm_qualification', model_id='Qwen/Qwen3-1.7B-Base',
        items=1024, steps=4, world_size=8, micro_batch=4, global_batch=256,
        accumulation=8, width=528, target_slots=527, head_chunk=32,
        schedule='warmup_cosine', lr=1e-5, minimum_lr_ratio=.1, warmup_steps=2,
        seed=11, save_every=2, parameter_dtype='bf16', optimizer_dtype='fp32',
        data_manifest_file='source-manifest.json', data_manifest_sha256='a'*64,
        data_sha256='b'*64, source_files=driver.source_files())


def test_every_original_shifted_token_has_one_label_without_inserted_text():
    source=row(0)
    pack=lm.training_pack(source,Tokenizer(),width=8,target_slots=7)
    assert pack['input_ids'].tolist()==[source['ids']]
    assert pack['target_positions'].tolist()==[list(range(1,8))]
    assert pack['labels'].tolist()==[source['ids'][1:]]
    assert pack['future_mask'].all() and not pack['note_close_mask'].any()
    # Even the actual native EOS/padding ID remains a real source target.
    source['ids'][3]=151643
    special=lm.training_pack(source,Tokenizer(151936),width=8,target_slots=7)
    assert special['labels'][0,2]==151643 and special['future_mask'][0,2]


def test_accumulated_rank_gradients_match_one_dense_uniform_token_mean():
    torch.manual_seed(73)
    model=Qwen3ForCausalLM(Qwen3Config(vocab_size=31,hidden_size=8,
        intermediate_size=16,num_hidden_layers=1,num_attention_heads=2,
        num_key_value_heads=1,head_dim=4,tie_word_embeddings=True,
        attn_implementation='eager')).eval()
    reference=copy.deepcopy(model);backend=NeuronCausalBackend(model,head_chunk=3)
    p=recipe();rows=[row(i) for i in range(12)];order=list(range(12));total=0.
    for rank in range(2):
        for micro in range(2):
            pack=lm.batch_at(rows,Tokenizer(),p,order,0,rank,micro)
            loss=masked_causal_ce(backend,**pack).loss/4
            total+=loss.detach().item();loss.backward()
    ids=torch.tensor([r['ids'] for r in rows[:4]])
    expected=F.cross_entropy(reference(ids[:,:-1]).logits.float().transpose(1,2),ids[:,1:])
    expected.backward()
    assert total==pytest.approx(expected.item(),abs=1e-6)
    for (name,a),(other,b) in zip(model.named_parameters(),reference.named_parameters()):
        assert name==other and a.grad is not None and b.grad is not None
        torch.testing.assert_close(a.grad,b.grad,rtol=1e-5,atol=1e-6)


def test_actual_driver_dispatch_covers_each_source_once_across_ranks_and_microsteps():
    p=protocol();driver.validate(p);rows=[row(i,528) for i in range(1024)]
    order=driver.order_for(p,rows);seen=[]
    for step in range(4):
        for micro in range(8):
            for rank in range(8):
                start=step*256+micro*32+rank*4;seen.extend(order[start:start+4])
                pack=driver.batch_at(rows,Tokenizer(),p,order,lm.ARM,step,rank,micro)
                assert pack['input_ids'].tolist()==[rows[i]['ids'] for i in order[start:start+4]]
                assert pack['future_mask'].sum()==4*527
                assert not pack['note_close_mask'].any()
    assert len(seen)==len(set(seen))==1024
    with pytest.raises(ValueError):driver.batch_at(rows,Tokenizer(),p,order,'computation',0,0)


@pytest.mark.parametrize('schedule',['warmup_constant','warmup_cosine'])
def test_learning_rate_is_an_optimizer_step_function_across_resume(schedule):
    p={**recipe(),'items':48,'steps':12,'warmup_steps':3,'schedule':schedule}
    rates=[lm.learning_rate(p,s) for s in range(12)]
    assert rates[:3]==pytest.approx([p['lr']/3,p['lr']*2/3,p['lr']])
    assert rates[-1]==pytest.approx(p['lr']*(.1 if schedule=='warmup_cosine' else 1))
    resumed=[lm.learning_rate(p,s) for s in range(6,12)]
    assert rates[6:]==resumed
    optimizer=torch.optim.AdamW([torch.nn.Parameter(torch.ones(1))],lr=1.)
    for step,expected in enumerate(rates):
        assert lm.set_learning_rate(optimizer,p,step)==expected
        assert optimizer.param_groups[0]['lr']==expected


@pytest.mark.parametrize('change',[dict(accumulation=1),dict(items=13),dict(width=9),
    dict(warmup_steps=3),dict(lr=float('nan')),dict(lr=True),dict(minimum_lr_ratio=-.1),
    dict(schedule='constant'),dict(loss='source_plus_continuation'),dict(world_size=True)])
def test_changed_weighting_exposure_or_schedule_is_rejected(change):
    with pytest.raises(ValueError):lm.validate_recipe({**recipe(),**change})


def test_source_selection_cannot_depend_on_arithmetic_labels():
    records=[dict(document_id=f'{i:064x}',positions=i+1,used=i%11==0,
                  verified_boundaries=i%3,chosen_verified=i%2) for i in range(80)]
    counts=dict(TRAIN=12,DEV=4,CONFIRM=4,QUALIFICATION=4);blocked={f'{2:064x}'}
    chosen=select_sources(iter(records),counts,blocked,'uniform-test')
    altered=[dict(r,verified_boundaries=1000-r['verified_boundaries'],chosen_verified=99) for r in records]
    assert select_sources(iter(altered),counts,blocked,'uniform-test')==chosen
    assert len({r['document_id'] for r in chosen})==24
    assert not {r['document_id'] for r in chosen}&blocked
    assert all(512<=r['cut']<512+r['positions'] for r in chosen)
    with pytest.raises(ValueError):select_sources(iter(records),dict(TRAIN=100),blocked,'uniform-test')


def test_source_manifest_cannot_fit_development_or_confirmation(tmp_path):
    path=tmp_path/'manifest.json'
    path.write_text(json.dumps(dict(status='COMPLETE_UNIFORM_LM_SOURCE_PREPARATION',
        files={'TRAIN':dict(sha256='b'*64,rows=2)})))
    p=dict(purpose='uniform_lm_development',data_manifest_file=str(path),
           data_manifest_sha256=sha256_file(path),data_sha256='b'*64)
    rows=[row(0),row(1)];lm.validate_sources(p,rows)
    for cohort in ('DEV','CONFIRM','QUALIFICATION'):
        with pytest.raises(ValueError):lm.validate_sources(p,[dict(r,cohort=cohort) for r in rows])


def test_cpu_protocol_is_not_a_native_execution_release(tmp_path):
    args=SimpleNamespace(arm=lm.ARM,execution_release=None,output=tmp_path/'run',protocol=tmp_path/'p.json')
    with pytest.raises(ValueError,match='native stage release'):
        lm.validate_execution(protocol(),args,driver.source_files())


def test_a_pass_label_without_native_lr_and_recovery_checks_does_not_release_fitting(tmp_path):
    p={**protocol(),'purpose':'uniform_lm_development'};source=driver.source_files()
    proto=tmp_path/'p.json';proto.write_text(json.dumps(p))
    receipt=tmp_path/'qualification.json';receipt.write_text(json.dumps(dict(
        status='PASS_UNIFORM_LM_NATIVE_QUALIFICATION',source_files=source,
        accumulation=8,world_size=8,width=528)))
    args=SimpleNamespace(arm=lm.ARM,execution_release=tmp_path/'release.json',
                         output=tmp_path/'run',protocol=proto)
    args.execution_release.write_text(json.dumps(dict(status='RELEASED_UNIFORM_LM_NATIVE_STAGE',
        source_files=source,maximum_reserved_chip_hours=18,
        jobs={str(args.output):dict(protocol_sha256=sha256_file(proto))},
        native_qualification=dict(file=str(receipt),sha256=sha256_file(receipt)))))
    with pytest.raises(ValueError,match='qualification must cover'):
        lm.validate_execution(p,args,source)

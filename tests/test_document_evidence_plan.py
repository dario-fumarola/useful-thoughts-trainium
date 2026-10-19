import copy

import pytest
import torch

from train import document_evidence_plan_packing as packing
from train.document_evidence_plan_data import fresh_candidates, select_fresh_panel, training_row
from train.document_qa_answers import parse_answer
from train.useful_note_sft_data import EOS


class Tokenizer:
    eos_token_id = EOS
    def encode(self, text, add_special_tokens=False): return list(text.encode())
    def decode(self, ids, **kwargs): return bytes(ids).decode()


def raw(index):
    return dict(id=f'item-{index}', composed_question_text=f'Question {index}?',
        answer_text=f'Answer {index}', decomposed_instances=[
            dict(dataset='test', id=2*index), dict(dataset='test', id=2*index+1)],
        contexts=[dict(wikipedia_title=f'Page {index}', paragraph_text=f'Fact {index}.', is_supporting=True),
                  dict(wikipedia_title=f'Other {index}', paragraph_text=f'Other fact {index}.', is_supporting=True)])


def test_offline_targets_drop_claims_and_policy_input_has_no_annotations():
    source = raw(0)
    source['trace'] = 'Unsupported claim SECRET'
    row = training_row(source)
    assert set(row) == {'item_id', 'split', 'question', 'answer', 'support_indices'}
    before = packing.prompt(row['question'], 'selection')
    row['answer'] = 'POISON ANSWER'
    row['support_indices'] = [999]
    assert packing.prompt(row['question'], 'selection') == before
    assert 'SECRET' not in before and 'POISON' not in before
    with pytest.raises(ValueError):
        packing.prompt(row, 'selection')


def test_three_views_preserve_source_and_final_labels_and_coefficients():
    row = training_row(raw(0))
    packs = {view: packing.training_pack(row, Tokenizer(), view, width=1024, target_slots=512)
             for view in ('direct', 'empty', 'selected')}
    for component in ('source', 'answer'):
        labels = [p['labels'][p[f'{component}_mask']] for p in packs.values()]
        assert all(torch.equal(labels[0], other) for other in labels[1:])
        coefficients = [.5 / p[f'{component}_mask'].sum().item() for p in packs.values()]
        assert len(set(coefficients)) == 1
    for view, pack in packs.items():
        masks = [pack[f'{name}_mask'] for name in packing.COMPONENTS]
        assert not any((a & b).any() for i, a in enumerate(masks) for b in masks[i+1:])
        final = pack['labels'][pack['answer_mask']].tolist()
        assert final[-1] == EOS and final.count(EOS) == 1
        assert Tokenizer().decode(final[:-1]) == packing.FINAL + row['answer']
        assert parse_answer(Tokenizer().decode(final[:-1]), 'prior', terminated=True).answer == row['answer']
        selection = pack['labels'][pack['selection_mask']].tolist()
        expected = '' if view == 'direct' else 'Evidence:\n' if view == 'empty' else 'Evidence: [1] [2]\n'
        assert Tokenizer().decode(selection) == expected
        active = torch.stack(masks).any(0)
        assert torch.equal(pack['labels'][active], pack['input_ids'][0, pack['target_positions'][active]])
    with pytest.raises(ValueError):
        packing.training_pack(row, Tokenizer(), 'selected', width=16, target_slots=512)


@pytest.mark.parametrize('indices', [[True], [0], [3], [1, 1], [2, 1]])
def test_selection_targets_cannot_silently_repair_invalid_ids(indices):
    with pytest.raises(ValueError):
        packing.selection_field(indices, 2)


def test_schedule_interleaves_and_gives_every_question_two_views_per_arm():
    for index in range(128):
        for arm in packing.ARMS:
            views = [packing.view_for(arm, epoch, index) for epoch in range(4)]
            assert views.count('direct') == 2
            assert sum(view != 'direct' for view in views) == 2
    for epoch in range(4):
        for arm in packing.ARMS:
            assert sum(packing.view_for(arm, epoch, i) == 'direct' for i in range(128)) == 64


def test_fresh_exclusions_cover_old_visible_distractors_pages_and_original_seeds():
    rows = [raw(i) for i in range(8)]
    rows[0]['contexts'].append(dict(wikipedia_title='Old train distractor',
        paragraph_text='TRAIN DISTRACTOR', is_supporting=False))
    rows[1]['contexts'].append(dict(wikipedia_title='Old held distractor',
        paragraph_text='HELD DISTRACTOR', is_supporting=False))
    rows[2]['contexts'][0]['wikipedia_title'] = 'Old held distractor'
    rows[3]['contexts'][0]['paragraph_text'] = 'TRAIN DISTRACTOR'
    rows[4]['decomposed_instances'][0] = copy.deepcopy(rows[0]['decomposed_instances'][0])
    rows[5]['composed_question_text'] = rows[1]['composed_question_text']
    lengths = {row['id']: 100 for row in rows}
    lengths['item-6'] = 3073
    candidates = fresh_candidates(rows, lengths, {'item-0'}, {'item-1'})
    assert [row['item_id'] for row in candidates] == ['item-7']


def identity(index, hops=2):
    return dict(item_id=str(index), question=f'q{index}', hops=hops,
        support_text={f'p{index}'}, support_titles={f't{index}'}, seed_ids={f's{index}'})


def test_fresh_selection_is_order_independent_and_keeps_high_hop_quota():
    candidates = [identity(i, 3 if i >= 8 else 2) for i in range(12)]
    chosen = select_fresh_panel(candidates, count=6, higher_hop_count=2)
    assert chosen == select_fresh_panel(list(reversed(candidates)), count=6, higher_hop_count=2)
    assert sum(row['hops'] >= 3 for row in chosen) >= 2
    with pytest.raises(ValueError, match='higher-hop'):
        select_fresh_panel(candidates[:8], count=6, higher_hop_count=2)


@pytest.mark.parametrize('field', ['support_text', 'support_titles', 'seed_ids', 'question'])
def test_shared_evidence_or_question_prevents_two_candidates_entering_same_panel(field):
    first, second = identity(0), identity(1)
    second[field] = copy.deepcopy(first[field])
    with pytest.raises(ValueError, match='1/2'):
        select_fresh_panel([first, second], count=2, higher_hop_count=0)


def test_driver_accumulation_covers_global_batch_with_shared_views():
    from train.partial_state_driver import EVIDENCE_SCHEMA, batch_at, order_for, source_files, validate
    protocol = dict(schema=EVIDENCE_SCHEMA, model_id='Qwen/Qwen3-1.7B-Base',
        world_size=8, micro_batch=1, global_batch=32, accumulation=4,
        width=3328, target_slots=512, head_chunk=32, schedule='constant',
        parameter_dtype='bf16', optimizer_dtype='fp32', items=128,
        purpose='document_evidence_plan_qualification', steps=8, save_every=2,
        seed=2026101923, lr=1e-5, source_sample_labels=256,
        view_schedule='interleaved_fixed_item_plus_epoch_parity',
        loss='separate_mean_source_answer_selection_per_example',
        source_loss_weight=.5, answer_loss_weight=.5, selection_loss_weight=.1,
        data_sha256='a'*64, data_manifest_sha256='b'*64, source_files=source_files())
    validate(protocol)
    for key, value in [('answer_loss_weight', .25), ('source_loss_weight', True),
                       ('loss', 'single_mean_task'), ('view_schedule', 'whole_pass')]:
        with pytest.raises(ValueError):
            validate({**protocol, key: value})
    rows = [training_row(raw(i)) for i in range(128)]
    order = order_for(protocol)
    for step in (0, 4):
        seen = []
        for micro in range(4):
            for rank in range(8):
                control = batch_at(rows, Tokenizer(), protocol, order, 'evidence_direct', step, rank, micro)
                selected = batch_at(rows, Tokenizer(), protocol, order, 'evidence_select', step, rank, micro)
                answer = control['labels'][control['answer_mask']].tolist()
                other = selected['labels'][selected['answer_mask']].tolist()
                assert answer == other
                seen.append(int(Tokenizer().decode(answer[:-1]).split('Answer ')[-1]))
        assert seen == order[step*32:(step+1)*32] and len(set(seen)) == 32

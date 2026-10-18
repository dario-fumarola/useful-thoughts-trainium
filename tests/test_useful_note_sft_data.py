"""Scientific packing checks: future separation, termination, and loss exposure."""
import copy

import pytest
import torch

from train import useful_note_sft_data as data


class ByteTokenizer:
    eos_token_id = data.EOS

    def encode(self, text, add_special_tokens=False):
        return list(text.encode('utf-8'))

    def decode(self, ids, **kwargs):
        return bytes(ids).decode('utf-8')


@pytest.fixture
def parts():
    return dict(old_header=[300], old_open=[301, 302], old_close=[151668, 198],
                inline_open=[303], inline_close=[151668, 220],
                published_header=[304, 305], scratch_open=[306], scratch_close=[340])


@pytest.fixture
def row():
    past = 'A' * 128 + '12*24 ='
    source = (past + ' 288.\nThe following original document text.').ljust(528, 'X')
    ids = list(source.encode('ascii')); cut = len(past)
    return dict(split='train', prefix_ids=ids[:cut], original_source_ids=ids,
        natural_future16_ids=ids[cut:cut+16], boundary=dict(expression='12*24'),
        numeric_value='288', numeric_surface_text='288',
        reanchor_ids=list(b'12*24 ='))


def active(batch):
    return batch['note_close_mask'] | batch['future_mask']


@pytest.mark.parametrize('interface', [f'I{i}' for i in range(6)])
def test_writer_future_is_absent_and_first_stop_is_learned(row, parts, interface):
    tok = ByteTokenizer(); teacher = data.note_ids(row, tok)
    expected = data.pack_writer(row['prefix_ids'], teacher, interface=interface,
                               parts=parts, width=832, target_slots=129)
    original = data.pack_example(row, tok, 'writer', interface=interface,
                                parts=parts, width=832, target_slots=129)
    changed = copy.deepcopy(row); cut = len(row['prefix_ids'])
    changed['original_source_ids'][cut:] = [ord('Q')] * (528-cut)
    changed['natural_future16_ids'] = [ord('Q')] * 16
    alternate = data.pack_example(changed, tok, 'writer', interface=interface,
                                 parts=parts, width=832, target_slots=129)
    for key in expected:
        assert torch.equal(expected[key], original[key])
        assert torch.equal(original[key], alternate[key])
    labels = original['labels'][active(original)].tolist()
    assert labels == teacher + [data.writer_stop_ids(interface, parts)[0]]
    assert not original['future_mask'].any()
    assert len(labels) == int(original['note_close_mask'].sum())


@pytest.mark.parametrize('interface', ['I0', 'I3', 'I5'])
def test_source_reader_and_augmented_exposures_are_distinct(row, parts, interface):
    tok = ByteTokenizer(); batches = {}
    for component in ('source', 'reader', 'augmented'):
        batch = data.pack_example(row, tok, component, interface=interface,
                                  parts=parts, width=1024, target_slots=1023)
        mask = active(batch); batches[component] = batch
        assert not batch['note_close_mask'].any()
        assert (batch['labels'][~mask] == -100).all()
        assert torch.equal(batch['labels'][mask],
                           batch['input_ids'].gather(1, batch['target_positions'])[mask])
    assert batches['source']['labels'][active(batches['source'])].tolist() == row['original_source_ids'][1:]
    assert batches['reader']['labels'][active(batches['reader'])].tolist() == row['natural_future16_ids']
    assert int(active(batches['augmented']).sum()) > 527
    prompt = data.writer_prompt(row['prefix_ids'], interface, parts)
    assert batches['augmented']['input_ids'][0, :len(prompt)].tolist() == prompt


def test_native_eos_label_and_padding_are_distinguished_by_mask():
    batch = data.packed([7, data.EOS], [1], note_positions=[1], width=5, target_slots=3)
    assert batch['input_ids'].tolist() == [[7, data.EOS, data.EOS, data.EOS, data.EOS]]
    assert batch['labels'].tolist() == [[data.EOS, -100, -100]]
    assert batch['note_close_mask'].tolist() == [[True, False, False]]


def test_full_answer_symbolic_label_accepts_exact_fraction_surface(row):
    row.update(boundary=dict(expression='(2/3)+(2/3)'),
               numeric_value='4/3', numeric_surface_text='4 / 3')
    assert data.symbolic_note(row) == 'Compute (2/3)+(2/3) = 4 / 3.'
    row['numeric_value'] = '5/3'
    with pytest.raises(ValueError, match='disagree'):
        data.symbolic_note(row)


def test_overwidth_cannot_silently_truncate_a_teacher(row, parts):
    with pytest.raises(ValueError, match='never truncate'):
        data.pack_example(row, ByteTokenizer(), 'writer', interface='I3',
                          parts=parts, width=130, target_slots=129)
    with pytest.raises(ValueError, match='first stop'):
        data.pack_writer(row['prefix_ids'], [340], interface='I5', parts=parts,
                         width=832, target_slots=129)


def test_union_pool_priority_keeps_real_distinct_cuts_and_detects_tampering(tmp_path):
    from weekend_thinking.artifacts import canonical_hash, sha256_file
    wrappers = []
    specs = [('one', 'D1', 128, 1, True), ('second', 'D2', 150, 3, False),
             ('primary', 'D2', 140, 2, True), ('third', 'D3', 130, 2, True)]
    for key, doc, cut, ops, representative in specs:
        payload = dict(item_id='original-'+key, document_id=doc, split='train')
        wrappers.append(dict(candidate_id=key, document_id=doc, split='train',
            absolute_source_cut=cut, binary_ops=ops, representative=representative,
            source_payload=payload, source_payload_sha256=canonical_hash(payload)))
    source=tmp_path/'candidates.jsonl'
    import json
    source.write_text(''.join(json.dumps(w)+'\n' for w in wrappers))
    manifest=dict(schema='compound-priority-equation-candidates-v1', candidates=4,
                  status='COMPLETE_CPU_CANDIDATE_INVENTORY_NOT_TRAINING_RELEASE',
                  candidates_file=str(source), candidates_sha256=sha256_file(source))
    rows, _, rule = data.load_candidate_rows(tmp_path,manifest,'exact-postorder-v1')
    assert [r['item_id'] for r in rows] == ['primary','third','second','one']
    assert len({r['document_id'] for r in rows}) == 3
    assert len(rows) == 4  # Real cuts, not repeated labels pretending to be docs.
    assert rows[0]['original_item_id'] == 'original-primary'
    assert all(r['teacher_note_kind']=='exact-postorder-v1' for r in rows)
    wrappers[0]['source_payload']['split']='dev'
    source.write_text(''.join(json.dumps(w)+'\n' for w in wrappers))
    manifest['candidates_sha256']=sha256_file(source)
    with pytest.raises(ValueError,match='payload/provenance'):
        data.load_candidate_rows(tmp_path,manifest,'exact-postorder-v1')


def test_union_pool_rejects_duplicate_source_cuts_even_with_new_names(tmp_path):
    import json
    from weekend_thinking.artifacts import canonical_hash, sha256_file
    payload=dict(item_id='old',document_id='D',split='train')
    wrapper=dict(candidate_id='new1',document_id='D',split='train',absolute_source_cut=128,
        binary_ops=2,representative=True,source_payload=payload,
        source_payload_sha256=canonical_hash(payload))
    source=tmp_path/'candidates.jsonl'
    source.write_text(json.dumps(wrapper)+'\n'+json.dumps(dict(wrapper,candidate_id='new2'))+'\n')
    manifest=dict(schema='finemath-compound-anchor-candidates-v1',candidates=2,
                  status='COMPLETE_CPU_CANDIDATE_INVENTORY_NOT_TRAINING_RELEASE',
                  candidates_file=str(source),candidates_sha256=sha256_file(source))
    with pytest.raises(ValueError,match='Unique TRAIN source-cut'):
        data.load_candidate_rows(tmp_path,manifest,'exact-postorder-v1')

"""Disclosed deterministic illustrations from both completed fixed readouts."""
from pathlib import Path
import gzip
import hashlib
import json
from transformers import AutoTokenizer
from train.partial_state_evaluation import donor_permutation

R = Path('/home/ubuntu/dario')
primary_path = R/'validation/real_trace_cpt_v10_complete_20261019.json'
scalar_path = R/'validation/real_trace_cpt_v10_scalar_secondary_20261019.json'
primary, scalar = (json.loads(p.read_text()) for p in (primary_path, scalar_path))
assert primary['status'] == 'COMPLETE_TWO_SEED_REAL_DOCUMENT_STUDY_ALL_FIXED_PANELS'
assert scalar['status'] == 'COMPLETE_ALL_SEVEN_MODELS_BOTH_FIXED_SCALAR_PANELS'
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def payload(report, name):
    path = R/name
    assert sha(path) == report['payloads'][name]['sha256']
    return [json.loads(line) for line in gzip.decompress(path.read_bytes()).splitlines()]
def original(path): return [json.loads(line) for line in path.open()]

tok = AutoTokenizer.from_pretrained(R/'local/assets/qwen3-1.7b-base', local_files_only=True)
out = R/'validation/real_trace_cpt_v10_complete_20261019'
examples = []
for panel, data_path, digest in [
    ('DEV_AST256', R/'local/nvme-study/data/real-trace-cpt-v10-held-panels-v2/DEV_novel_expression256.jsonl',
     '41d4aa6e6d217cdd57296953c669a95005ae1eb17b29ba5dc8565627b0c78fba'),
    ('DEV_legacy256', R/'local/nvme-study/data/real-prefix-partial-labels-v9/DEV.jsonl',
     '8ba676b1cb7cb1f46405d259911207e2d4467ffed922ef07ab2bf9e28738ce79')]:
    assert sha(data_path) == digest
    data = original(data_path)
    assert len(data) == 256
    order = [v['item_id'] for v in data]
    annotations = {v['item_id']: v for v in data}
    likelihood, notes, continuations, other_notes = {}, {}, {}, {}
    for key in primary['absolute_likelihood']:
        prefix = 'validation/real_trace_cpt_v10_complete_20261019/'
        values = payload(primary, prefix+f'{key}-{panel}-all1280-logps.jsonl.gz')
        likelihood[key] = {(v['item_id'], v['mode']): v for v in values}
        values = payload(primary, prefix+f'{key}-{panel}-all256-generated-notes.jsonl.gz')
        assert [v['item_id'] for v in values] == order
        notes[key] = {v['item_id']: v for v in values}
        donors = donor_permutation([len(v['content_ids']) for v in values], order)
        other_notes[key] = {item: values[donor] for item, donor in zip(order, donors)}
        prefix = 'validation/real_trace_cpt_v10_scalar_secondary_20261019/'
        values = payload(scalar, prefix+f'{key}-{panel}-all1280-scalar-continuations.jsonl.gz')
        continuations[key] = {(v['item_id'], v['mode']): v for v in values}
    for seed in (2026101913, 2026101914):
        raw, prior, post = (f'{arm}-seed{seed}' for arm in ('real_raw', 'real_prior', 'real_post'))
        selected = {key: ['first_three_in_frozen_document_order'] for key in order[:3]}
        def ll(model, key, mode): return likelihood[model][key, mode]['span16_logprob']
        for label, model, mode in [('training_post_raw', post, 'raw'), ('inference_prior_own', prior, 'own')]:
            difference = {key: ll(model, key, mode)-ll(raw, key, 'raw') for key in order}
            for name, key in [('maximum_observed_contrast', max(order, key=difference.get)),
                              ('minimum_observed_contrast', min(order, key=difference.get))]:
                selected.setdefault(key, []).append(label+':'+name)
        for label, model, mode in [('post_raw', post, 'raw'), ('prior_own', prior, 'own')]:
            discordant = [key for key in order if continuations[model][key, mode]['correct'] != continuations[raw][key, 'raw']['correct']]
            if discordant:
                selected.setdefault(discordant[0], []).append('first_scalar_disagreement:'+label)
        for key in order:
            if key not in selected:
                continue
            annotation = annotations[key]
            value = dict(panel=panel, seed=seed, item_id=key, document_id=annotation['document_id'],
                         selected_by=selected[key], normalized_prefix_expression=annotation['normalized_expression'],
                         expected_numeric_value=annotation['numeric_value'],
                         actual_future16=tok.decode(annotation['continuation16_ids'], skip_special_tokens=False,
                                                    clean_up_tokenization_spaces=False),
                         teacher_annotation=annotation['full_calculation_text'], models={})
            for model, modes in [('B0', ('raw', 'empty', 'own')), (raw, ('raw', 'own')),
                                  (prior, ('raw', 'empty', 'own', 'other', 'oracle')), (post, ('raw',))]:
                value['models'][model] = dict(
                    log_likelihood16={mode: ll(model, key, mode) for mode in modes},
                    scalar_continuations={mode: {field: continuations[model][key, mode][field] for field in
                        ('text', 'actions', 'stop_type', 'status', 'source_note_status', 'parsed', 'category', 'correct')}
                        for mode in modes},
                    original_own_note=({field: notes[model][key][field] for field in ('text', 'content_ids', 'stop_type', 'status', 'n_actions')} if 'own' in modes else None),
                    intact_other_note=({field: other_notes[model][key][field] for field in ('item_id','document_id','text','content_ids','stop_type','status','n_actions')} if 'other' in modes else None))
            examples.append(value)

file = out/'fixed-and-disclosed-extreme-examples.jsonl'
file.write_text(''.join(json.dumps(value, separators=(',', ':'))+'\n' for value in examples))
manifest = dict(status='ILLUSTRATIONS_REUSE_FIXED_COMPLETE_OUTPUTS_NO_NEW_OBSERVATIONS', examples=len(examples),
                file_sha256=sha(file), source_sha256=sha(__file__), primary_report_sha256=sha(primary_path),
                scalar_report_sha256=sha(scalar_path),
                selection='Per seed/panel: first3 frozen documents, largest/smallest observed post.raw−raw.raw and prior.own−raw.raw, and first scalar disagreement for each contrast when present. Ties follow frozen order; duplicate selections merge.',
                limits=['Illustrative extremes are explicitly selected; no conclusion is based on their frequency.',
                        'Every numerical endpoint uses all256 documents; these are not new independent samples.',
                        'Teacher annotations are shown for inspection and were not supplied to the deployed own-note generator.',
                        'Complete native prefixes remain in the fixed protected inputs; examples show the full arithmetic expression and actual future16.'])
(out/'example-selection.json').write_text(json.dumps(manifest, indent=2))
print(json.dumps(manifest, indent=2))

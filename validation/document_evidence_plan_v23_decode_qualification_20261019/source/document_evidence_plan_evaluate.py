"""Evidence-plan prompt preparation using the existing native QA generation engine.

No scoring occurs during generation. Held model calls require an explicit pilot
release; the disposable qualification opens only existing TRAIN questions.
"""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

from train import document_qa_evaluate as engine
from train.document_evidence_plan_packing import prompt, selection_field
from weekend_thinking.artifacts import atomic_json, sha256_file, utc_now

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = 'document-evidence-plan-complete-answers-v23'
MODES = ('direct', 'selection', 'empty', 'oracle')


def sources():
    names = ('train/document_evidence_plan_evaluate.py',
             'train/document_evidence_plan_packing.py',
             'train/document_evidence_plan_answers.py')
    return {**engine.sources(), **{name: sha256_file(ROOT/name) for name in names}}


def prompt_text(question, mode, gold=None):
    if mode == 'direct':
        return prompt(question, 'direct')
    if mode == 'selection':
        return prompt(question, 'selection')
    if mode == 'empty':
        return prompt(question, 'selection') + selection_field([], len(question['contexts']))
    if mode == 'oracle':
        if not isinstance(gold, dict):
            raise ValueError('Explicit offline annotated pointer control required')
        return prompt(question, 'selection') + selection_field(gold['support_indices'], len(question['contexts']))
    raise ValueError('Declared evidence-plan condition required')


def prepare(output, data_dir, model_dir, base_protocol, *, qualification=False,
            modes=None, binding=None, release=None):
    import torch
    from transformers import AutoTokenizer
    from train.evaluate import asset_provenance
    from train.evaluate_local import setup_model
    from train.useful_note_sft_driver import load_native_reference

    output, data_dir, model_dir, base_protocol = (Path(path).resolve()
        for path in (output, data_dir, model_dir, base_protocol))
    binding = Path(binding).resolve() if binding else None
    release = Path(release).resolve() if release else None
    manifest = engine.read(data_dir/'manifest.json')
    if manifest.get('status') != 'PASS_ALL_6144_NATIVE_POINTER_VIEWS_AND_FRESH255_CPU_ONLY':
        raise ValueError('Frozen evidence-plan data and native packing required')
    tok = AutoTokenizer.from_pretrained(model_dir, local_files_only=True, trust_remote_code=False)
    if qualification:
        if binding is not None or release is not None:
            raise ValueError('Disposable unchanged-B0 TRAIN qualification only')
        item = manifest['files']['training']
        if sha256_file(item['path']) != item['sha256']:
            raise ValueError('Frozen TRAIN data changed')
        pool = engine.rows(item['path'])[:128]
        ranked = sorted((row['question'] for row in pool),
            key=lambda q: (len(tok.encode(prompt(q, 'selection'), add_special_tokens=False)), q['item_id']))
        questions, golds, allowed = [ranked[-1], ranked[0]], {}, ('direct', 'selection')
    else:
        if release is None:
            raise ValueError('Explicit pilot and fixed-evaluation release required')
        decision = engine.read(release)
        if (decision.get('status') != 'RELEASED_DOCUMENT_EVIDENCE_V23_PILOT_AND_FIXED_EVALUATION'
                or decision.get('data_manifest_sha256') != sha256_file(data_dir/'manifest.json')
                or decision.get('evaluation_source_files') != sources()):
            raise ValueError('Pilot data/source release binding differs')
        for name in ('held_questions', 'held_gold'):
            item = manifest['files'][name]
            if sha256_file(item['path']) != item['sha256']:
                raise ValueError('Frozen fresh question/label file changed')
        questions = engine.rows(manifest['files']['held_questions']['path'])
        golds = {row['item_id']: row for row in engine.rows(manifest['files']['held_gold']['path'])}
        allowed = MODES
    modes = list(allowed if modes is None else modes)
    if not modes or len(set(modes)) != len(modes) or any(mode not in allowed for mode in modes):
        raise ValueError('Unique explicitly permitted conditions required')
    model = engine.read(binding) if binding else dict(id='B0', checkpoint=None)
    if not qualification:
        expected = decision.get('model_specs', {}).get(model['id'])
        if expected is None or any(model.get(key) != value for key, value in expected.items()):
            raise ValueError('Model lineage is absent from the frozen pilot release')
    if binding and (model.get('status') != 'AUDITED_EVIDENCE_PLAN_V23_TERMINAL256'
            or model.get('step') != 256 or sha256_file(model['checkpoint']) != model['checkpoint_sha256']
            or sha256_file(model['audit_file']) != model['audit_sha256']):
        raise ValueError('Independently audited original terminal256 required')
    assets = asset_provenance(model_dir)
    base = engine.read(base_protocol)
    refs, native = load_native_reference(base, assets)
    signature = base['base_signature']
    torch.set_num_threads(16)
    if binding:
        backend, _, loaded, signature, _ = setup_model(SimpleNamespace(
            model_dir=model_dir, checkpoint=Path(model['checkpoint']), device='cpu'))
        if any(loaded.get(key) != model[key] for key in ('run_id', 'step', 'spec_hash')):
            raise ValueError('Loaded CPU model lineage differs')
        values = []
        with torch.no_grad():
            for ref in refs:
                x = torch.full((1, 1024), 151643, dtype=torch.long)
                x[0, :len(ref['context_ids'])] = torch.tensor(ref['context_ids'])
                hidden = backend.hidden(x)
                values.append(backend.model.lm_head(hidden[:, len(ref['context_ids'])-1]).float().log_softmax(-1))
        native = torch.cat(values).cpu()
        del backend, hidden, x, values
    output.mkdir(parents=True, exist_ok=False)
    torch.save(native, output/'native-reference.pt')
    atomic_json(output/'native-contexts.json', dict(refs=refs))
    files = {name: sha256_file(output/name) for name in ('native-reference.pt', 'native-contexts.json')}
    maximum = {}
    for mode in modes:
        path = output/f'prompts-{mode}.jsonl'
        lengths = []
        with path.open('x') as stream:
            for question in questions:
                text = prompt_text(question, mode, golds.get(question['item_id']))
                ids = tok.encode(text, add_special_tokens=False)
                if len(ids) > 3584 or any(value in (151643, 151645) for value in ids):
                    raise ValueError('Complete native prompt exceeds bounds or contains a policy stop; never truncate')
                stream.write(json.dumps(dict(item_id=question['item_id'], mode=mode,
                    prompt_ids=ids, parser_mode='prior'))+'\n')
                lengths.append(len(ids))
        maximum[mode] = max(lengths)
        files[path.name] = sha256_file(path)
    runtime = dict(schema=1, environment=dict(MKL_NUM_THREADS='2', OMP_NUM_THREADS='2',
        PJRT_DEVICE='NEURON', NEURON_CC_FLAGS='--model-type=transformer',
        NEURON_NUM_RECENT_MODELS_TO_KEEP=None, XLA_COMPILATION_CACHE_SIZE='256',
        XLA_DOWNCAST_BF16=None, XLA_USE_BF16=None))
    protocol = dict(schema=SCHEMA,
        purpose='TRAIN_EVIDENCE_PLAN_MECHANICS_ONLY' if qualification else 'FIXED_HELD_EVIDENCE_PLAN_V23',
        created_utc=utc_now(), items=len(questions), modes=modes, shards=1 if qualification else 4,
        data_manifest_sha256=sha256_file(data_dir/'manifest.json'), model_dir=str(model_dir),
        model=model, assets=assets, signature=signature, source_files=sources(), files=files,
        runtime_policy=runtime, prefill_width=3584, max_actions=128,
        maximum_prompt_tokens=maximum, compute='FP32',
        question_file=None if qualification else manifest['files']['held_questions']['path'],
        question_file_sha256=None if qualification else manifest['files']['held_questions']['sha256'],
        ordered_item_ids=[q['item_id'] for q in questions], stop_ids=[151643, 151645],
        release_file=None if qualification else str(release),
        release_sha256=None if qualification else sha256_file(release),
        limits=['All responses use the same anchored Final answer parser, including direct responses.',
            'Generation opens only frozen prompt IDs and native references, never labels or annotated rows.',
            'Annotated pointer oracle adds no answer text, but evidence selection is privileged.',
            'All native actions, answer/selection failures and caps remain in denominators. No scoring here.'])
    atomic_json(output/'protocol.json', protocol)
    return protocol


def generate(output, shard, until):
    protocol = engine.read(Path(output)/'protocol.json')
    if (protocol['purpose'] not in ('TRAIN_EVIDENCE_PLAN_MECHANICS_ONLY', 'FIXED_HELD_EVIDENCE_PLAN_V23')
            or protocol['purpose'] == 'FIXED_HELD_EVIDENCE_PLAN_V23' and protocol['release_file'] is None):
        raise ValueError('Explicit TRAIN qualification or released fixed held evaluation required')
    return engine.generate(output, shard, until, schema=SCHEMA, expected_sources=sources())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare', 'generate'))
    parser.add_argument('--output', type=Path, required=True)
    for name in ('data-dir', 'model-dir', 'base-protocol', 'binding', 'release'):
        parser.add_argument('--'+name, type=Path)
    parser.add_argument('--qualification', action='store_true')
    parser.add_argument('--modes', nargs='+', choices=MODES)
    parser.add_argument('--shard', type=int)
    parser.add_argument('--until')
    args = parser.parse_args()
    if args.command == 'prepare':
        if any(path is None for path in (args.data_dir, args.model_dir, args.base_protocol)):
            parser.error('Explicit data/model/base paths required')
        prepare(args.output, args.data_dir, args.model_dir, args.base_protocol,
                qualification=args.qualification, modes=args.modes, binding=args.binding, release=args.release)
    else:
        if args.shard is None or args.until is None:
            parser.error('Explicit shard/deadline required')
        generate(args.output, args.shard, args.until)


if __name__ == '__main__':
    main()

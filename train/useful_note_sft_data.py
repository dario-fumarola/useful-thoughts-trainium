"""Causal packing for the next useful-note supervised study; no model launch.

W uses separate writer, reader and ordinary-source means. T uses ordinary
token CE on the interleaved augmented source; this is NOT an exact TPT recipe.
The caller freezes widths, interface, row inventory, weights and stop policy.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import torch

from train.interface_oracle_probe import reader_context
from weekend_thinking.artifacts import atomic_json, canonical_hash, require_utc_deadline, sha256_file, utc_now

EOS = 151643
NOTE_CAP = 128
COMPONENTS = ('source', 'writer', 'reader', 'augmented')


def writer_prompt(prefix_ids, interface, parts):
    """Prefix IDs only: this API cannot inspect a continuation or its labels."""
    past = list(prefix_ids)
    if interface == 'I0':
        return parts['old_header'] + past + parts['old_open']
    if interface in ('I1', 'I3'):
        return past + parts['old_open']
    if interface == 'I2':
        return past + parts['inline_open']
    if interface == 'I4':
        return parts['published_header'] + past + parts['old_open']
    if interface == 'I5':
        return past + parts['scratch_open']
    raise ValueError('Require a prospectively chosen reader interface I0..I5')


def symbolic_note(row):
    """An externally computed FULL calculation, not a hidden intermediate.

    Future labels choose the exact answer surface. The trained writer only sees
    writer_prompt(prefix_ids,...); label construction is supervised curation.
    """
    from train.symbolic_note_labels import teacher_note
    return teacher_note(row)


def note_ids(row, tokenizer):
    text = symbolic_note(row)
    ids = tokenizer.encode(text, add_special_tokens=False)
    decoded = tokenizer.decode(ids, skip_special_tokens=False,
                               clean_up_tokenization_spaces=False)
    if (not ids or len(ids) > NOTE_CAP or EOS in ids or decoded != text
            or tokenizer.eos_token_id != EOS):
        raise ValueError('Require a complete native calculation note within128 tokens')
    return ids


def writer_close(interface, parts):
    """The exact fixed reader close; a sampled terminator is stripped first."""
    if interface in ('I0', 'I1', 'I3', 'I4'):
        return list(parts['old_close'])
    if interface == 'I2':
        return list(parts['inline_close'])
    if interface == 'I5':
        return list(parts['scratch_close'])
    raise ValueError('Require a chosen reader interface I0..I5')


def writer_stop_ids(interface, parts):
    """One native first-close token plus base EOS; never a bare scratch ')'.

    Old interfaces close with the special151668 token, followed by a forced
    newline/space. Native I5 ')\\n' is the SINGLE token340. Thus the existing
    cached decoder suffices without changing any frozen evaluation code.
    """
    close = writer_close(interface, parts)
    if ((interface == 'I5' and close != [340])
            or (interface != 'I5' and (not close or close[0] != 151668))):
        raise ValueError('Native first-close stop contract changed')
    return (close[0], EOS)


def pack_writer(prefix_ids, note, *, interface, parts, width, target_slots):
    """Already constructed teacher labels are separate from prefix-only inputs."""
    prompt = writer_prompt(prefix_ids, interface, parts)
    stop = writer_stop_ids(interface, parts)[0]
    note = list(note)
    if not note or len(note) > NOTE_CAP or any(token in (stop, EOS) for token in note):
        raise ValueError('A complete bounded note must precede its first stop')
    ids = prompt + note + [stop]
    positions = list(range(len(prompt), len(ids)))
    return packed(ids, positions, note_positions=positions,
                  width=width, target_slots=target_slots)


def packed(ids, positions, *, note_positions, width, target_slots):
    """One fixed-width example with explicit, nonoverlapping causal labels.

    The shared CE kernel calls its non-note role ``future_mask``. For source
    and augmented CE this role covers ALL declared ordinary token labels,
    including prefix/format tokens; it does not mean only the next16 tokens.
    """
    ids = list(ids); positions = list(positions); note_positions = set(note_positions)
    if (type(width) is not int or type(target_slots) is not int
            or width < 2 or target_slots < 1 or not ids or len(ids) > width
            or len(positions) > target_slots or not positions
            or len(set(positions)) != len(positions)
            or any(type(p) is not int or not 1 <= p < len(ids) for p in positions)
            or not note_positions <= set(positions)):
        raise ValueError('Invalid/overwidth packed supervision; never truncate')
    labels = [ids[p] for p in positions]
    note = [p in note_positions for p in positions]
    future = [not value for value in note]
    inactive = target_slots - len(positions)
    return dict(input_ids=torch.tensor([ids + [EOS] * (width-len(ids))], dtype=torch.long),
        target_positions=torch.tensor([positions + [0]*inactive], dtype=torch.long),
        labels=torch.tensor([labels + [-100]*inactive], dtype=torch.long),
        note_close_mask=torch.tensor([note + [False]*inactive], dtype=torch.bool),
        future_mask=torch.tensor([future + [False]*inactive], dtype=torch.bool))


def pack_example(row, tokenizer, component, *, interface, parts, width, target_slots):
    """W reader uses all original16 targets, scalar+closure stays an endpoint.

    Reader/augmented inputs contain an external full-answer note by design.
    Writer inputs end at the sampled close token; later source tokens never enter
    that pass. T labels every real augmented token after the first, including
    inserted delimiters/reanchor; W excludes these from its reader labels.
    """
    if row.get('teacher_kind') == 'future-aware-general-firstline-native-v1':
        from train.general_hindsight_filtered_data import pack_native
        if interface != 'I3':
            raise ValueError('GENERAL native labels retain the original I3 interface')
        return pack_native(row, tokenizer, component, parts=parts,
                           width=width, target_slots=target_slots)
    if row.get('teacher_kind') is not None:
        raise ValueError('Unknown explicit teacher kind; never reinterpret as symbolic')
    if component not in COMPONENTS or row.get('split') != 'train':
        raise ValueError('A declared component and TRAIN document are required')
    past = list(row['prefix_ids']); original = list(row['original_source_ids'])
    cut = len(past)
    if (len(original) != 528 or not 128 <= cut <= 512
            or original[:cut] != past
            or row['natural_future16_ids'] != original[cut:cut+16]):
        raise ValueError('Exact original528-token source and causal cut required')
    note_positions = []
    if component == 'source':
        ids = original; positions = list(range(1, len(ids)))
    else:
        note = note_ids(row, tokenizer)
        if component == 'writer':
            return pack_writer(past, note, interface=interface, parts=parts,
                               width=width, target_slots=target_slots)
        else:
            context = reader_context(row, note, interface, parts)
            if context[:len(writer_prompt(past, interface, parts))] != writer_prompt(past, interface, parts):
                raise ValueError('Writer prefix and chosen reader serialization differ')
            suffix = original[cut:cut+16] if component == 'reader' else original[cut:]
            ids = context + suffix
            positions = (list(range(len(context), len(ids))) if component == 'reader'
                         else list(range(1, len(ids))))
    return packed(ids, positions, note_positions=note_positions,
                  width=width, target_slots=target_slots)


def load_candidate_rows(directory, manifest, teacher_kind):
    """Resolve a frozen TRAIN pool without selecting on likelihood outcomes.

    New union pools contain true distinct (document,source-cut) examples. Their
    priority is compound representatives, additional compound cuts, then the
    corresponding one-operation groups. Repeated documents are reported rather
    than disguised as independent documents. Old one-per-document pools retain
    their original order and identity.
    """
    from train.symbolic_note_labels import FULL, POSTORDER
    if teacher_kind not in (FULL, POSTORDER):
        raise ValueError('Freeze one explicit supported teacher-note kind')
    directory = Path(directory)
    if manifest.get('schema') in ('finemath-compound-anchor-candidates-v1',
                                   'compound-priority-equation-candidates-v1'):
        if manifest.get('status') != 'COMPLETE_CPU_CANDIDATE_INVENTORY_NOT_TRAINING_RELEASE':
            raise ValueError('Complete closed candidate publication required')
        source = Path(manifest['candidates_file'])
        if sha256_file(source) != manifest['candidates_sha256']:
            raise ValueError('Immutable union candidate publication changed')
        wrappers = [json.loads(line) for line in source.read_text().splitlines() if line.strip()]
        if (len(wrappers) != manifest['candidates']
                or any(w['split'] != 'train' for w in wrappers)
                or len({w['candidate_id'] for w in wrappers}) != len(wrappers)
                or len({(w['document_id'], w['absolute_source_cut']) for w in wrappers}) != len(wrappers)):
            raise ValueError('Unique TRAIN source-cut candidates required')
        wrappers.sort(key=lambda w: (w['binary_ops'] < 2, not w['representative']))
        rows = []
        for wrapper in wrappers:
            original = wrapper['source_payload']
            if (original['split'] != 'train' or original['document_id'] != wrapper['document_id']
                    or canonical_hash(original) != wrapper['source_payload_sha256']):
                raise ValueError('Candidate source payload/provenance changed')
            row = dict(original, original_item_id=original['item_id'],
                item_id=wrapper['candidate_id'], absolute_source_cut=wrapper['absolute_source_cut'],
                teacher_note_kind=teacher_kind,
                candidate_provenance={k:v for k,v in wrapper.items() if k != 'source_payload'})
            rows.append(row)
        return rows, [source], 'Compound representatives, other compound cuts, one-op representatives, other one-op cuts; stable frozen hash order within groups'
    accepted = directory/'accepted.jsonl'
    if sha256_file(accepted) != manifest['accepted_sha256']:
        raise ValueError('Immutable complete curator publication changed')
    pool = manifest['selection']['train']
    order = pool['selected_item_ids'] + pool['reserve_item_ids']
    wanted = set(order); selected = {}
    with accepted.open() as stream:
        for line in stream:
            row = json.loads(line)
            if row['item_id'] in wanted:
                if row['split'] != 'train' or row['item_id'] in selected:
                    raise ValueError('Unique TRAIN candidates required')
                selected[row['item_id']] = dict(row, teacher_note_kind=teacher_kind)
    if len(selected) != len(order) or len({r['document_id'] for r in selected.values()}) != len(order):
        raise ValueError('Complete unique-document original TRAIN pool required')
    return [selected[key] for key in order], [accepted], 'Original curator selected+reserve order, one example/document'


def prepare_candidates(directory, output, model_dir, deadline_utc,
                       teacher_kind='full-calculation-v1'):
    """TRAIN-only symbolic labels, before interface choice or model filtering.

    All candidate/rejection records stay in the output. This is NOT the final
    positive-gain training inventory and authorizes no optimizer updates.
    """
    from transformers import AutoTokenizer
    from train.interface_oracle_probe import serialization
    directory = Path(directory); output = Path(output)
    deadline = require_utc_deadline(deadline_utc)
    def check():
        if datetime.now(timezone.utc) >= deadline:
            raise TimeoutError('Finite candidate preparation deadline; partials retained')
    check()
    if output.exists():
        raise FileExistsError(output)
    manifest_file = directory/'manifest.json'
    manifest = json.loads(manifest_file.read_text())
    rows, source_files, order_rule = load_candidate_rows(directory, manifest, teacher_kind)
    order = [row['item_id'] for row in rows]
    source_pins = {str(p.resolve()):sha256_file(p) for p in [manifest_file]+source_files}
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True,
                                              trust_remote_code=False)
    fmt = json.loads(Path('local/resolved_format.json').read_text())
    published = json.loads(Path('configs/rlp_published_prompt.json').read_text())
    parts = serialization(tokenizer, dict(fmt['format'], header_text=fmt['header_text']), published)
    widths = {i:{c:dict(real_tokens_max=0,active_labels_max=0,active_labels_total=0)
                 for c in COMPONENTS} for i in (f'I{k}' for k in range(6))}
    output.mkdir(parents=True)
    count = rejected = 0; retained_documents=set()
    with (output/'candidates.jsonl').open('x') as candidates, (output/'audit.jsonl').open('x') as audit:
        for row in rows:
            check(); key = row['item_id']
            try:
                note = note_ids(row, tokenizer)
            except ValueError as error:
                rejected += 1
                audit.write(json.dumps(dict(item_id=key,document_id=row['document_id'],
                    status='label_rejected',reason=str(error)))+'\n')
                continue
            annotated = dict(row,teacher_note_text=symbolic_note(row),teacher_note_ids=note)
            candidates.write(json.dumps(annotated,allow_nan=False)+'\n'); count += 1
            retained_documents.add(row['document_id'])
            audit.write(json.dumps(dict(item_id=key,document_id=row['document_id'],
                                        status='label_accepted',note_tokens=len(note)))+'\n')
            for interface in widths:
                prompt = writer_prompt(row['prefix_ids'],interface,parts)
                context = reader_context(row,note,interface,parts)
                measures = dict(source=(528,527),writer=(len(prompt)+len(note)+1,len(note)+1),
                    reader=(len(context)+16,16),
                    augmented=(len(context)+528-len(row['prefix_ids']),
                               len(context)+527-len(row['prefix_ids'])))
                for component,(tokens,labels) in measures.items():
                    record = widths[interface][component]
                    record['real_tokens_max'] = max(record['real_tokens_max'],tokens)
                    record['active_labels_max'] = max(record['active_labels_max'],labels)
                    record['active_labels_total'] += labels
    check()
    if any(sha256_file(name) != digest for name,digest in source_pins.items()):
        raise ValueError('Source publication changed during preparation')
    result = dict(schema='symbolic-calculation-note-candidates-v1',status='complete_CPU_labels_NOT_final_training_inventory',
        created_utc=utc_now(),candidate_count=count,label_rejections=rejected,attempted=len(order),
        unique_documents=len(retained_documents),
        attempted_unique_documents=len({row['document_id'] for row in rows}),teacher_note_kind=teacher_kind,
        candidate_order_rule=order_rule,
        original_order_sha256=canonical_hash(order),all_conditions_widths=widths,parts=parts,
        candidates_file=str((output/'candidates.jsonl').resolve()),
        candidates_sha256=sha256_file(output/'candidates.jsonl'),audit_sha256=sha256_file(output/'audit.jsonl'),
        input_hashes={**source_pins,**{str(p.resolve()):sha256_file(p) for p in
            (Path('local/resolved_format.json'),Path('configs/rlp_published_prompt.json'))}},
        tokenizer_vocab_sha256=canonical_hash(tokenizer.get_vocab()),source_sha256=sha256_file(__file__),
        label_source_sha256=sha256_file('train/symbolic_note_labels.py'),
        supervision='External FULL symbolic calculation, source-matched answer surface; policy inputs remain prefix-only',
        next='Select interface from complete Phase1, frozen-B0 full16 gain filtering versus empty; save all attempts and acceptance. No training launch.',
        limitations=['TRAIN-only teacher labels; not a deployed policy or spontaneous reasoning result.',
            'No likelihood filtering or final5000-example quota has been satisfied by this preparation alone.',
            'W writer samples the first native close token; remaining close text/reanchor are fixed formatting.',
            'T is interleaved ordinary CE, not the published document-then-trajectory TPT recipe.'])
    atomic_json(output/'manifest.json',result)
    return result


def prepare_training(filter_output, output, model_dir, deadline_utc):
    """Verify the full positive inventory and freeze compact masks/shapes.

    CPU qualification only; all arms share this exact row inventory. Actual
    BF16 compute, optimizer/master state and distributed updates qualify later.
    """
    from transformers import AutoTokenizer
    from train import filter_symbolic_notes as filtering
    filter_output=Path(filter_output);output=Path(output)
    deadline=require_utc_deadline(deadline_utc)
    def check():
        if datetime.now(timezone.utc)>=deadline:
            raise TimeoutError('Finite final-inventory preparation; partials retained')
    check();p,x=filtering.load(filter_output)
    summary=json.loads((filter_output/'summary.json').read_text())
    training=filter_output/'training.jsonl'
    if (summary['status']!='complete_filter_training_ready'
            or summary['requested_training_items']!=5120 or summary['positive']<5120
            or summary['protocol_sha256']!=sha256_file(filter_output/'protocol.json')
            or summary['training_sha256']!=sha256_file(training)
            or summary['decisions_sha256']!=sha256_file(filter_output/'decisions.json')):
        raise ValueError('A complete5120 positive TRAIN inventory is required')
    for name,digest in summary['score_hashes'].items():
        if sha256_file(name)!=digest:
            raise ValueError('Source filter scores changed before final preparation')
    if Path(model_dir).resolve()!=Path(p['model_dir']).resolve():
        raise ValueError('Use the same released native tokenizer as the frozen filter')
    rows=[json.loads(s) for s in training.read_text().splitlines() if s.strip()]
    ids=[r['item_id'] for r in rows];docs=[r['document_id'] for r in rows]
    decisions=json.loads((filter_output/'decisions.json').read_text())['rows']
    accepted=[v['item_id'] for v in decisions if v['teacher_minus_empty']>0]
    if (len(rows)!=5120 or len(set(ids))!=5120
            or len({(r['document_id'],r.get('absolute_source_cut',r['item_id'])) for r in rows})!=5120
            or ids!=summary['selected_item_ids'] or ids!=accepted[:5120]
            or any(r['split']!='train' for r in rows)):
        raise ValueError('No selection, replacement, duplicate source cuts or split changes')
    tok=AutoTokenizer.from_pretrained(model_dir,local_files_only=True,trust_remote_code=False)
    interface=p['interface'];parts=x['parts'];maxima={c:[0,0] for c in COMPONENTS}
    for row in rows:
        check();note=note_ids(row,tok)
        if note!=row['teacher_note_ids']:raise ValueError('Teacher note changed after filtering')
        prompt=writer_prompt(row['prefix_ids'],interface,parts)
        context=reader_context(row,note,interface,parts)
        measures=dict(source=(528,527),writer=(len(prompt)+len(note)+1,len(note)+1),
            reader=(len(context)+16,16),augmented=(len(context)+528-len(row['prefix_ids']),
                                                len(context)+527-len(row['prefix_ids'])))
        for c,(length,labels) in measures.items():
            maxima[c]=[max(maxima[c][0],length),max(maxima[c][1],labels)]
    round32=lambda value:((value+31)//32)*32
    packing={c:dict(width=round32(length),target_slots=16 if c=='reader' else round32(labels))
             for c,(length,labels) in maxima.items()}
    counts={c:dict(examples=0,active_labels=0,note_close_labels=0,non_note_labels=0) for c in COMPONENTS}
    for row in rows:
        check()
        for c in COMPONENTS:
            batch=pack_example(row,tok,c,interface=interface,parts=parts,**packing[c])
            note=batch['note_close_mask'];future=batch['future_mask'];active=note|future
            if ((note&future).any() or not active.any() or (batch['labels'][~active]!=-100).any()
                    or not torch.equal(batch['labels'][active],batch['input_ids'].gather(1,batch['target_positions'])[active])):
                raise ValueError('Invalid native shifted labels or component masks')
            record=counts[c];record['examples']+=1;record['active_labels']+=int(active.sum())
            record['note_close_labels']+=int(note.sum());record['non_note_labels']+=int(future.sum())
    check();output.mkdir(parents=True,exist_ok=False)
    (output/'training.jsonl').write_bytes(training.read_bytes())
    inputs={str(f.resolve()):sha256_file(f) for f in
        (filter_output/'protocol.json',filter_output/'summary.json',filter_output/'decisions.json',training)}
    result=dict(schema='useful-note-positive-train-v2',status='complete_CPU_inventory_NOT_Neuron_training_qualification',
        created_utc=utc_now(),purpose='science',items=5120,unique_documents=len(set(docs)),
        split='train',interface=interface,parts=parts,packing=packing,
        component_counts=counts,ordered_item_ids_sha256=canonical_hash(ids),ordered_document_ids_sha256=canonical_hash(docs),
        training_file=str((output/'training.jsonl').resolve()),training_sha256=sha256_file(output/'training.jsonl'),
        tokenizer_vocab_sha256=canonical_hash(tok.get_vocab()),input_hashes=inputs,
        filter_input_hashes=inputs,source_files=filtering.sources(),
        writer_stop_ids=list(writer_stop_ids(interface,parts)),reader_target='Original natural16, not only scalar endpoint',
        filter_target='Complete original numeral plus actual closing-boundary tokens',
        mask_semantics='Legacy future_mask is the non-note role: source/augmented label all declared ordinary tokens; reader labels only natural16.',
        limitations=['Future-aware supervised teacher-positive TRAIN selection.',
            'Same documents/updates do not imply matched supervision, gradient dose, tokens or latency.',
            'T is an interleaved ordinary-CE ablation, not an exact TPT reproduction.',
            'Teacher-note utility does not establish prefix-generated note utility.'])
    atomic_json(output/'manifest.json',result);return result


def prepare_qualification(candidates, output, model_dir, deadline_utc, interface='I3'):
    """First64 real TRAIN labels, with shapes measured on the complete pool.

    A declared two-update hardware qualification is not a scientific training
    arm and does not pretend to satisfy the positive-filtered 5000-example gate.
    This subset can run while the separate frozen-reader filter is in progress.
    """
    from transformers import AutoTokenizer
    from train import filter_symbolic_notes as filtering
    candidates=Path(candidates);output=Path(output)
    deadline=require_utc_deadline(deadline_utc)
    if datetime.now(timezone.utc)>=deadline:raise TimeoutError('Finite qualification preparation')
    cm=json.loads((candidates/'manifest.json').read_text());source=candidates/'candidates.jsonl'
    if (cm['status']!='complete_CPU_labels_NOT_final_training_inventory'
            or sha256_file(source)!=cm['candidates_sha256']
            or cm['candidate_count']<64 or output.exists()):
        raise ValueError('Require a complete immutable real TRAIN label pool and new output')
    tok=AutoTokenizer.from_pretrained(model_dir,local_files_only=True,trust_remote_code=False)
    rows=[json.loads(line) for line in source.read_text().splitlines() if line.strip()][:64]
    if any(r['split']!='train' or note_ids(r,tok)!=r['teacher_note_ids'] for r in rows):
        raise ValueError('Qualification labels changed or left TRAIN')
    ids=[r['item_id'] for r in rows];docs=[r['document_id'] for r in rows]
    if len(set(ids))!=64 or len({(r['document_id'],r.get('absolute_source_cut',r['item_id'])) for r in rows})!=64:
        raise ValueError('Distinct qualification source cuts required')
    round32=lambda value:((value+31)//32)*32
    packing={c:dict(width=round32(v['real_tokens_max']),
                    target_slots=16 if c=='reader' else round32(v['active_labels_max']))
             for c,v in cm['all_conditions_widths'][interface].items()}
    counts={c:dict(examples=0,active_labels=0,note_close_labels=0,non_note_labels=0) for c in COMPONENTS}
    for row in rows:
        for c in COMPONENTS:
            batch=pack_example(row,tok,c,interface=interface,parts=cm['parts'],**packing[c])
            note=batch['note_close_mask'];future=batch['future_mask'];active=note|future
            if (note&future).any() or not torch.equal(batch['labels'][active],
                    batch['input_ids'].gather(1,batch['target_positions'])[active]):
                raise ValueError('Invalid qualification shifted labels/masks')
            counts[c]['examples']+=1;counts[c]['active_labels']+=int(active.sum())
            counts[c]['note_close_labels']+=int(note.sum());counts[c]['non_note_labels']+=int(future.sum())
    if datetime.now(timezone.utc)>=deadline:raise TimeoutError('Finite qualification preparation')
    output.mkdir(parents=True)
    with (output/'training.jsonl').open('x') as stream:
        for row in rows:stream.write(json.dumps(row,allow_nan=False)+'\n')
    inputs={str(p.resolve()):sha256_file(p) for p in (candidates/'manifest.json',source)}
    result=dict(schema='useful-note-positive-train-v2',status='complete_CPU_inventory_NOT_Neuron_training_qualification',
        purpose='qualification',created_utc=utc_now(),items=64,unique_documents=len(set(docs)),split='train',
        interface=interface,parts=cm['parts'],packing=packing,component_counts=counts,
        ordered_item_ids_sha256=canonical_hash(ids),ordered_document_ids_sha256=canonical_hash(docs),
        training_file=str((output/'training.jsonl').resolve()),training_sha256=sha256_file(output/'training.jsonl'),
        tokenizer_vocab_sha256=canonical_hash(tok.get_vocab()),source_files=filtering.sources(),
        input_hashes=inputs,filter_input_hashes=inputs,writer_stop_ids=list(writer_stop_ids(interface,cm['parts'])),
        selection='First64 fixed candidate rows; no gain/outcome ranking. Global shapes measured on ALL candidate labels.',
        reader_target='Original natural16, not only scalar endpoint',
        mask_semantics='Legacy future_mask is the non-note role: source/augmented label all declared ordinary tokens; reader labels only natural16.',
        limitations=['Hardware-only two-update qualification; this is not a positive-filtered scientific arm.',
            'Some supplied notes may be unhelpful to frozen B0; qualification does not measure note utility.',
            'Global-pool shapes may be larger than eventual selected science shapes.'])
    atomic_json(output/'manifest.json',result);return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-dir',type=Path)
    parser.add_argument('--filter-output',type=Path)
    parser.add_argument('--qualification-candidates',type=Path)
    for name in ('output','model-dir'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--deadline-utc',required=True)
    parser.add_argument('--teacher-kind',choices=('full-calculation-v1','exact-postorder-v1'),
                        default='full-calculation-v1')
    args = parser.parse_args()
    if sum(bool(x) for x in (args.dataset_dir,args.filter_output,args.qualification_candidates))!=1:
        parser.error('Supply exactly one input pool, filter output or qualification candidate pool')
    if args.qualification_candidates:
        result=prepare_qualification(args.qualification_candidates,args.output,args.model_dir,args.deadline_utc)
        print(json.dumps(dict(items=result['items'],purpose=result['purpose'],packing=result['packing'])))
    elif args.filter_output:
        result=prepare_training(args.filter_output,args.output,args.model_dir,args.deadline_utc)
        print(json.dumps(dict(items=result['items'],packing=result['packing'])))
    else:
        result = prepare_candidates(args.dataset_dir,args.output,args.model_dir,args.deadline_utc,args.teacher_kind)
        print(json.dumps(dict(candidates=result['candidate_count'],rejected=result['label_rejections'])))


if __name__ == '__main__':
    main()

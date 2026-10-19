"""Separate immutable prefix questions, likelihood targets and scalar gold."""
from pathlib import Path
import hashlib
import json
import os
import shutil
from transformers import AutoTokenizer
from train.real_trace_notes import ROOT
from train.real_trace_contexts import question_projection, writer_prompt, reader_context, markers, boundary_stop_ids
from train.real_trace_packing import prefix_calculation

R = Path('/home/ubuntu/dario')
W = R / 'local/worktrees/real-source-eval-v17'
assert ROOT == W
sha = lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
held = R / 'local/nvme-study/data/real-source-balanced-v17-confirmation-v1'
design = json.loads((R/'validation/real_source_balanced_v17_design_20261019.json').read_text())
assert sha(held/'manifest.json') == design['evaluation']['held_manifest_sha256']
assert sha(held/'DEV_confirm512.jsonl') == design['evaluation']['data_files']['DEV_confirm512.jsonl']['sha256']
tok = AutoTokenizer.from_pretrained(R/'local/assets/qwen3-1.7b-base',local_files_only=True)
output = R / 'local/nvme-cpt/data/real-source-balanced-v17-evaluation-inputs-v1'
gold_output = R / 'local/nvme-cpt/data/real-source-balanced-v17-scoring-gold-v1'
output.mkdir(exist_ok=False)
gold_output.mkdir(exist_ok=False)
rows = [json.loads(v) for v in (held/'DEV_confirm512.jsonl').read_text().splitlines()]
assert len(rows) == 512
questions, targets, controls, gold, unsupported = [], [], [], [], []
for row in rows:
    assert row['split'] == 'followup_dev'
    q = question_projection(row)
    assert set(q) == {'item_id','document_id','prefix_ids'} and len(writer_prompt(q,tok)) <= 544
    text,note = prefix_calculation(q['prefix_ids'],tok)
    assert text == row['full_calculation_text'] and note == row['full_calculation_ids']
    assert len(reader_context(q,tok,note))+16 <= 672
    y = dict(item_id=q['item_id'],document_id=q['document_id'],target16=row['continuation16_ids'])
    ids = y['target16']
    assert len(ids) == 16
    # Native tokens may merge leading space with a minus sign. Align decoded
    # character boundaries instead of retokenizing the literal in isolation.
    decoded = tok.decode(ids,skip_special_tokens=False,clean_up_tokenization_spaces=False)
    stripped = decoded.lstrip(' \t\r\n')
    literal = row['numeric_surface_text']
    try:
        assert stripped.startswith(literal)
        left = len(decoded)-len(stripped)
        right = left+len(literal)
        ends = [0]+[len(tok.decode(ids[:i],skip_special_tokens=False,clean_up_tokenization_spaces=False)) for i in range(1,17)]
        begin = next(i for i in range(16) if ends[i+1]>left)
        end = next(i for i in range(1,17) if ends[i]>=right)
        close = row['closing_boundary_ids']
        found = [i for i in range(end,16)
                 if not tok.decode(ids[end:i],skip_special_tokens=False,clean_up_tokenization_spaces=False).strip()
                 and ids[i:i+len(close)] == close]
        assert found
        stop = found[0]+len(close)
        y.update(component_status='exact_native_partition',component_ranges=dict(
            leading=[0,begin],numeric=[begin,end],closing=[end,stop],remainder=[stop,16]))
    except (AssertionError,StopIteration):
        y['component_status'] = 'unsupported_static_native_partition'
        unsupported.append(q['item_id'])
    questions.append(q)
    targets.append(y)
    controls.append(dict(item_id=q['item_id'],oracle_ids=note))
    gold.append(dict(item_id=q['item_id'],document_id=q['document_id'],numeric_value=row['numeric_value']))

files = {}
for kind,values in [('questions',questions),('targets',targets),('controls',controls)]:
    path = output/f'DEV_confirm512-{kind}.jsonl'
    path.write_text(''.join(json.dumps(v,separators=(',',':'))+'\n' for v in values))
    files[path.name] = dict(bytes=path.stat().st_size,sha256=sha(path))
ordinary_file=held/'DEV_ordinary256.jsonl'
assert sha(ordinary_file)==design['evaluation']['data_files']['DEV_ordinary256.jsonl']['sha256']
ordinary_rows=[json.loads(v) for v in ordinary_file.read_text().splitlines()]
assert len(ordinary_rows)==256
ordinary_questions=[question_projection(row) for row in ordinary_rows]
ordinary_targets=[dict(item_id=row['item_id'],document_id=row['document_id'],target16=row['continuation16_ids'],component_status='not_applicable') for row in ordinary_rows]
for kind,values in [('questions',ordinary_questions),('targets',ordinary_targets),('controls',[])]:
    path=output/f'DEV_ordinary256-{kind}.jsonl'
    path.write_text(''.join(json.dumps(v,separators=(',',':'))+'\n' for v in values))
    files[path.name]=dict(bytes=path.stat().st_size,sha256=sha(path))
manifest = dict(status='FIXED_PREFIX_ONLY_QUESTIONS_AND_SEPARATE_SCORING_TARGETS',
    schema='real-source-balanced-v17-evaluation-inputs-v1',source_script_sha256=sha(__file__),
    source_held_manifest_sha256=sha(held/'manifest.json'),
    panels=dict(DEV_confirm512=dict(documents=512,source_sha256=sha(held/'DEV_confirm512.jsonl'),
                arithmetic=True,component_unsupported=len(unsupported),unsupported_item_ids=unsupported),
                DEV_ordinary256=dict(documents=256,source_sha256=sha(ordinary_file),arithmetic=False,fresh_source_confirmation=True)),
    files=files,opening_ids=markers(tok)[0],closing_ids=markers(tok)[1],stop_ids=boundary_stop_ids(tok),
    max_writer_prefill=544,max_writer_actions=128,likelihood_width=672,
    source_files={name:sha(W/name) for name in ['train/real_trace_contexts.py','train/real_trace_packing.py',
        'train/real_trace_evaluation.py','train/real_trace_scope.py']},
    limits=['Generation question files contain onlyid/document/nativeprefix.',
            'Every512 document is retained even if static component alignment is unsupported.',
            'Numeral tokens include any merged leading whitespace;boundary includes intervening whitespace before the frozen curator cue.',
            'Scalar initial-scalar-v2 parser is unchanged;this prospective token accounting is a different secondary diagnostic.',
            'Ordinary256 is a new source-TRAIN-derived FineMath panel;not a broad-domain or officialTEST result.',
            'Oracle annotations are prefix-computable full traces including root,not deployed policy observations.'])
(output/'manifest.json').write_text(json.dumps(manifest,indent=2))
gold_path = gold_output/'DEV_confirm512-gold.jsonl'
gold_path.write_text(''.join(json.dumps(v,separators=(',',':'))+'\n' for v in gold))
(gold_output/'manifest.json').write_text(json.dumps(dict(status='SCORING_ONLY_GOLD_NO_GENERATOR_ACCESS',
    files={gold_path.name:dict(bytes=gold_path.stat().st_size,sha256=sha(gold_path))},
    source_held_manifest_sha256=sha(held/'manifest.json')),indent=2))
A = R / 'local/backups/data/real-source-balanced-v17-evaluation-inputs-v1'
A.mkdir(exist_ok=False)
archived = {}
for folder in [output,gold_output]:
    for src in folder.iterdir():
        dst = A/folder.name/src.name
        dst.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(src,dst)
        with dst.open('rb') as stream:
            os.fsync(stream.fileno())
        assert sha(src) == sha(dst)
        archived[str(dst.relative_to(A))] = dict(bytes=src.stat().st_size,sha256=sha(src))
shutil.copyfile(__file__,A/Path(__file__).name)
(A/'COMPLETE.json').write_text(json.dumps(dict(status='PASS_EVAL_INPUTS_AND_SEPARATE_GOLD_READBACK_FSYNC',files=archived),indent=2))
print(json.dumps(manifest,indent=2),flush=True)

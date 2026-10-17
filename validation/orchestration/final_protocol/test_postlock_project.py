"""Synthetic stdlib fixtures only; never inspect real final datasets/models."""
import copy
from datetime import datetime, timedelta, timezone
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch


SPEC = importlib.util.spec_from_file_location('postlock_project_fixture_target', Path(__file__).with_name('postlock_project.py'))
p = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(p)

PURE = '''raise RuntimeError("The model module must never be imported")
def arc_context(question: str) -> str:
    return f"Question: {question}\\nAnswer:"
def lambada_parts(text: str) -> tuple[str, str]:
    parts = text.rsplit(" ", 1)
    if len(parts) != 2 or not parts[0].strip() or not parts[1].strip():
        raise ValueError("Expected LAMBADA context and a nonempty final word")
    return parts[0], " " + parts[1]
'''
SERIALIZE = '''def jsonl_bytes(rows):
    return "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":"))+"\\n" for row in rows).encode()
'''
DEPENDENCIES = '''def evaluation_dependencies(root=None):
    return {name:sha256_file(root/name) for name in ("train/evaluate_likelihood.py", "train/evaluation_integrity.py", "weekend_thinking/numeric_projection.py")}
'''


class ProjectionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='postlock-project-unit-')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root/'synthetic-source'
        for name, text in [('train/evaluate_likelihood.py', PURE), ('train/evaluation_integrity.py', DEPENDENCIES),
                           ('weekend_thinking/numeric_projection.py', SERIALIZE)]:
            file = self.source/name
            file.parent.mkdir(parents=True, exist_ok=True)
            file.write_text(text)
        files = {str(file.relative_to(self.source)):p.sha(file) for file in self.source.rglob('*.py')}
        source = dict(files=files, sha256=p.canonical(files))
        self.manifest = self.root/'synthetic-source-manifest.json'
        self.manifest.write_text(json.dumps(dict(schema='science-source-archive-v1', archive_dir=str(self.source), source=source)))
        self.canonical = {task:self.root/(task+'.synthetic') for task in p.PINNED}
        for file in self.canonical.values():
            file.write_bytes(b'Synthetic bytes only, never a real dataset')
        end = (datetime.now(timezone.utc)+timedelta(days=1)).isoformat()
        spec = dict(status='FROZEN_ASSETS_NOT_HARDWARE_QUALIFIED', terminal_steps=256, deadline_utc=end,
            site=dict(world_size_per_arm=32, training_run_order=['C1234','C5678','R1234','R5678']),
            proposal=dict(optimization=dict(seeds=[1234,5678], compute_dtype='fp32')),
            assets=dict(model_manifest_sha256='b'*64, model_id='fixture/model', model_revision='c'*40), source=source)
        spec['spec_hash'] = p.canonical(spec)
        common = dict(base_assets_sha256='a'*64, model_manifest_sha256='b'*64, model_id='fixture/model', model_revision='c'*40)
        models = [dict(common, arm='B0', seed=None)]
        for arm, letter in [('vanilla','C'), ('thought','R')]:
            for seed in (1234,5678):
                models.append(dict(common, arm=arm, seed=seed, spec_hash=spec['spec_hash'], step=256,
                    run_id=letter+str(seed), order_hash=('d' if seed==1234 else 'e')*64,
                    model_sha256='1'*64, complete_sha256='2'*64))
        self.lock = dict(schema='final-evaluation-pre-gold-lock-v1', status='locked', training_spec=spec,
            models=models, model_inventory_sha256=p.canonical(models),
            models_by_label={'B0' if model['arm']=='B0' else model['run_id']+'@256':model for model in models},
            source_root=str(self.source), source_manifest_file=str(self.manifest), source_manifest_sha256=p.sha(self.manifest),
            hard_deadline_utc=end, projector_sha256=p.sha(Path(p.__file__)),
            canonical_sources={task:dict(info, path=str(self.canonical[task])) for task,info in p.PINNED.items()})
        self.lock_file = self.root/'synthetic-input.json'
        self.helpers = p.pure_definitions(self.source/'train/evaluate_likelihood.py', ('arc_context','lambada_parts'))

    def write_lock(self, lock=None):
        self.lock_file.write_text(json.dumps(self.lock if lock is None else lock))
        return p.sha(self.lock_file)

    def reject_before_io(self, lock=None, expected=None):
        lock_hash = self.write_lock(lock) if expected is None else expected
        original_sha = p.sha
        def guarded_sha(path):
            self.assertNotIn(Path(path), self.canonical.values(), 'Canonical bytes accessed before lock rejection')
            return original_sha(path)
        with patch.object(p, 'read_table', side_effect=AssertionError('Canonical arrays must remain unread')) as reader, \
             patch.object(p, 'sha', side_effect=guarded_sha):
            with self.assertRaises((ValueError, KeyError, FileNotFoundError)):
                p.execute(self.lock_file, lock_hash, 'arc_challenge', self.root/'must-not-exist')
            reader.assert_not_called()
        self.assertFalse((self.root/'must-not-exist').exists())

    def test_nonstandard_arc_labels_preserve_order_and_map_answer(self):
        row = dict(id='question-X', question='Which?', choices=dict(label=['D','A','C','B'],
            text=['delta',' already spaced','gamma','beta']), answerKey='C')
        questions, gold, orders = p.project_rows('arc_challenge', [row], self.helpers)
        self.assertEqual(questions[0]['context'], 'Question: Which?\nAnswer:')
        self.assertEqual(questions[0]['choices'], [' delta','  already spaced',' gamma',' beta'])
        self.assertEqual(gold, [dict(item_id='arc_challenge:test:question-X', gold=2)])
        self.assertEqual(orders[0]['labels'], ['D','A','C','B'])

    def test_lambada_literal_space_boundary_and_unicode_are_unchanged(self):
        questions, gold, orders = p.project_rows('lambada', [dict(text='Opening\tline\nnext café'),dict(text='A  final!')], self.helpers)
        self.assertEqual([row['context'] for row in questions], ['Opening\tline\nnext','A '])
        self.assertEqual([row['target'] for row in gold], [' café',' final!'])
        self.assertEqual([row['item_id'] for row in questions], ['lambada:test:00000','lambada:test:00001'])
        self.assertEqual(orders, [])

    def test_malformed_rows_are_fatal_not_dropped(self):
        good = dict(id='x', question='Question?', choices=dict(label=['C','A'], text=['third','first']), answerKey='A')
        bad = [dict(good, answerKey='missing'), dict(good, choices=dict(label=['A','A'],text=['one','two'])),
               dict(good, choices=dict(label=['C','A'],text=['one'])), dict(good, question='')]
        for row in bad:
            with self.subTest(row=row), self.assertRaises(ValueError):
                p.project_rows('arc_challenge', [good,row], self.helpers)
        with self.assertRaises(ValueError):
            p.project_rows('arc_challenge', [good,good], self.helpers)
        for text in ['singleword','context trailing ','   word']:
            with self.subTest(text=text), self.assertRaises(ValueError):
                p.project_rows('lambada', [dict(text='valid word'),dict(text=text)], self.helpers)

    def test_complete_synthetic_lock_and_source_closure_validate_without_arrays(self):
        self.assertEqual(p.validate_lock(self.lock_file, self.write_lock())['models'], self.lock['models'])

    def test_missing_lock_fails_before_canonical_io(self):
        self.reject_before_io(expected='0'*64)

    def test_wrong_explicit_hash_and_unlocked_status_fail_before_io(self):
        self.write_lock()
        self.reject_before_io(expected='0'*64)
        changed = copy.deepcopy(self.lock);changed['status']='draft'
        self.reject_before_io(changed)

    def test_missing_early_duplicate_or_wrong_spec_models_fail_before_io(self):
        for change in ['missing','early','duplicate','wrong-spec','b0-seed']:
            lock=copy.deepcopy(self.lock)
            if change=='missing':lock['models'].pop()
            elif change=='early':lock['models'][1]['step']=64
            elif change=='duplicate':lock['models'][-1]=copy.deepcopy(lock['models'][1])
            elif change=='wrong-spec':lock['models'][1]['spec_hash']='f'*64
            else:lock['models'][0]['seed']=1234
            with self.subTest(change=change):self.reject_before_io(lock)

    def test_wrong_projector_source_revision_and_changed_source_fail_before_io(self):
        changed=copy.deepcopy(self.lock);changed['projector_sha256']='0'*64
        self.reject_before_io(changed)
        changed=copy.deepcopy(self.lock);changed['canonical_sources']['arc_challenge']['revision']='0'*40
        self.reject_before_io(changed)
        (self.source/'train/evaluate_likelihood.py').write_text(PURE+'\n# changed\n')
        self.reject_before_io()

    def test_expired_deadline_fails_before_io(self):
        changed=copy.deepcopy(self.lock)
        end=(datetime.now(timezone.utc)-timedelta(seconds=1)).isoformat()
        changed['hard_deadline_utc']=changed['training_spec']['deadline_utc']=end
        changed['training_spec']['spec_hash']=p.canonical({k:v for k,v in changed['training_spec'].items() if k!='spec_hash'})
        self.reject_before_io(changed)

    def test_full_fixture_projection_is_canonical_and_never_overwrites(self):
        rows=[dict(id='id1',question='First?',choices=dict(label=['D','B'],text=['delta','beta']),answerKey='B'),
              dict(id='id2',question='Second?',choices=dict(label=['2','1'],text=['two','one']),answerKey='2')]
        pinned={task:dict(info,sha256=p.sha(self.canonical[task]),items=2) for task,info in p.PINNED.items()}
        lock=copy.deepcopy(self.lock)
        lock['canonical_sources']={task:dict(info,path=str(self.canonical[task])) for task,info in pinned.items()}
        class Table:
            def to_pylist(self):return rows
        output=self.root/'synthetic-projection'
        with patch.dict(p.PINNED,pinned,clear=True),patch.object(p,'read_table',return_value=Table()) as reader:
            result=p.execute(self.lock_file,self.write_lock(lock),'arc_challenge',output)
            reader.assert_called_once_with(self.canonical['arc_challenge'],columns=['id','question','choices','answerKey'])
            self.assertEqual(result['items'],2);self.assertEqual(result['rows_dropped'],0)
            self.assertEqual(result['questions_sha256'],p.sha(output/'questions.jsonl'))
            self.assertEqual(result['gold_sha256'],p.sha(output/'gold.jsonl'))
            self.assertEqual(result['pre_gold_lock_sha256'],p.sha(self.lock_file))
            self.assertEqual(json.loads((output/'gold.jsonl').read_text().splitlines()[0])['gold'],1)
            self.assertEqual(result['candidate_order'][0]['labels'],['D','B'])
            self.assertEqual(set(result['dependencies']),set(self.lock['training_spec']['source']['files']))
            reader.reset_mock()
            with self.assertRaisesRegex(ValueError,'New output'):
                p.execute(self.lock_file,p.sha(self.lock_file),'arc_challenge',output)
            reader.assert_not_called()
        for name in ('torch','torch_xla','pyarrow'):
            self.assertNotIn(name,sys.modules)


if __name__=='__main__':
    unittest.main(verbosity=2)

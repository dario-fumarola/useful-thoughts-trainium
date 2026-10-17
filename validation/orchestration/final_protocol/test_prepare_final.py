"""Synthetic provenance regressions; no canonical data or model-stack imports."""
from pathlib import Path
from datetime import datetime, timezone, timedelta
import importlib.util
import json
import os
import sys
import tempfile
import types
import unittest
import subprocess
from unittest.mock import patch, Mock

spec=importlib.util.spec_from_file_location('final_plan_fixture',Path(__file__).with_name('prepare_final.py'))
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

def write(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj))

class FinalPreparationTests(unittest.TestCase):
    def test_optimized_python_cannot_bypass_provenance_guards(self):
        with tempfile.TemporaryDirectory() as temp:
            output=Path(temp)/'never-created.json'
            result=subprocess.run([sys.executable,'-O',str(Path(__file__).with_name('prepare_final.py')),
                'declare','--output',str(output)],capture_output=True,text=True,timeout=10)
            self.assertNotEqual(result.returncode,0)
            self.assertIn('requires enabled assertions',result.stderr)
            self.assertFalse(output.exists())

    def test_added_active_source_invalidates_seal(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); source=root/'source'
            for name in ['train','scripts','weekend_thinking']:
                (source/name).mkdir(parents=True)
            (source/'train/driver.py').write_text('VALUE = 1\n')
            files={'train/driver.py':module.sha(source/'train/driver.py')}
            source_sha=module.canonical(files)
            frozen=dict(source=dict(sha256=source_sha),terminal_steps=256,
                deadline_utc=(datetime.now(timezone.utc)+timedelta(hours=1)).isoformat())
            frozen['spec_hash']=module.canonical(frozen)
            manifest=root/'manifest.json'; training=root/'frozen.json'
            write(manifest,dict(source=dict(sha256=source_sha,files=files)))
            write(training,frozen)
            with patch.multiple(module,SOURCE=source,SOURCE_SHA=source_sha,FROZEN=training,
                    FROZEN_SHA=module.sha(training),MANIFEST=manifest,MANIFEST_SHA=module.sha(manifest)):
                self.assertEqual(module.source_check(),frozen)
                (source/'scripts/unrecorded.py').write_text('VALUE = 2\n')
                with self.assertRaises(AssertionError):module.source_check()

    def test_incomplete_terminal_cohort_stops_before_identity_verification(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); verify=Mock()
            write(root/'local/science/shared-backups.state.json',dict(terminal_verified_runs=[]))
            with patch.object(module,'ROOT',root),patch.object(module,'read_plan',return_value={}),\
                    patch.object(module,'source_check',return_value={}),patch.dict(sys.modules,{'train.evaluate':types.SimpleNamespace(asset_provenance=verify)}):
                with self.assertRaises(AssertionError):module.lock_models(root/'plan','unused',root/'never-created')
            verify.assert_not_called()
            self.assertFalse((root/'never-created').exists())

    def test_wrong_returned_run_is_rejected_without_a_lock(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            with patch.object(module,'ROOT',root):
                expected=module.expected_models()
                runs=[v['run_id'] for v in expected.values() if v['checkpoint']]
                write(root/'local/science/shared-backups.state.json',dict(terminal_verified_runs=runs))
                for index,v in enumerate(expected.values()):
                    if not v['checkpoint']:continue
                    c=root/'local/science/control'/v['run_id']
                    write(c/'exit.json',dict(driver_exit_code=0))
                    identity=dict(pid=90000000+index,starttime_ticks=-1,pgid=-1)
                    write(c/'processes.json',dict(run_id=v['run_id'],spec_hash='fixture-spec',source_sha256=module.SOURCE_SHA,
                        launcher=identity,timeout=identity,driver=identity,
                        workers=[dict(pid=91000000+n,starttime_ticks=-1,pgid=-1) for n in range(32)]))
                    complete=Path(v['checkpoint']).parent/'COMPLETE.json'
                    write(complete,{})
                    dest=root/'local/backups/science'/v['run_id']/'step-0000256'
                    write(dest/'COMPLETE.json',{})
                    write(dest/'BACKUP.json',dict(manifest_sha256=module.sha(complete)))
                frozen=dict(spec_hash='fixture-spec',assets=dict(model_dir=str(root/'fixture-assets')))
                verify=Mock(side_effect=lambda path,assets,f,arm,seed:
                    dict(arm=arm,seed=seed,run_id='wrong-run') if arm!='B0' else dict(arm='B0',seed=None))
                fake_train=types.ModuleType('train');fake_train.__path__=[]
                fake_eval=types.ModuleType('train.evaluate');fake_eval.asset_provenance=Mock(return_value={})
                fake_integrity=types.ModuleType('train.evaluation_integrity')
                fake_integrity.final_model_identity=verify;fake_integrity.evaluation_dependencies=lambda:{}
                original_iterdir=Path.iterdir
                def fixture_iterdir(path):
                    return iter(()) if path==Path('/proc') else original_iterdir(path)
                with patch.object(module,'read_plan',return_value=dict(expected_models=expected)),\
                        patch.object(module,'source_check',return_value=frozen),patch.object(module,'proc',return_value=None),\
                        patch.object(Path,'iterdir',fixture_iterdir),patch.dict(sys.modules,
                            {'train':fake_train,'train.evaluate':fake_eval,'train.evaluation_integrity':fake_integrity}),\
                        patch.object(sys,'path',list(sys.path)),patch.object(sys,'dont_write_bytecode',sys.dont_write_bytecode),\
                        patch.dict(os.environ),\
                        patch.object(module,'save') as save:
                    with self.assertRaises(AssertionError):module.lock_models(root/'plan','unused',root/'never-created')
                self.assertEqual(verify.call_count,2)
                save.assert_not_called()

    def test_changed_gsm_manifest_binding_fails_before_protocol_assembly(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            write(root/'gsm8k/scoring-manifest.json',dict(items=1319))
            write(root/'gsm8k/pre-gold-lock-binding.json',dict(pre_gold_lock_sha256='fixture-lock',scoring_manifest_sha256='0'*64))
            with patch.object(module,'read_lock',return_value=({},{})),patch.object(module,'save') as save:
                with self.assertRaises(AssertionError):module.assemble(root/'lock','fixture-lock',root,root/'never-created')
            save.assert_not_called()

if __name__=='__main__':unittest.main()

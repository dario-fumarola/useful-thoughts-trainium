"""Offline integration tests. Any Codex/Hub endpoint here is a test double."""
from pathlib import Path
import importlib.util
import json
import os
import subprocess
import sys
import pytest
from weekend_thinking.artifacts import atomic_json,sha256_file

ROOT=Path(__file__).resolve().parents[1]


def test_overlap_screen_end_to_end(tmp_path):
    docs=tmp_path/'docs.jsonl';questions=tmp_path/'questions.jsonl';out=tmp_path/'filtered.jsonl'
    docs.write_text('\n'.join(json.dumps(x) for x in [
        {'text':'alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi'},
        {'text':'this unrelated document should remain available for the training corpus'}])+'\n')
    questions.write_text(json.dumps({'question':'alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu'})+'\n')
    p=subprocess.run([sys.executable,str(ROOT/'scripts/audit_benchmark_overlap.py'),
                      '--documents',str(docs),'--benchmark-questions',str(questions),'--out',str(out)],
                      capture_output=True,text=True,check=True)
    report=json.loads(p.stdout)
    assert report['excluded_documents']==1 and report['kept_documents']==1
    assert len(out.read_text().splitlines())==1
    again=subprocess.run([sys.executable,str(ROOT/'scripts/audit_benchmark_overlap.py'),
                      '--documents',str(docs),'--benchmark-questions',str(questions),'--out',str(out)],
                      capture_output=True,text=True)
    assert again.returncode!=0


def test_codex_script_supplies_prompt_without_unsafe_flags(tmp_path):
    (tmp_path/'scripts').mkdir();(tmp_path/'bin').mkdir()
    source=ROOT/'scripts/start_codex.sh';dest=tmp_path/'scripts/start_codex.sh';dest.write_text(source.read_text())
    (tmp_path/'CODEX_START_PROMPT.md').write_text('A complete test task')
    fake=tmp_path/'bin/codex';fake.write_text('#!/bin/sh\nprintf "%s\\n" "$#" "$1"\n');fake.chmod(0o755)
    env=dict(os.environ,PATH=str(tmp_path/'bin')+':'+os.environ['PATH'])
    p=subprocess.run(['bash',str(dest)],env=env,capture_output=True,text=True,check=True)
    assert p.stdout.splitlines()==['1','A complete test task']


def test_package_verifier_rejects_tamper(tmp_path):
    (tmp_path/'scripts').mkdir()
    script=tmp_path/'scripts/verify_package.py';script.write_text((ROOT/'scripts/verify_package.py').read_text())
    target=tmp_path/'README.md';target.write_text('test')
    atomic_json(tmp_path/'DELIVERY_MANIFEST.json',{'files':{'README.md':{'sha256':sha256_file(target),'bytes':4}}})
    p=subprocess.run([sys.executable,str(script)],capture_output=True,text=True);assert p.returncode==0
    target.write_text('changed')
    p=subprocess.run([sys.executable,str(script)],capture_output=True,text=True);assert p.returncode!=0


@pytest.mark.parametrize('name',['fetch_model.py','fetch_corpus.py','freeze_experiment.py','check_ready.py'])
def test_cli_help_does_not_download_or_reserve_device(name):
    result=subprocess.run([sys.executable,str(ROOT/'scripts'/name),'--help'],capture_output=True,text=True,timeout=15)
    assert result.returncode==0 and 'usage:' in result.stdout


def _load_script(name):
    spec=importlib.util.spec_from_file_location('offline_test_'+name.replace('.','_'),ROOT/'scripts'/name)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def test_model_fetch_pins_commit_with_mock_hub(tmp_path,monkeypatch):
    import types
    calls=[];commit='b'*40
    class API:
        def model_info(self,*args,**kwargs):return types.SimpleNamespace(sha=commit)
    def download(**kwargs):
        calls.append(kwargs);out=Path(kwargs['local_dir']);out.mkdir(parents=True,exist_ok=True)
        (out/'config.json').write_text('{"model_type":"qwen3"}')
        (out/'model.safetensors').write_bytes(b'UNIT TEST FAKE WEIGHTS')
    monkeypatch.setitem(sys.modules,'huggingface_hub',types.SimpleNamespace(HfApi=API,snapshot_download=download))
    module=_load_script('fetch_model.py');out=tmp_path/'model'
    monkeypatch.setattr(sys,'argv',['fetch_model.py','--out',str(out)])
    module.main()
    manifest=json.loads((out/'ASSET_MANIFEST.json').read_text())
    assert calls[0]['revision']==commit and manifest['resolved_revision']==commit
    with pytest.raises(SystemExit):module.main()
    assert len(calls)==1


def test_model_fetch_rejects_unidentified_nonempty_directory(tmp_path,monkeypatch):
    import types
    class API:
        def model_info(self,*a,**k):return types.SimpleNamespace(sha='b'*40)
    def forbidden(**kwargs):raise AssertionError('Should not download into unidentified directory.')
    monkeypatch.setitem(sys.modules,'huggingface_hub',types.SimpleNamespace(HfApi=API,snapshot_download=forbidden))
    out=tmp_path/'model';out.mkdir();(out/'unrelated').write_text('keep')
    module=_load_script('fetch_model.py');monkeypatch.setattr(sys,'argv',['fetch_model.py','--out',str(out)])
    with pytest.raises(SystemExit):module.main()
    assert (out/'unrelated').read_text()=='keep'


def test_corpus_fetch_is_bounded_with_mock_dataset(tmp_path,monkeypatch):
    import types
    calls=[];commit='c'*40
    class API:
        def dataset_info(self,*a,**k):return types.SimpleNamespace(sha=commit)
    class Stream:
        def shuffle(self,**kwargs):calls.append(('shuffle',kwargs));return self
        def __iter__(self):return iter([{'text':'one','url':None},{'text':'two','url':'https://example.invalid/x'}])
    def load(*a,**k):calls.append(('load',k));return Stream()
    monkeypatch.setitem(sys.modules,'huggingface_hub',types.SimpleNamespace(HfApi=API))
    monkeypatch.setitem(sys.modules,'datasets',types.SimpleNamespace(load_dataset=load))
    out=tmp_path/'raw.jsonl';module=_load_script('fetch_corpus.py')
    monkeypatch.setattr(sys,'argv',['fetch_corpus.py','--out',str(out),'--max-documents','1'])
    module.main()
    assert len(out.read_text().splitlines())==1
    assert calls[0][1]['revision']==commit
    assert json.loads(out.with_suffix('.source.json').read_text())['data_audit_passed'] is False

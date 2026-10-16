import json
from pathlib import Path
import subprocess
import sys
import pytest
from weekend_thinking.hardware import report
from weekend_thinking.core import Record,Format
from weekend_thinking.data import file_sha256

ROOT=Path(__file__).resolve().parents[1]

def test_doctor_never_dumps_secret_environment(monkeypatch):
    monkeypatch.setenv('AWS_SECRET_ACCESS_KEY','SHOULD_NEVER_APPEAR')
    monkeypatch.setenv('HF_TOKEN','ANOTHER_SECRET')
    result=report()
    assert 'SHOULD_NEVER_APPEAR' not in json.dumps(result)
    assert 'ANOTHER_SECRET' not in json.dumps(result)
    assert result['production_qualified'] is False


def test_file_hash_streaming(tmp_path):
    p=tmp_path/'x';p.write_text('abc')
    assert file_sha256(p)=='ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad'


@pytest.mark.parametrize('prefix,future,split',[((),(1,),'train'),((1,),(),'train'),((1,),(2,),'invalid'),((-1,),(2,),'train')])
def test_invalid_records_rejected(prefix,future,split):
    with pytest.raises(ValueError):Record('a','b',split,prefix,future)


def test_pretokenized_preparation_executes(tmp_path):
    src=tmp_path/'input.jsonl';dest=tmp_path/'records.jsonl'
    rows=[{'document_id':str(i),'tokens':[i+1]*24,'domain':'toy'} for i in range(40)]
    rows.append(rows[0])
    src.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    cmd=[sys.executable,str(ROOT/'scripts'/'prepare_records.py'),str(src),'--out',str(dest),
         '--tokenizer-identity','toy-v1','--source-revision','toy-v1','--prefix-length','8','--future-length','4']
    result=subprocess.run(cmd,capture_output=True,text=True,check=True)
    manifest=json.loads(result.stdout)
    assert manifest['records']==40
    assert manifest['counts']['exact_duplicates_removed']==1
    assert manifest['output_sha256']==file_sha256(dest)

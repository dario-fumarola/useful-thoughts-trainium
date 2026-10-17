"""Host-only process/pipe smoke; never starts the installed Neuron monitor."""
from datetime import datetime,timedelta,timezone
import gzip
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import pytest

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('shared_monitor',HERE/'shared_node_monitor.py')
monitor=importlib.util.module_from_spec(spec);spec.loader.exec_module(monitor)
FAKE=[sys.executable,'-u','-c',
      'import json,time\nwhile True:\n print(json.dumps({"fake_host_monitor":True}),flush=True);time.sleep(.01)']


def config(tmp_path,seconds=.4):
    return dict(deadline_utc=(datetime.now(timezone.utc)+timedelta(seconds=seconds)).isoformat(),
        output_dir=str(tmp_path/'capture'),config_sha256='fake-config',spec_hash='fake-spec',source_sha256='fake-source',
        helper_sha256='host-test-helper',launcher_sha256='host-test-launcher',frozen_experiment_sha256='fake-frozen')


def assert_reaped(state):
    assert state['owned_children_reaped']
    for key in ['producer','compressor']:
        assert not Path('/proc/'+str(state[key]['pid'])).exists()


def test_deadline_drains_real_gzip_and_publishes_complete_archive(tmp_path):
    cfg=config(tmp_path)
    assert monitor.supervise(cfg,producer_command=FAKE)==0
    output=Path(cfg['output_dir']);state=json.loads((output/'state.json').read_text())
    assert state['status']=='DEADLINE_COMPLETE' and state['complete'] and state['gzip_closed_successfully']
    assert not (output/'node.jsonl.gz.partial').exists()
    rows=gzip.decompress((output/'node.jsonl.gz').read_bytes()).splitlines()
    assert rows and all(json.loads(x)=={'fake_host_monitor':True} for x in rows)
    assert_reaped(state)


@pytest.mark.parametrize('which',['producer','compressor'])
def test_early_child_exit_never_appears_complete(tmp_path,which):
    cfg=config(tmp_path)
    kwargs=dict(producer_command=FAKE)
    kwargs[which+'_command']=[sys.executable,'-c','raise SystemExit(7)']
    assert monitor.supervise(cfg,**kwargs)==1
    output=Path(cfg['output_dir']);state=json.loads((output/'state.json').read_text())
    assert not state['complete'] and state['status'].startswith('FAILED_')
    assert not (output/'node.jsonl.gz').exists() and (output/'node.jsonl.gz.partial').exists()
    assert_reaped(state)


@pytest.mark.parametrize('number',[signal.SIGTERM,signal.SIGINT])
def test_signal_cleans_producer_and_compressor_and_retains_interrupted_prefix(tmp_path,number):
    cfg=config(tmp_path,seconds=20)
    code=('import importlib.util,json\n'
          f's=importlib.util.spec_from_file_location("monitor",{str(HERE/"shared_node_monitor.py")!r})\n'
          'm=importlib.util.module_from_spec(s);s.loader.exec_module(m)\n'
          f'raise SystemExit(m.supervise(json.loads({json.dumps(cfg)!r}),producer_command={FAKE!r}))\n')
    child=subprocess.Popen([sys.executable,'-c',code])
    state_path=Path(cfg['output_dir'])/'state.json'
    try:
        for _ in range(100):
            if state_path.exists() and json.loads(state_path.read_text())['status']=='RUNNING':break
            time.sleep(.02)
        else:raise AssertionError('Fake host pipeline failed to start')
        child.send_signal(number)
        assert child.wait(timeout=10)==128+number
        state=json.loads(state_path.read_text())
        assert state['status']=='INTERRUPTED' and not state['complete'] and state['gzip_closed_successfully']
        assert not (state_path.parent/'node.jsonl.gz').exists()
        assert_reaped(state)
    finally:
        if child.poll() is None:child.kill();child.wait()


def test_duplicate_node_leader_cannot_start_a_second_pipeline(tmp_path):
    with monitor.leader_lock(tmp_path/'node.lock'):
        with pytest.raises(RuntimeError,match='already owns'):
            with monitor.leader_lock(tmp_path/'node.lock'):
                raise AssertionError('Duplicate leader entered')

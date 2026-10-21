"""Run one frozen uniform-LM job with owned-process and resource accounting."""
import argparse
from datetime import datetime, timezone
import importlib.util
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time

from train import partial_state_driver as driver
from train import uniform_lm
from weekend_thinking.artifacts import atomic_json, sha256_file, utc_now


def main():
    a = argparse.ArgumentParser(description=__doc__)
    a.add_argument('--release', type=Path, required=True)
    a.add_argument('--job', required=True)
    args = a.parse_args()
    release = json.loads(args.release.read_text())
    if (release['supervisor_sha256'] != sha256_file(__file__)
            or release['source_files'] != driver.source_files()):
        raise ValueError('Frozen execution source differs')
    job = release['named_jobs'][args.job]
    protocol, output = Path(job['protocol']), Path(job['output'])
    p = json.loads(protocol.read_text())
    validation = argparse.Namespace(arm=uniform_lm.ARM, execution_release=args.release,
                                   output=output, protocol=protocol)
    driver.validate(p)
    uniform_lm.validate_execution(p, validation, driver.source_files())
    helper = release['process_helper']
    if sha256_file(helper['path']) != helper['sha256']:
        raise ValueError('Process helper differs')
    spec = importlib.util.spec_from_file_location('owned_process_helpers', helper['path'])
    h = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(h)
    control = Path(release['control'])
    stage = control/'jobs'/args.job
    stage.mkdir(parents=True, exist_ok=False)
    output.parent.mkdir(parents=True, exist_ok=True)
    topology = [r for r in h.topology() if r['neuron_device'] in (8, 9)]
    if (len(topology) != 2 or p['visible_cores'] != '32-39'
            or any(r['neuron_processes'] or r['logical_neuroncore_config'] != 2 for r in topology)):
        raise RuntimeError('Assigned existing chips must be idle')
    roots = [Path(release['repository_root']), Path(release['cache_filesystem']), output.parent]
    if min(shutil.disk_usage(q).free for q in roots) < 35*2**30 or h.available() < 768*2**30:
        raise RuntimeError('Insufficient prelaunch disk or memory')
    ledgers = list((control/'jobs').glob('*/closure.json'))
    used = sum(json.loads(q.read_text())['reserved_chip_hours'] for q in ledgers)
    seconds = min(job['maximum_wall_seconds'],
                  1800*(release['maximum_reserved_chip_hours']-used))
    if seconds <= 30:
        raise RuntimeError('Insufficient remaining bounded reservation')
    deadline = datetime.fromtimestamp(time.time()+seconds-10, timezone.utc).isoformat()
    tmp = Path('/dev/shm')/('ulm_'+str(os.getpid()))
    tmp.mkdir(mode=0o700)
    with socket.socket(socket.AF_UNIX) as sock:
        sock.bind(str(tmp/'socket-probe'))
    (tmp/'socket-probe').unlink()
    port = job['port']
    for value in (port, port+1):
        with socket.socket() as sock:
            sock.bind(('127.0.0.1', value))
    template = json.loads(Path(release['environment_template']).read_text())
    env = os.environ.copy()
    for key in template['unset']+['MALLOC_ARENA_MAX']:
        env.pop(key, None)
    env.update(template['environment'])
    env.update(p['runtime_policy']['environment'])
    env.update(PATH=str(Path(sys.executable).parent)+os.pathsep+os.environ['PATH'],
               PYTHONPATH=release['source_root']+':'+str(roots[0]/'local/test-deps'),
               NEURON_RT_VISIBLE_CORES='32-39', NEURONCORE_NUM_DEVICES='8',
               MASTER_PORT=str(port), NEURON_RT_ROOT_COMM_ID=f'127.0.0.1:{port+1}',
               TMPDIR=str(tmp))
    command = ['timeout', '--signal=TERM', '--kill-after=5s', str(seconds-5)+'s',
               sys.executable, '-u', '-m', 'train.partial_state_driver',
               '--protocol', str(protocol), '--output', str(output), '--arm', uniform_lm.ARM,
               '--stop-file', str(stage/'STOP'), '--deadline-utc', deadline,
               '--execution-release', str(args.release)]
    if job.get('resume'):
        command += ['--resume', job['resume'], '--attempt-id', 'reproduce-3-4']
    started = time.monotonic()
    failure = None
    observed = False
    with (stage/'run.log').open('xb') as log, (stage/'host.jsonl').open('x') as samples:
        process = subprocess.Popen(command, cwd=release['source_root'], env=env,
                                   stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        owned = h.identity(process.pid)
        if not owned or owned['session'] != process.pid:
            raise RuntimeError('Cannot establish owned process session')
        atomic_json(stage/'launch.json', dict(utc=utc_now(), identity=owned, command=command,
                    protocol_sha256=sha256_file(protocol), release_sha256=sha256_file(args.release)))
        try:
            while process.poll() is None:
                members = h.members(owned['session'])
                free = {str(q): shutil.disk_usage(q).free for q in roots}
                sample = dict(utc=utc_now(), available_bytes=h.available(), free_bytes=free,
                              elapsed_seconds=time.monotonic()-started)
                samples.write(json.dumps(sample)+'\n')
                samples.flush()
                if not observed:
                    actual = [v for r in h.topology() if r['neuron_device'] in (8, 9)
                              for v in r['neuron_processes']]
                    if len(actual) == 8:
                        if not {v['pid'] for v in actual} <= {v['pid'] for v in members}:
                            raise RuntimeError('Assigned devices include an unowned worker')
                        observed = True
                        atomic_json(stage/'actual-eight-workers.json', dict(utc=utc_now(), workers=actual))
                if (sample['available_bytes'] < 512*2**30 or min(free.values()) < 35*2**30
                        or sample['elapsed_seconds'] >= seconds-10):
                    raise RuntimeError('Declared memory, disk or reservation boundary reached')
                time.sleep(2)
        except BaseException as error:
            failure = repr(error)
            h.kill_remaining(owned)
        code = process.wait()
    if h.members(owned['session']):
        h.kill_remaining(owned)
    absent = not h.members(owned['session'])
    idle = all(not r['neuron_processes'] for r in h.topology() if r['neuron_device'] in (8, 9))
    closure = dict(utc=utc_now(), exit_code=code, failure=failure, actual_eight_observed=observed,
                   owned_session_absent=absent, assigned_devices_idle=idle,
                   wall_seconds=time.monotonic()-started,
                   reserved_chip_hours=2*(time.monotonic()-started)/3600)
    atomic_json(stage/'closure.json', closure)
    print(json.dumps(closure), flush=True)
    if code or failure or not observed or not absent or not idle:
        raise RuntimeError('Native stage did not close successfully; evidence retained')


if __name__ == '__main__':
    main()

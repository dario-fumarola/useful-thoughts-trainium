"""One node-wide neuron-monitor -> gzip stream, bounded by the frozen deadline.

Outside active training source. A partial stream is never published as complete;
TERM/INT stops the producer, drains gzip, and records an interrupted prefix.
"""
from __future__ import annotations
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import stat
import subprocess
import time

NEURON_MONITOR = '/opt/aws/neuron/bin/neuron-monitor'
GZIP = '/usr/bin/gzip'
NODE_LOCK = Path('/home/ubuntu/dario/local/science/shared-node-monitor.lock')


def now():
    return datetime.now(timezone.utc)


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):digest.update(chunk)
    return digest.hexdigest()


def canonical(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def ordinary(path):
    path=Path(path)
    if path.is_symlink() or not path.is_file():raise ValueError('Regular file required: '+str(path))
    return path


def utc_deadline(value):
    date=datetime.fromisoformat(value.replace('Z','+00:00'))
    if date.tzinfo is None:raise ValueError('Timezone-aware frozen deadline required')
    return date.astimezone(timezone.utc)


def read_config(path,expected_sha256):
    path=ordinary(path)
    if sha(path)!=expected_sha256:raise ValueError('Frozen monitor configuration changed')
    config=json.loads(path.read_text())
    if config['schema']!='shared-node-monitor-v1':raise ValueError('Wrong monitor configuration schema')
    frozen_path=ordinary(config['frozen_experiment'])
    frozen=json.loads(frozen_path.read_text())
    if (sha(frozen_path)!=config['frozen_experiment_sha256']
            or frozen['spec_hash']!=config['spec_hash']
            or canonical({k:v for k,v in frozen.items() if k!='spec_hash'})!=frozen['spec_hash']
            or config['deadline_utc']!=frozen['deadline_utc']):
        raise ValueError('Monitor identity/deadline differs from the frozen experiment')
    if utc_deadline(config['deadline_utc'])<=now():raise ValueError('Frozen monitor deadline has passed')
    directory=Path(__file__).resolve().parent
    if (sha(directory/'shared_node_monitor.py')!=config['helper_sha256']
            or sha(directory/'launch_shared_node_monitor.sh')!=config['launcher_sha256']):
        raise ValueError('Separately frozen monitor helper/launcher changed')
    output,nvme=Path(config['output_dir']),Path(config['nvme_root'])
    if (not output.is_absolute() or not nvme.is_absolute() or output.is_symlink()
            or not nvme.is_mount() or not output.resolve().is_relative_to(nvme.resolve())
            or output.exists() or not output.parent.is_dir()
            or output.parent.stat().st_dev!=nvme.stat().st_dev):
        raise ValueError('Require a fresh output directory on the mounted NVMe filesystem')
    config.update(config_sha256=expected_sha256,config_file=str(path),source_sha256=frozen['source']['sha256'])
    return config


@contextmanager
def leader_lock(path=NODE_LOCK):
    path.parent.mkdir(parents=True,exist_ok=True)
    descriptor=os.open(path,os.O_RDWR|os.O_CREAT|os.O_NOFOLLOW,0o600)
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):raise ValueError('Node monitor lock must be regular')
        try:fcntl.flock(descriptor,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError as error:raise RuntimeError('A shared node monitor already owns the node lock') from error
        yield
    finally:os.close(descriptor)


def atomic_state(path,value):
    temporary=path.with_suffix('.tmp')
    with temporary.open('x') as stream:
        stream.write(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n')
        stream.flush();os.fsync(stream.fileno())
    temporary.replace(path)


def identity(pid):
    path=Path('/proc')/str(pid)
    fields=(path/'stat').read_text().rsplit(')',1)[1].split()
    return dict(pid=pid,starttime_ticks=int(fields[19]),parent_pid=int(fields[1]),pgid=int(fields[2]),
        state=fields[0],argv=(path/'cmdline').read_bytes().decode().rstrip('\0').split('\0'))


def stop_child(child,grace):
    """Only direct owned Popen children; outer timeout is their hard group bound."""
    forced=False
    if child is None:return dict(exit_code=None,forced_kill=False)
    if child.poll() is None:
        child.terminate()
        try:child.wait(timeout=grace)
        except subprocess.TimeoutExpired:
            forced=True;child.kill();child.wait(timeout=5)
    return dict(exit_code=child.returncode,forced_kill=forced)


def supervise(config,*,producer_command=None,compressor_command=None):
    """Command injection is a host-test seam; production CLI has no override."""
    deadline=utc_deadline(config['deadline_utc'])
    output=Path(config['output_dir']);output.mkdir(exist_ok=False)
    partial=output/'node.jsonl.gz.partial';final=output/'node.jsonl.gz'
    producer=compressor=None
    interrupted=[];handlers={}
    started=now();started_mono=time.monotonic()
    metadata=dict(schema='shared-node-monitor-execution-v1',status='STARTING',started_utc=started.isoformat(),
        deadline_utc=deadline.isoformat(),config_sha256=config['config_sha256'],
        spec_hash=config['spec_hash'],source_sha256=config['source_sha256'],
        helper_sha256=config['helper_sha256'],launcher_sha256=config['launcher_sha256'],
        frozen_experiment_sha256=config['frozen_experiment_sha256'],
        config_file=config.get('config_file'),frozen_experiment=config.get('frozen_experiment'),
        cleanup_grace_seconds=30,outer_final_kill_grace_seconds=5,
        supervisor=identity(os.getpid()),launcher_exec_parent=identity(os.getppid()),
        lock_file=str(NODE_LOCK),output_partial=str(partial),output_final=str(final),
        capture_reached_frozen_deadline=False,complete=False)
    atomic_state(output/'state.json',metadata)
    def received(number,_frame):
        interrupted.append(number)
    reason='FAILED_STARTUP';error=None
    producer_result=compressor_result=None
    try:
        for number in (signal.SIGTERM,signal.SIGINT):handlers[number]=signal.signal(number,received)
        with (output/'producer.stderr.log').open('xb') as producer_error, \
             (output/'gzip.stderr.log').open('xb') as compressor_error, partial.open('xb') as archive:
            try:
                if now()>=deadline:raise TimeoutError('Frozen deadline expired before producer launch')
                # Same outer timeout process group: its final KILL also reaches
                # both children if the supervisor itself becomes unresponsive.
                producer=subprocess.Popen(producer_command or [NEURON_MONITOR],stdout=subprocess.PIPE,stderr=producer_error)
                compressor=subprocess.Popen(compressor_command or [GZIP,'-1','-c'],stdin=producer.stdout,
                    stdout=archive,stderr=compressor_error)
                producer.stdout.close()
                metadata.update(status='RUNNING',producer=identity(producer.pid),compressor=identity(compressor.pid),
                    producer_command=producer.args,compressor_command=compressor.args)
                atomic_state(output/'launch.json',metadata)
                atomic_state(output/'state.json',metadata)
                while True:
                    if interrupted:reason='INTERRUPTED';break
                    exits=dict(producer=producer.poll(),compressor=compressor.poll())
                    if any(value is not None for value in exits.values()):
                        metadata['first_observed_early_exit_codes']=exits
                        reason='FAILED_EARLY_PIPELINE_EXIT';break
                    if now()>=deadline:reason='DEADLINE';break
                    time.sleep(min(.2,max(0.,(deadline-now()).total_seconds())))
            except BaseException as caught:
                error=f'{type(caught).__name__}: {caught}';reason='FAILED_SUPERVISOR'
            finally:
                # Stop producer first, close its pipe, and let gzip consume EOF
                # and write its trailer. TERM to gzip is only a bounded fallback.
                producer_result=stop_child(producer,5)
                if producer is not None and producer.stdout is not None:producer.stdout.close()
                if compressor is not None:
                    try:compressor.wait(timeout=15)
                    except subprocess.TimeoutExpired:pass
                compressor_result=stop_child(compressor,5)
                archive.flush();os.fsync(archive.fileno())
    except BaseException as caught:
        error=f'{type(caught).__name__}: {caught}';reason='FAILED_SUPERVISOR'
        producer_result=stop_child(producer,5);compressor_result=stop_child(compressor,5)
    finally:
        for number,handler in handlers.items():signal.signal(number,handler)
    gzip_closed=(compressor_result is not None and compressor_result['exit_code']==0
                 and not compressor_result['forced_kill'])
    clean=(gzip_closed
           and not compressor_result['forced_kill'] and producer_result is not None
           and producer_result['exit_code'] in (0,-signal.SIGTERM) and not producer_result['forced_kill'])
    complete=reason=='DEADLINE' and clean
    exit_code=0 if complete else (128+interrupted[0] if interrupted else 1)
    if complete:partial.rename(final)
    artifact=final if complete else partial
    metadata.update(status='DEADLINE_COMPLETE' if complete else ('FAILED_DEADLINE_CLEANUP' if reason=='DEADLINE' else reason),complete=complete,
        capture_reached_frozen_deadline=reason=='DEADLINE',finished_utc=now().isoformat(),
        elapsed_seconds=time.monotonic()-started_mono,signals_received=interrupted,error=error,
        supervisor_exit_code=exit_code,
        producer_result=producer_result,compressor_result=compressor_result,
        gzip_closed_successfully=gzip_closed,artifact=str(artifact),artifact_bytes=artifact.stat().st_size if artifact.exists() else None,
        owned_children_reaped=all(child is None or child.poll() is not None for child in (producer,compressor)))
    atomic_state(output/'state.json',metadata)
    return exit_code


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--config-sha256',required=True)
    args=parser.parse_args()
    config=read_config(args.config,args.config_sha256)
    with leader_lock():return supervise(config)


if __name__=='__main__':
    raise SystemExit(main())

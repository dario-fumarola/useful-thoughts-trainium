"""Independently verify and preserve the bounded batching engineering evidence."""
from pathlib import Path
from datetime import datetime, timezone
from statistics import median
import json
import os
import shutil
import tarfile
import torch
from weekend_thinking.artifacts import atomic_json, sha256_file

R = Path('/home/ubuntu/dario')
W = R / 'local/worktrees/causal-decoder-batching-v1'
A = R / 'local/diagnostics/causal-decoder-batching-v1'
B = R / 'local/diagnostics/causal-decoder-batching-io-v2'
P = R / 'validation/causal_decoder_batching_v1_20261020'
read = lambda p: json.loads(Path(p).read_text())


def main():
    old, close = read(A/'closure.json'), read(B/'closure.json')
    assert old['exit_code']==1 and close['exit_code']==0
    assert all(v['assigned_physical_device_empty'] and v['all_original_session_processes_absent'] for v in (old,close))
    result = read(B/'worker-result.json')
    assert result['status']=='PASS_B0_TRAIN_NATIVE_AND_EXACT_FORCED_GREEDY_BATCH_PARITY'
    assert read(A/'plan.json')['candidate_commit']==read(B/'plan.json')['candidate_commit']
    for name in ('inputs.json','CPU-native.pt'):
        assert sha256_file(A/name)==sha256_file(B/name)
    expected = read(B/'original-serial-attempts.json')
    assert len(expected)==8 and all(v['stop_type']=='length_cap' and len(v['actions'])==32 for v in expected)
    assert expected==read(B/'batched-B4-attempts.json')==read(B/'batched-B8-attempts.json')
    cpu = torch.load(B/'CPU-native.pt',weights_only=True,map_location='cpu')
    assert cpu.shape[0]==8 and cpu.shape[1]>150000 and torch.isfinite(cpu).all()
    errors = {}
    for size in (1,4,8):
        actual = torch.load(B/f'XLA-native-B{size}.pt',weights_only=True,map_location='cpu')
        assert actual.shape==cpu.shape and torch.isfinite(actual).all()
        error = float((actual-cpu).abs().max())
        assert error<=.001
        assert all(v['passed'] for v in read(B/f'native-B{size}.json'))
        errors[str(size)] = error
        times = read(B/'timings.json')[str(size)]
        assert len(times)==3 and min(times)>0 and median(times)==result['median_seconds'][str(size)]
    assert all(sha256_file(Path(n))==v for n,v in read(B/'plan.json')['sources'].items())
    P.mkdir(exist_ok=False)
    files = {}
    def copy(source, relative):
        target=P/relative
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(source,target)
        assert sha256_file(source)==sha256_file(target)
        with target.open('rb') as f:os.fsync(f.fileno())
        files[str(target.relative_to(R))]=sha256_file(target)
    for source,prefix in ((A,'original-failed-v1'),(B,'explicit-io-v2')):
        for path in sorted(source.rglob('*')):
            if path.is_file() and path.suffix!='.pt':copy(path,str(Path(prefix)/path.relative_to(source)))
    witnesses = P/'full-vocabulary-numerical-witnesses.tar.gz'
    with tarfile.open(witnesses,'x:gz') as archive:
        for source,prefix in ((A,'original-failed-v1'),(B,'explicit-io-v2')):
            for path in sorted(source.glob('*.pt')):archive.add(path,arcname=str(Path(prefix)/path.name))
    with tarfile.open(witnesses,'r:gz') as archive:
        for member in archive.getmembers():
            source=(A if member.name.startswith('original-failed-v1/') else B)/Path(member.name).name
            import hashlib
            assert hashlib.sha256(archive.extractfile(member).read()).hexdigest()==sha256_file(source)
    with witnesses.open('rb') as f:os.fsync(f.fileno())
    files[str(witnesses.relative_to(R))]=sha256_file(witnesses)
    for name in ('train/batched_causal_decoder.py','tests/test_batched_causal_decoder.py'):
        copy(W/name,Path(name).name)
    for name in ('qualify_causal_decoder_batching_v1_20261020.py','supervise_causal_decoder_batching_v1_20261020.py',
                 'qualify_causal_decoder_batching_io_v2_20261020.py','supervise_causal_decoder_batching_io_v2_20261020.py',
                 Path(__file__).name):
        copy(R/'local/analysis'/name,name)
    copy(R/'local/causal-decoder-batching-v1-checks-20261020.log','CPU-full-checks.log')
    for directory in sorted([P,*[p for p in P.rglob('*') if p.is_dir()]],key=lambda p:len(p.parts),reverse=True):
        fd=os.open(directory,os.O_RDONLY);os.fsync(fd);os.close(fd)
    report=dict(status='PASS_BOUNDED_B0_TRAIN_ONLY_BATCHED_DECODER_ENGINEERING_CHECK',
        utc=datetime.now(timezone.utc).isoformat(),experiment_id='causal-decoder-batching-v1',
        candidate_commit=read(B/'plan.json')['candidate_commit'],CPU_tests=1873,CPU_full_seconds=67.43,
        native_vectors=24,native_max_logp_error=errors,exact_forced_greedy_sequences=8,
        actions_per_sequence=32,actions_per_timing=256,timed_repeats_per_condition=3,
        median_seconds=result['median_seconds'],speedup_vs_serial=result['speedup_vs_serial'],
        successful_probe_reserved_chip_hours=close['reserved_chip_hours'],
        preserved_failed_io_probe_reserved_chip_hours=old['reserved_chip_hours'],
        total_reserved_chip_hours=old['reserved_chip_hours']+close['reserved_chip_hours'],
        independent_verification='Exact full attempts,all24 full-vocabulary vectors,identical original/IOv2 inputs,median timing and physical closure independently rechecked; publication/archive readback and fsync pass.',
        files=files,
        limits=['Engineering evidence on unchanged B0 and eight question-only TRAIN prefixes,not capability or generalization evidence.',
            'Forced32-action warm microbenchmark on one logical core; shared-host timings are descriptive,not an end-to-end serving benchmark.',
            'Equal real prefix lengths only. No general ragged batching,sampling,trained-model or full128-action qualification claimed.',
            'Original producer failed on missing atomic progress overwrite; its entire original record is retained. Explicit IOv2 changes output paths and that overwrite only.',
            'No weights,held task data,labels,prompts,precision,parsers,gates or numerical source in live V28 changed. The candidate is not adopted there.',
            'A future adoption needs its own trained-model and natural-stopping integration check. Existing checks do not imply global greedy equivalence.'])
    output=R/'validation/causal_decoder_batching_v1_20261020.json'
    atomic_json(output,report)
    print(json.dumps(dict(status=report['status'],report=str(output),sha256=sha256_file(output),
        median_seconds=result['median_seconds'],speedup_vs_serial=result['speedup_vs_serial'],
        total_reserved_chip_hours=report['total_reserved_chip_hours'],public_files=len(files))))


if __name__=='__main__':main()

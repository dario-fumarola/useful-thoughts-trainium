from pathlib import Path
import hashlib,json
import pytest
from train import permutation_program_retention as retention
from weekend_thinking.artifacts import canonical_hash


def checkpoint(run,step,p):
    d=run/f'step-{step:07d}';d.mkdir(parents=True)
    files={}
    for name in ['model.pt',*[f'rank-{i:05d}.pt' for i in range(4)]]:
        data=(name+str(step)).encode();(d/name).write_bytes(data)
        files[name]=dict(bytes=len(data),sha256=hashlib.sha256(data).hexdigest())
    m=dict(run_id=run.name,step=step,world_size=4,spec_hash=canonical_hash(p),files=files,
           metadata=dict(pipeline_schema='permutation-program-sft-v28',arm=p['program_arm'],seed=p['seed']))
    (d/'COMPLETE.json').write_text(json.dumps(m));return d


def setup(tmp_path,monkeypatch):
    root=tmp_path/'new-study';root.mkdir();monkeypatch.setattr(retention,'RUN_ROOT',root)
    monkeypatch.setattr(retention,'MODEL64_ROOT',tmp_path/'development64')
    p=dict(checkpoint_retention=retention.POLICY,program_arm='trace',seed=2026102028)
    return root/'program-trace-seed2026102028-v1',p


def test_new_full_readback_precedes_any_intermediate_retirement(tmp_path,monkeypatch):
    run,p=setup(tmp_path,monkeypatch);old=checkpoint(run,64,p);retention.full_readback(run,old,p)
    new=checkpoint(run,1024,p);(new/'rank-00002.pt').write_bytes(b'broken')
    with pytest.raises(ValueError):retention.rotate(run,new,p)
    assert all((old/name).exists() for name in json.loads((old/'COMPLETE.json').read_text())['files'])
    assert not (old/'RETIREMENT_INTENT.json').exists()


def test_rotation_preserves_current_state_and_old_receipts(tmp_path,monkeypatch):
    run,p=setup(tmp_path,monkeypatch);old=checkpoint(run,64,p);retention.full_readback(run,old,p)
    new=checkpoint(run,1024,p);retention.rotate(run,new,p)
    assert (old/'PAYLOAD_RETIRED.json').exists() and not (old/'model.pt').exists()
    assert all((new/name).exists() for name in json.loads((new/'COMPLETE.json').read_text())['files'])
    assert (old/'COMPLETE.json').exists() and (old/'READBACK.json').exists()


def test_scope_rejects_historical_runs_and_symlinked_directories(tmp_path,monkeypatch):
    run,p=setup(tmp_path,monkeypatch)
    with pytest.raises(ValueError):retention.scoped(tmp_path/'old-qwen-run')
    run.mkdir();alias=run.parent/'program-direct-seed2026102028-v1';alias.symlink_to(run,target_is_directory=True)
    with pytest.raises(ValueError):retention.scoped(alias)


def test_prior_seed_terminal_model_survives_optimizer_retirement(tmp_path,monkeypatch):
    run,p=setup(tmp_path,monkeypatch);terminal=checkpoint(run,4096,p);retention.full_readback(run,terminal,p)
    audit=dict(status='PASS_V28_TERMINAL_ALL_STATE',step=4096,COMPLETE_sha256=retention.sha256_file(terminal/'COMPLETE.json'))
    (run/'terminal-full-audit.json').write_text(json.dumps(audit))
    later=run.parent/'program-trace-seed2026102029-v1';p2=dict(p,seed=2026102029)
    new=checkpoint(later,64,p2);retention.rotate(later,new,p2)
    assert (terminal/'model.pt').exists() and not (terminal/'rank-00000.pt').exists()
    assert (terminal/'COMPLETE.json').exists() and (new/'rank-00000.pt').exists()


def test_prior_seed_cannot_retire_before_independent_terminal_audit(tmp_path,monkeypatch):
    run,p=setup(tmp_path,monkeypatch);terminal=checkpoint(run,4096,p);retention.full_readback(run,terminal,p)
    later=run.parent/'program-trace-seed2026102029-v1';p2=dict(p,seed=2026102029);new=checkpoint(later,64,p2)
    with pytest.raises(FileNotFoundError):retention.rotate(later,new,p2)
    assert (terminal/'model.pt').exists() and (terminal/'rank-00000.pt').exists()


def test_actual_model64_bytes_survive_the_next_rotation(tmp_path,monkeypatch):
    run,p=setup(tmp_path,monkeypatch);first=checkpoint(run,64,p);original=(first/'model.pt').read_bytes()
    retention.rotate(run,first,p)
    new=checkpoint(run,1024,p);retention.rotate(run,new,p)
    archived=retention.MODEL64_ROOT/(run.name+'-step64.pt')
    assert archived.read_bytes()==original and not (first/'model.pt').exists()


def test_other_arm_replacement_cannot_authorize_retirement(tmp_path,monkeypatch):
    run,p=setup(tmp_path,monkeypatch);old=checkpoint(run,64,p);retention.full_readback(run,old,p)
    wrong=dict(status='FULL_STATE_READBACK_FSYNC',run_id='program-pause-seed2026102029-v1',step=64)
    with pytest.raises(ValueError):retention.retire_payload(old,wrong,keep_model=False)
    assert (old/'model.pt').exists()


def test_host_phases_are_bounded_and_bound_to_protocol(tmp_path):
    from datetime import datetime,timezone,timedelta
    p=dict(one='protocol');past=datetime.now(timezone.utc)-timedelta(seconds=1)
    with pytest.raises(TimeoutError):retention.await_phase(tmp_path,'lease-64',p,past)
    retention.publish_phase(tmp_path,'lease-64',p)
    retention.await_phase(tmp_path,'lease-64',p,past)
    with pytest.raises(ValueError):retention.await_phase(tmp_path,'lease-64',dict(other='protocol'),past)

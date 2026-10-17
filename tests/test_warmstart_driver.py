import pytest

from datetime import datetime, timedelta, timezone

from train.warmstart_driver import order_index, parse_args, stop_due, validate_protocol, validate_source_files


def protocol():
    return dict(seed=20261018,world_size=8,steps=64,lr=1e-5,betas=[.9,.95],
        eps=1e-8,weight_decay=.1,clip=1.,note_weight=1.,future_weight=1.,
        runtime_policy={"schema":1,"environment":dict(PJRT_DEVICE="NEURON",XLA_COMPILATION_CACHE_SIZE="2",
            NEURON_NUM_RECENT_MODELS_TO_KEEP=None,NEURON_CC_FLAGS="--model-type=transformer",
            XLA_USE_BF16=None,XLA_DOWNCAST_BF16=None,OMP_NUM_THREADS="2",MKL_NUM_THREADS="2")},
        **{key:"a"*64 for key in ("data_sha256","dataset_sha256","format_sha256","model_manifest_sha256")})


def test_order_repeats_complete_permuted_epochs_and_preserves_cursor():
    for count in (1, 3, 13):
        stream=[order_index(i,count,20261018) for i in range(count*4)]
        assert all(sorted(stream[i:i+count])==list(range(count)) for i in range(0,len(stream),count))
    # A cursor in the middle of an epoch must preserve the fixed stream.
    assert [order_index(i,3,42) for i in range(12)] == [1,0,2,2,1,0,2,0,1,0,2,1]
    assert [order_index(i,3,42) for i in range(5,9)] == [0,2,0,1]
    assert [order_index(i,13,20261018) for i in range(13)] != [order_index(i,13,20261019) for i in range(13)]
    with pytest.raises(ValueError):
        order_index(0,0,20261018)
    with pytest.raises(ValueError):
        order_index(-1,13,20261018)


@pytest.mark.parametrize("key,value",[("world_size",32),("steps",256),("lr",1e-6),
                                      ("note_weight",0.),("clip",True),("betas",[.9,.999])])
def test_recipe_changes_fail_closed(key,value):
    p=protocol();validate_protocol(p,8);p[key]=value
    with pytest.raises(ValueError,match=key):
        validate_protocol(p,8)


def test_runtime_and_cli_reject_unplanned_world_or_arm():
    p=protocol();p["runtime_policy"]["environment"]["XLA_COMPILATION_CACHE_SIZE"]=None
    with pytest.raises(ValueError,match="LRU2"):
        validate_protocol(p,8)
    args=["--protocol","p.json","--output","out","--arm","warmstart",
          "--deadline-utc","2026-10-18T12:00:00Z","--stop-file","STOP"]
    assert parse_args(args).processes==8
    for extra in (["--processes","32"], ["--arm","thought"]):
        with pytest.raises(SystemExit):
            parse_args(args+extra)


@pytest.mark.parametrize("missing", ["artifacts.py", "core.py", "runtime_policy.py"])
def test_source_closure_rejects_missing_imported_runtime_module(tmp_path, monkeypatch, missing):
    import train.warmstart_driver as driver
    from weekend_thinking.artifacts import sha256_file

    monkeypatch.setattr(driver, "ROOT", tmp_path)
    files = {}
    for name in driver.REQUIRED_SOURCES:
        path = tmp_path/name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(name)
        files[name] = sha256_file(path)
    validate_source_files(files)
    del files[f"weekend_thinking/{missing}"]
    with pytest.raises(ValueError, match="omits"):
        validate_source_files(files)


def test_stop_arriving_during_setup_and_expired_deadline_prevent_first_update(tmp_path):
    stop = tmp_path/"STOP"
    before_setup = datetime(2026, 10, 18, tzinfo=timezone.utc)
    deadline = before_setup + timedelta(minutes=1)
    assert not stop_due(stop, deadline, now=before_setup)
    assert stop_due(stop, deadline, now=deadline)
    assert stop_due(stop, deadline, now=deadline+timedelta(seconds=1))
    # The deadline is still in the future, but the owner creates STOP while
    # expensive setup runs. The post-setup guard must observe it independently.
    stop.write_text("owner stop")
    assert stop_due(stop, deadline, now=before_setup+timedelta(seconds=30))

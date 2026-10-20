import json
from copy import deepcopy
import pytest
from train import permutation_program_science_evaluate as evaluation


def release():
    models={'B0':dict(id='B0',checkpoint=None)}
    for key in evaluation.MODEL_IDS:
        models[key]=dict(id=key,run_id=key,arm=key.split('-')[1],status='AUDITED_V28_TERMINAL4096',step=4096)
    return dict(status='RELEASED_V28_FIXED_ALL_TERMINAL_EVALUATION',
        data_manifest_sha256=evaluation.MANIFEST_SHA,prospective_design_sha256=evaluation.DESIGN_SHA,
        evaluation_source_files=evaluation.sources(),model_specs=models)


def test_no_held_evaluation_with_partial_models_missing_seed_or_wrong_source(tmp_path):
    good=release();path=tmp_path/'release.json';path.write_text(json.dumps(good))
    assert evaluation.verify_release(path)['status']==good['status']
    changes=[]
    missing=deepcopy(good);missing['model_specs'].pop(sorted(evaluation.MODEL_IDS)[0]);changes.append(missing)
    partial=deepcopy(good);partial['model_specs'][sorted(evaluation.MODEL_IDS)[0]]['step']=64;changes.append(partial)
    bad_base=deepcopy(good);bad_base['model_specs']['B0']['checkpoint']='trained-model.pt';changes.append(bad_base)
    stale=deepcopy(good);stale['evaluation_source_files']={};changes.append(stale)
    no_release=deepcopy(good);no_release['status']='PREPARED';changes.append(no_release)
    for value in changes:
        path.write_text(json.dumps(value))
        with pytest.raises(ValueError):evaluation.verify_release(path)


def test_generator_rejects_substituted_checkpoint_before_dispatch(tmp_path,monkeypatch):
    r=release();rp=tmp_path/'release.json';rp.write_text(json.dumps(r))
    p=dict(purpose='FIXED_V28_ALL_TERMINAL_EVALUATION',prefill_width=256,release_file=str(rp),
        model=dict(id='B0',checkpoint='substituted.pt'))
    (tmp_path/'protocol.json').write_text(json.dumps(p))
    def forbidden(*args,**kwargs):raise AssertionError('Must fail before generation')
    monkeypatch.setattr(evaluation.engine,'generate',forbidden)
    with pytest.raises(ValueError):evaluation.generate(tmp_path,0,'2099-01-01T00:00:00+00:00')

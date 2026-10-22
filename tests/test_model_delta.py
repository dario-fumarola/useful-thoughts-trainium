import json
import numpy as np
import pytest
import torch
from weekend_thinking.model_delta import bits,decode,encode


def test_all_bf16_bit_patterns_and_tied_aliases_roundtrip(tmp_path):
    # Includes signed zero, infinities and every NaN payload: numerical equality
    # alone is insufficient for a lossless model artifact.
    a=torch.arange(65536,dtype=torch.int64).to(torch.uint16).view(torch.bfloat16)
    b=torch.flip(a,[0]);base={'a':a,'alias':a};trained={'a':b,'alias':b};path=tmp_path/'delta.npz'
    manifest=encode(base,trained,path,provenance={'toy':True});restored,_=decode(base,path)
    assert manifest['entries'][1]['alias_of']=='a'
    assert restored['a'].data_ptr()==restored['alias'].data_ptr()
    for name in base:assert np.array_equal(bits(restored[name]),bits(trained[name]))


def test_wrong_base_or_modified_delta_fails(tmp_path):
    base={'a':torch.arange(16,dtype=torch.bfloat16)};trained={'a':base['a']+1};path=tmp_path/'delta.npz';encode(base,trained,path,provenance={})
    with pytest.raises(ValueError,match='Wrong base'):decode({'a':base['a']+2},path)
    with np.load(path,allow_pickle=False) as old:arrays={name:old[name] for name in old.files}
    arrays['tensor_0000']=arrays['tensor_0000'].copy();arrays['tensor_0000'][0]^=1
    changed=tmp_path/'changed.npz';np.savez_compressed(changed,**arrays)
    with pytest.raises(ValueError,match='differs'):decode(base,changed)


def test_scalar_tensor_roundtrip(tmp_path):
    base={'scalar':torch.tensor(1.,dtype=torch.bfloat16)};trained={'scalar':torch.tensor(-2.,dtype=torch.bfloat16)};path=tmp_path/'scalar.npz'
    encode(base,trained,path,provenance={});restored,_=decode(base,path)
    assert restored['scalar'].shape==torch.Size([])
    assert np.array_equal(bits(restored['scalar']),bits(trained['scalar']))


def test_no_silent_precision_shape_or_key_change(tmp_path):
    tensor=torch.ones(3,dtype=torch.bfloat16)
    with pytest.raises(ValueError,match='key sets'):encode({'a':tensor},{'b':tensor},tmp_path/'keys.npz',provenance={})
    with pytest.raises(ValueError,match='CPU BF16'):encode({'a':tensor.float()},{'a':tensor},tmp_path/'dtype.npz',provenance={})
    with pytest.raises(ValueError,match='shape'):encode({'a':tensor},{'a':tensor[:2]},tmp_path/'shape.npz',provenance={})


def test_missing_arrays_or_forward_alias_fails(tmp_path):
    tensor=torch.ones(3,dtype=torch.bfloat16);path=tmp_path/'delta.npz';encode({'a':tensor},{'a':tensor+1},path,provenance={})
    with np.load(path,allow_pickle=False) as old:arrays={name:old[name] for name in old.files}
    manifest=json.loads(arrays['manifest'].tobytes());entry=manifest['entries'][0];del entry['array'];entry['alias_of']='a';arrays={'manifest':np.frombuffer(json.dumps(manifest).encode(),dtype=np.uint8)};changed=tmp_path/'alias.npz';np.savez_compressed(changed,**arrays)
    with pytest.raises(ValueError,match='preceding'):decode({'a':tensor},changed)

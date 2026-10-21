import numpy as np
import pytest

from train.causal_reader_head_driver import bootstrap


def test_paired_document_bootstrap_preserves_constant_contrasts_exactly():
    values=np.tile([.5,-.125],(4096,1))
    summaries,draws=bootstrap(values,np.random.default_rng(2026102133))
    assert draws.shape==(5000,2)
    assert summaries[0]['CI95']==[.5,.5]
    assert summaries[1]['CI95']==[-.125,-.125]
    assert summaries[0]['mean']==.5 and summaries[0]['document_SD']==0
    # Cannot silently reinterpret16 tokens/document as independent questions,
    # or substitute the former1024-item proposal after a source shortfall.
    with pytest.raises(ValueError):bootstrap(np.zeros((4096*16,1)),np.random.default_rng(1))
    with pytest.raises(ValueError):bootstrap(np.zeros((1024,1)),np.random.default_rng(1))

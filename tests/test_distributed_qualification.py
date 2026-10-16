"""CPU checks of the independent oracle and strict resume state comparator."""

import copy

import numpy as np
import pytest
import torch
from torch.nn import functional as F

from train.qualify_distributed import ACCUMULATION, MICRO_BATCH, WORLD_SIZE, build_model, compare_tree


def test_accumulated_rank_means_equal_concatenated_batch_gradients():
    torch.manual_seed(154)
    reference = build_model()
    rank_models = [copy.deepcopy(reference) for _ in range(WORLD_SIZE)]
    x = torch.randn(WORLD_SIZE, ACCUMULATION * MICRO_BATCH, 5)
    y = torch.randn(WORLD_SIZE, ACCUMULATION * MICRO_BATCH, 3)
    F.mse_loss(reference(x.flatten(0, 1)), y.flatten(0, 1)).backward()
    for rank, model in enumerate(rank_models):
        for micro in range(ACCUMULATION):
            lo, hi = micro * MICRO_BATCH, (micro + 1) * MICRO_BATCH
            (F.mse_loss(model(x[rank, lo:hi]), y[rank, lo:hi]) / ACCUMULATION).backward()
    for parameter, *rank_parameters in zip(reference.parameters(), *(m.parameters() for m in rank_models)):
        mean_gradient = sum(p.grad for p in rank_parameters) / WORLD_SIZE
        torch.testing.assert_close(mean_gradient, parameter.grad)
        # This fixture must detect the historical extra world-size division.
        assert not torch.allclose(mean_gradient / WORLD_SIZE, parameter.grad)


def test_resume_comparator_rejects_changed_moments_and_rng():
    expected = {"optimizer": {"exp_avg": torch.tensor([0.3, 0.7])},
                "rng": (np.array([5, 7], dtype=np.uint32), 12)}
    compare_tree(copy.deepcopy(expected), expected, exact=True)
    changed = copy.deepcopy(expected)
    changed["optimizer"]["exp_avg"][0] += 1e-7
    with pytest.raises(AssertionError, match="bitwise"):
        compare_tree(changed, expected, exact=True)
    changed = copy.deepcopy(expected)
    changed["rng"][0][0] += 1
    with pytest.raises(AssertionError, match="RNG"):
        compare_tree(changed, expected, exact=True)
    changed = copy.deepcopy(expected)
    del changed["optimizer"]["exp_avg"]
    with pytest.raises(AssertionError, match="keys"):
        compare_tree(changed, expected, exact=True)

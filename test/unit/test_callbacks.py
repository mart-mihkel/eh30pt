"""Tests for trainer callbacks."""

import pytest
import torch

from eh30pt.modeling.training.callbacks import _cuda_metrics


@pytest.mark.skipif(torch.cuda.is_available(), reason="CUDA is available")
def test_cuda_metrics_returns_empty_without_cuda() -> None:
    assert _cuda_metrics() == {}

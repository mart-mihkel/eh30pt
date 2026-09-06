"""Tests for pretrained model loading and parameter freezing."""

from torch.nn import Linear

from eh30pt.modeling.loader import _freeze


def test_freeze_default_freezes_all_params() -> None:
    model = Linear(4, 2)
    _freeze(model)

    assert all(not p.requires_grad for p in model.parameters())


def test_freeze_keeps_skipped_params_trainable() -> None:
    model = Linear(4, 2)
    _freeze(model, skip={"weight"})

    assert model.weight.requires_grad
    assert not model.bias.requires_grad

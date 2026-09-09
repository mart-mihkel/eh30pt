"""Dataset loading and schema integration tests."""

from typing import TYPE_CHECKING, Final

import pytest

from eh30pt.datasets.multinerd import Multinerd
from eh30pt.datasets.oblique import Oblique
from eh30pt.datasets.registry import get_dataset, get_labels, get_sys_prompt
from eh30pt.datasets.truncation import get_max_length
from eh30pt.modeling.arch import infer_architecture

if TYPE_CHECKING:
    from test.conftest import ModelSetup

_sample_size: Final = 32


def test_splits_and_columns(model_setup: ModelSetup) -> None:
    _, _, dataset, _ = model_setup

    data = dataset.data
    assert {"train", "validation"} <= set(data.keys())

    expected = {"input_ids", "attention_mask", "labels", "truncated"}
    for split in data:
        assert len(data[split]) > 0
        assert set(data[split].column_names) >= expected


def test_truncation_budget(model_setup: ModelSetup) -> None:
    _, tokenizer, dataset, _ = model_setup
    arch = infer_architecture(model_setup.config)

    max_length = get_max_length(tokenizer, 0)
    if max_length is None:
        pytest.skip("tokenizer has no length limit")

    sample = dataset.data["train"][:_sample_size]
    for ids, truncated in zip(sample["input_ids"], sample["truncated"], strict=True):
        if arch == "encoder":
            assert truncated == (len(ids) >= max_length)
        elif not truncated:
            assert len(ids) < max_length


def test_registry(model_setup: ModelSetup) -> None:
    _, tokenizer, _, _ = model_setup
    arch = infer_architecture(model_setup.config)

    multinerd = get_dataset("multinerd", tokenizer, arch)
    oblique = get_dataset("oblique", tokenizer, arch)

    assert isinstance(multinerd, Multinerd)
    assert isinstance(oblique, Oblique)

    assert get_sys_prompt("multinerd", tokenizer, arch) == multinerd.system_prompt
    assert get_sys_prompt("oblique", tokenizer, arch) == oblique.system_prompt


def test_unloaded_access_raises(model_setup: ModelSetup) -> None:
    tokenizer = model_setup.tokenizer
    arch = infer_architecture(model_setup.config)
    unloaded = type(model_setup.dataset)(tokenizer, arch)

    with pytest.raises(RuntimeError, match="not loaded"):
        _ = unloaded.data

    with pytest.raises(RuntimeError, match="not loaded"):
        _ = unloaded.raw_data


def test_get_labels_multinerd() -> None:
    labels = get_labels("multinerd")
    n_labels = 15

    assert labels[:3] == ("PER", "ORG", "LOC")
    assert len(labels) == n_labels

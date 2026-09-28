"""Tests for architecture family inference."""

from typing import TYPE_CHECKING, cast

import pytest
from transformers import BertConfig, GPT2Config, T5Config

from eh30pt.modeling.arch import get_pipeline_task, get_task_type, infer_architecture

if TYPE_CHECKING:
    from eh30pt.types import Architecture


def test_get_arch_override_short_circuits() -> None:
    assert infer_architecture(T5Config(), override="decoder") == "decoder"


def test_get_arch_encoder_decoder() -> None:
    assert infer_architecture(T5Config()) == "encoder-decoder"


def test_get_arch_encoder() -> None:
    assert infer_architecture(BertConfig()) == "encoder"


def test_get_arch_decoder() -> None:
    assert infer_architecture(GPT2Config()) == "decoder"


def test_get_pipeline_task_maps_architectures() -> None:
    assert get_pipeline_task("encoder") == "text-classification"
    assert get_pipeline_task("decoder") == "text-generation"
    assert get_pipeline_task("encoder-decoder") == "text-generation"


def test_get_pipeline_task_rejects_invalid() -> None:
    with pytest.raises(ValueError, match="invalid architecture"):
        get_pipeline_task(cast("Architecture", "unknown"))


def test_get_task_type_rejects_invalid() -> None:
    with pytest.raises(ValueError, match="invalid architecture"):
        get_task_type(cast("Architecture", "unknown"))


def test_get_task_type_maps_architectures() -> None:
    from peft import TaskType

    assert get_task_type("encoder") == TaskType.SEQ_CLS
    assert get_task_type("decoder") == TaskType.CAUSAL_LM
    assert get_task_type("encoder-decoder") == TaskType.SEQ_2_SEQ_LM

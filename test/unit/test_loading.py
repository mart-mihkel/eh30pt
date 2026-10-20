"""Tests for pretrained model loading and parameter freezing."""

import re
from typing import TYPE_CHECKING, Final, cast

import pytest
import torch
from peft import TaskType
from torch.nn import Linear

from eh30pt.modeling.loader import (
    _freeze,
    _load_with_attn_fallback,
    load_model,
)
from eh30pt.modeling.lora import (
    _lora_kwargs,
    build_lora_config,
    build_quantization_config,
)

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerFast

    from eh30pt.types import Architecture

_lora_r: Final = 16
_lora_alpha: Final = 32
_lora_dropout: Final = 0.05


def test_freeze_default_freezes_all_params() -> None:
    model = Linear(4, 2)
    _freeze(model)

    assert all(not p.requires_grad for p in model.parameters())


def test_freeze_keeps_skipped_params_trainable() -> None:
    model = Linear(4, 2)
    _freeze(model, skip={"weight"})

    assert model.weight.requires_grad
    assert not model.bias.requires_grad


def test_build_quantization_config_uses_double_quant_nf4() -> None:
    config = build_quantization_config()

    assert config.load_in_4bit is True
    assert config.bnb_4bit_quant_type == "nf4"
    assert config.bnb_4bit_use_double_quant is True
    assert config.bnb_4bit_compute_dtype == torch.bfloat16


@pytest.mark.parametrize(
    ("arch", "task"),
    [
        ("encoder", TaskType.SEQ_CLS),
        ("decoder", TaskType.CAUSAL_LM),
        ("encoder-decoder", TaskType.SEQ_2_SEQ_LM),
    ],
)
def test_build_lora_config_maps_task_type(arch: Architecture, task: TaskType) -> None:
    config = build_lora_config(_lora_r, _lora_alpha, _lora_dropout, arch)

    assert config.r == _lora_r
    assert config.lora_alpha == _lora_alpha
    assert config.lora_dropout == _lora_dropout
    assert config.task_type == task


@pytest.mark.parametrize("arch", ["decoder", "encoder-decoder"])
def test_build_lora_config_maps_sequence_classification(arch: Architecture) -> None:
    config = build_lora_config(
        _lora_r,
        _lora_alpha,
        _lora_dropout,
        arch,
        sequence_classification=True,
    )

    assert config.task_type == TaskType.SEQ_CLS


def test_build_lora_config_passes_explicit_target_modules() -> None:
    config = build_lora_config(
        _lora_r,
        _lora_alpha,
        _lora_dropout,
        "decoder",
        lora_kwargs={"target_modules": "q_proj|v_proj"},
    )

    assert config.target_modules == "q_proj|v_proj"


def test_build_lora_config_passes_explicit_target_parameters() -> None:
    params = ["model.layers.0.mlp.experts.0.gate_proj.weight"]
    config = build_lora_config(
        _lora_r,
        _lora_alpha,
        _lora_dropout,
        "decoder",
        lora_kwargs={"target_parameters": params},
    )

    assert config.target_parameters == params


def test_medgemma_lora_targets_only_the_text_language_model() -> None:
    kwargs = _lora_kwargs("google/medgemma-4b-it")
    target = kwargs["target_modules"]
    assert isinstance(target, str)

    assert re.fullmatch(target, "model.language_model.layers.0.self_attn.q_proj")
    assert re.fullmatch(target, "model.language_model.layers.0.mlp.down_proj")
    assert (
        re.fullmatch(
            target,
            "model.vision_tower.encoder.layers.0.self_attn.q_proj",
        )
        is None
    )


def test_lora_kwargs_leave_other_models_to_peft_defaults() -> None:
    assert _lora_kwargs("hf-internal-testing/tiny-random-gpt2") == {}


@pytest.mark.parametrize("arch", ["encoder", "decoder", "encoder-decoder"])
def test_load_model_rejects_extraction_dataset(arch: Architecture) -> None:
    tokenizer = cast("PreTrainedTokenizerFast", None)

    with pytest.raises(ValueError, match="extraction dataset"):
        load_model(
            tokenizer, "unused", "discont-extract/trigger", arch, head_only=False
        )


def test_load_with_attn_fallback_uses_requested_implementation() -> None:
    assert (
        _load_with_attn_fallback(
            lambda implementation: implementation, "flash_attention_2"
        )
        == "flash_attention_2"
    )


@pytest.mark.parametrize("error", [ValueError, ImportError])
def test_load_with_attn_fallback_retries_without_flash_attention(
    error: type[Exception],
) -> None:
    def load(implementation: str | None) -> str | None:
        if implementation is not None:
            raise error
        return implementation

    assert _load_with_attn_fallback(load, "flash_attention_2") is None


def test_load_with_attn_fallback_reraises_without_requested_attention() -> None:
    def load(_implementation: str | None) -> str | None:
        msg = "broken"
        raise ValueError(msg)

    with pytest.raises(ValueError, match="broken"):
        _load_with_attn_fallback(load, None)

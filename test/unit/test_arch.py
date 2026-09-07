"""Tests for architecture family inference."""

from transformers import BertConfig, GPT2Config, T5Config

from eh30pt.modeling.arch import infer_architecture


def test_get_arch_override_short_circuits() -> None:
    assert infer_architecture(T5Config(), override="decoder") == "decoder"


def test_get_arch_encoder_decoder() -> None:
    assert infer_architecture(T5Config()) == "encoder-decoder"


def test_get_arch_encoder() -> None:
    assert infer_architecture(BertConfig()) == "encoder"


def test_get_arch_decoder() -> None:
    assert infer_architecture(GPT2Config()) == "decoder"

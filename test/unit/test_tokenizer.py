"""Tests for tokenizer loading utilities."""

from typing import TYPE_CHECKING, cast
from unittest.mock import MagicMock

from transformers import BertConfig, GPT2Config, T5Config

from eh30pt.constants import UNSET_MAX_LENGTH
from eh30pt.modeling.tokenizer import cap_tokenizer_length

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerFast


def _tokenizer(max_length: int) -> PreTrainedTokenizerFast:
    return cast("PreTrainedTokenizerFast", MagicMock(model_max_length=max_length))


def test_cap_prefers_model_positions() -> None:
    config = GPT2Config(n_positions=512)
    tokenizer = _tokenizer(config.max_position_embeddings)
    cap_tokenizer_length(tokenizer, config)
    assert tokenizer.model_max_length == config.n_positions


def test_cap_uses_embedding_limit() -> None:
    config = BertConfig()
    tokenizer = _tokenizer(2 * config.max_position_embeddings)
    cap_tokenizer_length(tokenizer, config)
    assert tokenizer.model_max_length == config.max_position_embeddings


def test_cap_keeps_shorter_tokenizer_limit() -> None:
    config = GPT2Config(n_positions=512)
    tokenizer = _tokenizer(config.n_positions // 4)
    cap_tokenizer_length(tokenizer, config)
    assert tokenizer.model_max_length == config.n_positions // 4


def test_cap_skips_models_without_limit() -> None:
    tokenizer = _tokenizer(GPT2Config().max_position_embeddings)
    cap_tokenizer_length(tokenizer, T5Config())
    assert tokenizer.model_max_length == GPT2Config().max_position_embeddings


def test_cap_applies_to_unset_tokenizer_limit() -> None:
    config = BertConfig()
    tokenizer = _tokenizer(UNSET_MAX_LENGTH)
    cap_tokenizer_length(tokenizer, config)
    assert tokenizer.model_max_length == config.max_position_embeddings

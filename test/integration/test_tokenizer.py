"""Tokenizer length integration tests with real tokenizers."""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

import pytest
from transformers import BertConfig, GPT2Config, T5Config

from eh30pt.constants import UNSET_MAX_LENGTH
from eh30pt.datasets.mixin import DatasetMixin
from eh30pt.modeling.tokenizer import cap_tokenizer_length, load_tokenizer

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerFast

_bert_max_length: Final = 512
_expected_budget: Final = 504


@pytest.fixture
def fresh_gpt2() -> PreTrainedTokenizerFast:
    """Load an isolated GPT-2 tiny tokenizer."""
    return load_tokenizer("hf-internal-testing/tiny-random-gpt2")


@pytest.fixture
def fresh_bert() -> PreTrainedTokenizerFast:
    """Load an isolated BERT tiny tokenizer."""
    return load_tokenizer("hf-internal-testing/tiny-random-bert")


def test_cap_prefers_model_positions(fresh_gpt2: PreTrainedTokenizerFast) -> None:
    config = GPT2Config(n_positions=512)
    cap_tokenizer_length(fresh_gpt2, config)
    assert fresh_gpt2.model_max_length == config.n_positions


def test_cap_uses_embedding_limit(fresh_bert: PreTrainedTokenizerFast) -> None:
    config = BertConfig()
    fresh_bert.model_max_length = 2 * config.max_position_embeddings
    cap_tokenizer_length(fresh_bert, config)
    assert fresh_bert.model_max_length == config.max_position_embeddings


def test_cap_keeps_shorter_tokenizer_limit(
    fresh_gpt2: PreTrainedTokenizerFast,
) -> None:
    config = GPT2Config(n_positions=512)
    fresh_gpt2.model_max_length = config.n_positions // 4
    cap_tokenizer_length(fresh_gpt2, config)
    assert fresh_gpt2.model_max_length == config.n_positions // 4


def test_cap_skips_models_without_limit(
    fresh_gpt2: PreTrainedTokenizerFast,
) -> None:
    before = fresh_gpt2.model_max_length
    cap_tokenizer_length(fresh_gpt2, T5Config())
    assert fresh_gpt2.model_max_length == before


def test_cap_applies_to_unset_tokenizer_limit(
    fresh_bert: PreTrainedTokenizerFast,
) -> None:
    config = BertConfig()
    fresh_bert.model_max_length = UNSET_MAX_LENGTH
    cap_tokenizer_length(fresh_bert, config)
    assert fresh_bert.model_max_length == config.max_position_embeddings


def test_model_max_len_reserves_virtual_tokens_and_floors_to_pad_multiple(
    bert_tokenizer: PreTrainedTokenizerFast,
) -> None:
    assert bert_tokenizer.model_max_length == _bert_max_length

    assert DatasetMixin._model_max_len(bert_tokenizer, 6) == _expected_budget


def test_model_max_len_returns_none_for_unset_sentinel(
    modernbert_tokenizer: PreTrainedTokenizerFast,
) -> None:
    assert modernbert_tokenizer.model_max_length >= UNSET_MAX_LENGTH

    assert DatasetMixin._model_max_len(modernbert_tokenizer, 6) is None

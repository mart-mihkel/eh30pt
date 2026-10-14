"""Tokenizer length integration tests with real tokenizers."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from eh30pt.constants import PAD_MULTIPLE, UNSET_MAX_LENGTH
from eh30pt.datasets.mixin import DatasetMixin
from eh30pt.modeling.tokenizer import cap_tokenizer_length, load_tokenizer

if TYPE_CHECKING:
    from transformers import PreTrainedConfig, PreTrainedTokenizerFast


@pytest.fixture
def fresh_gpt2() -> PreTrainedTokenizerFast:
    """Load an isolated GPT-2 tiny tokenizer."""
    return load_tokenizer("hf-internal-testing/tiny-random-gpt2")


@pytest.fixture
def fresh_bert() -> PreTrainedTokenizerFast:
    """Load an isolated BERT tiny tokenizer."""
    return load_tokenizer("hf-internal-testing/tiny-random-bert")


def test_cap_prefers_model_positions(
    fresh_gpt2: PreTrainedTokenizerFast,
    gpt2_config: PreTrainedConfig,
) -> None:
    cap_tokenizer_length(fresh_gpt2, gpt2_config)
    assert fresh_gpt2.model_max_length == gpt2_config.n_positions


def test_cap_uses_embedding_limit(
    fresh_bert: PreTrainedTokenizerFast,
    bert_config: PreTrainedConfig,
) -> None:
    limit = bert_config.max_position_embeddings
    fresh_bert.model_max_length = 2 * limit
    cap_tokenizer_length(fresh_bert, bert_config)
    assert fresh_bert.model_max_length == limit


def test_cap_keeps_shorter_tokenizer_limit(
    fresh_gpt2: PreTrainedTokenizerFast,
    gpt2_config: PreTrainedConfig,
) -> None:
    limit = gpt2_config.n_positions // 4
    fresh_gpt2.model_max_length = limit
    cap_tokenizer_length(fresh_gpt2, gpt2_config)
    assert fresh_gpt2.model_max_length == limit


def test_cap_skips_models_without_limit(
    fresh_gpt2: PreTrainedTokenizerFast,
    t5_config: PreTrainedConfig,
) -> None:
    before = fresh_gpt2.model_max_length
    cap_tokenizer_length(fresh_gpt2, t5_config)
    assert fresh_gpt2.model_max_length == before


def test_cap_applies_to_unset_tokenizer_limit(
    fresh_bert: PreTrainedTokenizerFast,
    bert_config: PreTrainedConfig,
) -> None:
    fresh_bert.model_max_length = UNSET_MAX_LENGTH
    cap_tokenizer_length(fresh_bert, bert_config)
    assert fresh_bert.model_max_length == bert_config.max_position_embeddings


def test_model_max_len_reserves_virtual_tokens_and_floors_to_pad_multiple(
    bert_tokenizer: PreTrainedTokenizerFast,
    bert_config: PreTrainedConfig,
) -> None:
    n_virtual = 6
    assert bert_tokenizer.model_max_length == bert_config.max_position_embeddings

    expected = (
        (bert_tokenizer.model_max_length - n_virtual) // PAD_MULTIPLE
    ) * PAD_MULTIPLE
    assert DatasetMixin._model_max_len(bert_tokenizer, n_virtual) == expected


def test_model_max_len_returns_none_for_unset_sentinel(
    modernbert_tokenizer: PreTrainedTokenizerFast,
) -> None:
    assert modernbert_tokenizer.model_max_length >= UNSET_MAX_LENGTH

    assert DatasetMixin._model_max_len(modernbert_tokenizer, 6) is None

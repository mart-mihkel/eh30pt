"""Metric accumulation integration tests."""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

import numpy as np
import pytest
from transformers import EvalPrediction

from eh30pt.modeling.metrics import (
    MetricsAccumulator,
)

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerFast

_batch: Final = 2
_seq: Final = 4


@pytest.fixture
def bert_token_id(bert_tokenizer: PreTrainedTokenizerFast) -> int:
    """Provide a non-special token id from the bert vocabulary."""
    pad_token_id = bert_tokenizer.pad_token_id
    assert pad_token_id is not None
    return (pad_token_id + 1) % bert_tokenizer.vocab_size


@pytest.fixture
def gpt2_token_id(gpt2_tokenizer: PreTrainedTokenizerFast) -> int:
    """Provide a non-special token id from the gpt2 vocabulary."""
    pad_token_id = gpt2_tokenizer.pad_token_id
    assert pad_token_id is not None
    return (pad_token_id + 1) % gpt2_tokenizer.vocab_size


@pytest.fixture
def t5_token_id(t5_tokenizer: PreTrainedTokenizerFast) -> int:
    """Provide a non-special token id from the t5 vocabulary."""
    pad_token_id = t5_tokenizer.pad_token_id
    assert pad_token_id is not None
    return (pad_token_id + 1) % t5_tokenizer.vocab_size


@pytest.fixture
def encoder_accumulator(
    bert_tokenizer: PreTrainedTokenizerFast,
) -> MetricsAccumulator:
    """Provide an accumulator for encoder models."""
    return MetricsAccumulator(bert_tokenizer, "encoder")


@pytest.fixture
def encoder_pred(
    bert_tokenizer: PreTrainedTokenizerFast,
    bert_token_id: int,
) -> EvalPrediction:
    """Provide an encoder eval batch voting for a constant token id."""
    logits = np.zeros((_batch, _seq, bert_tokenizer.vocab_size))
    logits[:, :, bert_token_id] = 1.0
    return EvalPrediction(
        predictions=logits,
        label_ids=np.full((_batch, _seq), bert_token_id),
    )


@pytest.fixture
def decoder_accumulator(
    gpt2_tokenizer: PreTrainedTokenizerFast,
) -> MetricsAccumulator:
    """Provide an accumulator for decoder models."""
    return MetricsAccumulator(gpt2_tokenizer, "decoder")


@pytest.fixture
def decoder_pred(
    gpt2_tokenizer: PreTrainedTokenizerFast,
    gpt2_token_id: int,
) -> EvalPrediction:
    """Provide a decoder eval batch voting for a constant token id."""
    logits = np.zeros((_batch, _seq, gpt2_tokenizer.vocab_size))
    logits[:, :, gpt2_token_id] = 1.0
    return EvalPrediction(
        predictions=logits,
        label_ids=np.full((_batch, _seq), gpt2_token_id),
    )


@pytest.fixture
def long_decoder_pred(
    gpt2_tokenizer: PreTrainedTokenizerFast,
    gpt2_token_id: int,
) -> EvalPrediction:
    """Provide a decoder eval batch with one extra trailing logit position."""
    logits = np.zeros((_batch, _seq + 1, gpt2_tokenizer.vocab_size))
    logits[:, :, gpt2_token_id] = 1.0
    return EvalPrediction(
        predictions=logits,
        label_ids=np.full((_batch, _seq), gpt2_token_id),
    )


@pytest.fixture
def seq2seq_accumulator(
    t5_tokenizer: PreTrainedTokenizerFast,
) -> MetricsAccumulator:
    """Provide an accumulator for encoder-decoder models."""
    return MetricsAccumulator(t5_tokenizer, "encoder-decoder")


@pytest.fixture
def seq2seq_pred(t5_token_id: int) -> EvalPrediction:
    """Provide an encoder-decoder eval batch with tuple-wrapped id outputs."""
    ids = np.full((_batch, _seq), t5_token_id)
    return EvalPrediction(predictions=(ids,), label_ids=(ids,))


def test_accumulates_batches_without_computing(
    encoder_accumulator: MetricsAccumulator,
    encoder_pred: EvalPrediction,
) -> None:
    assert encoder_accumulator(encoder_pred, compute_result=False) == {}

    assert encoder_accumulator._bar is not None
    assert encoder_accumulator._bar.n == _batch

    encoder_accumulator(encoder_pred, compute_result=False)

    assert encoder_accumulator._bar is not None
    assert encoder_accumulator._bar.n == 2 * _batch


def test_compute_returns_perfect_metrics_and_resets(
    encoder_accumulator: MetricsAccumulator,
    encoder_pred: EvalPrediction,
) -> None:
    encoder_accumulator(encoder_pred, compute_result=False)
    metrics = encoder_accumulator(encoder_pred)

    assert metrics == {
        "accuracy": 1.0,
        "precision": 1.0,
        "recall": 1.0,
        "f1": 1.0,
    }
    assert encoder_accumulator._labels == []
    assert encoder_accumulator._preds == []
    assert encoder_accumulator._bar is None


def test_reset_drops_batches_and_closes_bar(
    encoder_accumulator: MetricsAccumulator,
    encoder_pred: EvalPrediction,
) -> None:
    encoder_accumulator(encoder_pred, compute_result=False)
    encoder_accumulator.reset()

    assert encoder_accumulator._labels == []
    assert encoder_accumulator._preds == []
    assert encoder_accumulator._bar is None


def test_decoder_truncates_longer_predictions(
    decoder_accumulator: MetricsAccumulator,
    long_decoder_pred: EvalPrediction,
) -> None:
    decoder_accumulator(long_decoder_pred, compute_result=False)

    assert len(decoder_accumulator._preds) == _batch
    assert all(len(row) == _seq for row in decoder_accumulator._preds)


def test_decoder_compute_scores_shifted_sequences(
    decoder_accumulator: MetricsAccumulator,
    decoder_pred: EvalPrediction,
) -> None:
    metrics = decoder_accumulator(decoder_pred)

    assert metrics["accuracy"] == 1.0


def test_encoder_decoder_unwraps_tuple_outputs(
    seq2seq_accumulator: MetricsAccumulator,
    seq2seq_pred: EvalPrediction,
) -> None:
    metrics = seq2seq_accumulator(seq2seq_pred)

    assert metrics["accuracy"] == 1.0

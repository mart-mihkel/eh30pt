"""Metric accumulation integration tests from real model outputs."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
import torch
from transformers import EvalPrediction

from eh30pt.modeling.collator import get_collator
from eh30pt.modeling.metrics import (
    MetricsAccumulator,
)

if TYPE_CHECKING:
    from collections.abc import Iterable

    from transformers import PreTrainedModel, PreTrainedTokenizerFast

    from eh30pt.types import Architecture, DataSplits


def _first_rows(split: Iterable[dict], n: int) -> list[dict]:
    from itertools import islice

    return list(islice(split, n))


def _forward_logits(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerFast,
    arch: Architecture,
    splits: DataSplits,
    n: int = 2,
) -> EvalPrediction:
    rows = _first_rows(splits.train, n)
    assert rows
    collated = get_collator(tokenizer, arch)(rows)
    device = next(model.parameters()).device
    batch = {
        k: v.to(device) if isinstance(v, torch.Tensor) else v
        for k, v in collated.items()
        if k != "truncated"
    }
    model.eval()
    with torch.inference_mode():
        output = model(
            **{
                k: v
                for k, v in batch.items()
                if k in ("input_ids", "attention_mask", "labels")
            }
        )
    logits = output.logits.float().detach().cpu().numpy()
    labels = batch["labels"].detach().cpu().numpy()
    return EvalPrediction(predictions=logits, label_ids=labels)


@pytest.fixture
def bert_token_id(bert_tokenizer: PreTrainedTokenizerFast) -> int:
    """Provide a real non-special token id from encoding a registry prompt."""
    from eh30pt.datasets.registry import get_sys_prompt

    ids = bert_tokenizer(get_sys_prompt("oblique"), truncation=True)["input_ids"]
    real = next(
        i
        for i in ids
        if i
        not in (
            bert_tokenizer.pad_token_id,
            bert_tokenizer.cls_token_id,
            bert_tokenizer.sep_token_id,
        )
    )
    return int(real)


@pytest.fixture
def gpt2_token_id(gpt2_tokenizer: PreTrainedTokenizerFast) -> int:
    """Provide a real non-special token id from encoding a registry prompt."""
    from eh30pt.datasets.registry import get_sys_prompt

    ids = gpt2_tokenizer(get_sys_prompt("oblique"), truncation=True)["input_ids"]
    real = next(i for i in ids if i != gpt2_tokenizer.pad_token_id)
    return int(real)


@pytest.fixture
def t5_token_id(t5_tokenizer: PreTrainedTokenizerFast) -> int:
    """Provide a real non-special token id from encoding a registry prompt."""
    from eh30pt.datasets.registry import get_sys_prompt

    ids = t5_tokenizer(get_sys_prompt("oblique"), truncation=True)["input_ids"]
    real = next(i for i in ids if i != t5_tokenizer.pad_token_id)
    return int(real)


@pytest.fixture
def encoder_accumulator(
    bert_tokenizer: PreTrainedTokenizerFast,
) -> MetricsAccumulator:
    """Provide an accumulator for encoder models."""
    return MetricsAccumulator(bert_tokenizer, "encoder")


@pytest.fixture
def encoder_pred(
    bert_tokenizer: PreTrainedTokenizerFast,
    oblique_bert_model: PreTrainedModel,
    oblique_bert_dataset: DataSplits,
) -> EvalPrediction:
    """Provide a real encoder forward pass relabeled to its own argmax."""
    pred = _forward_logits(
        oblique_bert_model, bert_tokenizer, "encoder", oblique_bert_dataset
    )
    labels = np.argmax(pred.predictions, axis=-1)
    return EvalPrediction(predictions=pred.predictions, label_ids=labels)


@pytest.fixture
def decoder_accumulator(
    gpt2_tokenizer: PreTrainedTokenizerFast,
) -> MetricsAccumulator:
    """Provide an accumulator for decoder models."""
    return MetricsAccumulator(gpt2_tokenizer, "decoder")


@pytest.fixture
def decoder_pred(
    gpt2_tokenizer: PreTrainedTokenizerFast,
    oblique_gpt2_model: PreTrainedModel,
    oblique_gpt2_dataset: DataSplits,
) -> EvalPrediction:
    """Provide real decoder labels with logits voting for the shifted sequence."""
    pred = _forward_logits(
        oblique_gpt2_model, gpt2_tokenizer, "decoder", oblique_gpt2_dataset
    )
    labels = pred.label_ids
    assert isinstance(labels, np.ndarray)
    shifted = np.empty_like(labels)
    shifted[:, :-1] = labels[:, 1:]
    shifted[:, -1:] = labels[:, -1:]
    vocab = gpt2_tokenizer.vocab_size
    logits = np.zeros((labels.shape[0], labels.shape[1], vocab), dtype=np.float32)
    clipped = np.clip(shifted, 0, vocab - 1)
    np.put_along_axis(logits, clipped[:, :, None], 1.0, axis=-1)
    return EvalPrediction(predictions=logits, label_ids=labels)


@pytest.fixture
def long_decoder_pred(
    decoder_pred: EvalPrediction,
) -> EvalPrediction:
    """Provide a real decoder batch with one extra trailing logit position."""
    logits = decoder_pred.predictions
    assert isinstance(logits, np.ndarray)
    extra = logits[:, -1:, :]
    long_logits = np.concatenate([logits, extra], axis=1)
    return EvalPrediction(predictions=long_logits, label_ids=decoder_pred.label_ids)


@pytest.fixture
def seq2seq_accumulator(
    t5_tokenizer: PreTrainedTokenizerFast,
) -> MetricsAccumulator:
    """Provide an accumulator for encoder-decoder models."""
    return MetricsAccumulator(t5_tokenizer, "encoder-decoder")


@pytest.fixture
def seq2seq_pred(
    t5_tokenizer: PreTrainedTokenizerFast,
) -> EvalPrediction:
    """Provide real label token ids wrapped as tuple outputs."""
    from eh30pt.datasets.registry import get_few_shots

    shots = get_few_shots("oblique", 2)
    encoded = [t5_tokenizer(shot.label, truncation=True)["input_ids"] for shot in shots]
    width = max(len(ids) for ids in encoded)
    pad = t5_tokenizer.pad_token_id
    assert pad is not None
    ids = np.array(
        [row + [pad] * (width - len(row)) for row in encoded], dtype=np.int64
    )
    return EvalPrediction(predictions=(ids,), label_ids=(ids,))


def test_accumulates_batches_without_computing(
    encoder_accumulator: MetricsAccumulator,
    encoder_pred: EvalPrediction,
) -> None:
    batch = len(encoder_pred.label_ids)
    assert encoder_accumulator(encoder_pred, compute_result=False) == {}

    assert encoder_accumulator._bar is not None
    assert encoder_accumulator._bar.n == batch

    encoder_accumulator(encoder_pred, compute_result=False)

    assert encoder_accumulator._bar is not None
    assert encoder_accumulator._bar.n == 2 * batch


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


def test_compute_retains_decoded_predictions(
    encoder_accumulator: MetricsAccumulator,
    encoder_pred: EvalPrediction,
    decoder_accumulator: MetricsAccumulator,
    decoder_pred: EvalPrediction,
) -> None:
    encoder_accumulator(encoder_pred)

    assert len(encoder_accumulator.last_predictions) == len(encoder_pred.label_ids)
    assert encoder_accumulator.last_labels == encoder_accumulator.last_predictions

    decoder_accumulator(decoder_pred)

    assert len(decoder_accumulator.last_predictions) == len(decoder_pred.label_ids)
    assert all(isinstance(pred, str) for pred in decoder_accumulator.last_predictions)


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

    label_ids = long_decoder_pred.label_ids
    assert isinstance(label_ids, np.ndarray)
    label_len = label_ids.shape[1]
    assert len(decoder_accumulator._preds) == len(label_ids)
    assert all(len(row) == label_len for row in decoder_accumulator._preds)


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

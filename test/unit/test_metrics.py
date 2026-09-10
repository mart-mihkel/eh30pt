"""Metric accumulation unit tests."""

from __future__ import annotations

from typing import Final, cast

import numpy as np
import torch
from transformers import EvalPrediction

from eh30pt.modeling.metrics import (
    StreamingClassificationMetrics,
    _batch_to_numpy,
    _filter_other,
    compute_classification_metrics,
)

_batch: Final = 2
_classes: Final = 3
_seq: Final = 4


def test_batch_to_numpy_handles_tensors_and_tuples() -> None:
    labels = torch.zeros(_batch, _seq, dtype=torch.long)
    logits = torch.zeros(_batch, _seq, _classes)
    pred = EvalPrediction(
        predictions=cast("tuple[np.ndarray]", (logits,)),
        label_ids=cast("tuple[np.ndarray]", (labels,)),
    )

    batch_labels, batch_preds = _batch_to_numpy(pred)

    assert isinstance(batch_labels, np.ndarray)
    assert batch_labels.shape == (_batch, _seq)
    assert batch_preds.shape == (_batch, _seq)


def test_filter_other_maps_unseen_predictions() -> None:
    assert _filter_other(["a", "b"], ["a", "c"]) == ["a", "other"]


def test_compute_classification_metrics_on_perfect_predictions() -> None:
    metrics = compute_classification_metrics(["a", "b", "a"], ["a", "b", "a"])

    assert metrics == {
        "accuracy": 1.0,
        "precision": 1.0,
        "recall": 1.0,
        "f1": 1.0,
    }


def test_streaming_matches_sklearn_on_perfect_predictions() -> None:
    labels = ["a", "b", "a", "c", "b"]
    accumulator = StreamingClassificationMetrics()
    for label in labels:
        accumulator.update(label, label)

    assert accumulator.finalize() == compute_classification_metrics(labels, labels)


def test_streaming_matches_sklearn_with_other_fallback() -> None:
    labels = ["a", "b", "a", "c", "b", "c", "a"]
    preds = ["a", "other", "b", "c", "b", "other", "a"]
    accumulator = StreamingClassificationMetrics()
    for label, pred in zip(labels, preds, strict=True):
        accumulator.update(label, pred)

    assert accumulator.finalize() == compute_classification_metrics(labels, preds)


def test_streaming_matches_sklearn_on_unseen_prediction() -> None:
    labels = ["a", "b", "a", "b"]
    preds = ["b", "c", "a", "c"]
    accumulator = StreamingClassificationMetrics()
    for label, pred in zip(labels, preds, strict=True):
        accumulator.update(label, pred)

    assert accumulator.finalize() == compute_classification_metrics(labels, preds)


def test_streaming_matches_sklearn_on_single_class() -> None:
    labels = ["a", "a", "a"]
    preds = ["a", "a", "a"]
    accumulator = StreamingClassificationMetrics()
    for label, pred in zip(labels, preds, strict=True):
        accumulator.update(label, pred)

    assert accumulator.finalize() == compute_classification_metrics(labels, preds)


def test_streaming_finalize_without_updates_returns_zeros() -> None:
    assert StreamingClassificationMetrics().finalize() == {
        "accuracy": 0.0,
        "precision": 0.0,
        "recall": 0.0,
        "f1": 0.0,
    }

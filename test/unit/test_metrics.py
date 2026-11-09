"""Metric accumulation unit tests."""

from __future__ import annotations

from typing import TYPE_CHECKING, Final, cast

import numpy as np
import pytest
import torch
from transformers import EvalPrediction

from eh30pt.modeling.metrics import (
    MetricsAccumulator,
    StreamingClassificationMetrics,
    _batch_to_numpy,
    _filter_other,
    compute_classification_metrics,
)

_batch: Final = 2
_classes: Final = 3
_seq: Final = 4

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerFast


def test_batch_to_numpy_handles_tensors_and_tuples() -> None:
    labels = torch.zeros(_batch, _seq, dtype=torch.long)
    logits = torch.zeros(_batch, _seq, _classes)
    pred = EvalPrediction(
        predictions=cast("tuple[np.ndarray]", (logits,)),
        label_ids=cast("tuple[np.ndarray]", (labels,)),
    )

    batch_labels, batch_preds, batch_logits = _batch_to_numpy(pred)

    assert isinstance(batch_labels, np.ndarray)
    assert batch_labels.shape == (_batch, _seq)
    assert batch_preds.shape == (_batch, _seq)
    assert batch_logits.shape == (_batch, _seq, _classes)


def test_filter_other_maps_unseen_predictions() -> None:
    assert _filter_other(["a", "b"], ["a", "c"]) == ["a", "..."]


def test_compute_classification_metrics_on_perfect_predictions() -> None:
    metrics = compute_classification_metrics(["a", "b", "a"], ["a", "b", "a"])

    assert metrics == {
        "accuracy": 1.0,
        "precision": 1.0,
        "recall": 1.0,
        "f1": 1.0,
        "tpr": 1.0,
        "fpr": 0.0,
        "mcc": 1.0,
        "kappa": 1.0,
    }


def test_compute_classification_metrics_reports_hard_rates() -> None:
    """TPR/FPR come from the confusion matrix, MCC and kappa from agreement."""
    metrics = compute_classification_metrics(["a", "a", "b", "b"], ["a", "b", "b", "b"])

    assert metrics["tpr"] == pytest.approx(0.75)
    assert metrics["fpr"] == pytest.approx(0.25)
    assert metrics["mcc"] == pytest.approx(0.5773502691896258)
    assert metrics["kappa"] == pytest.approx(0.5)


def test_compute_classification_metrics_zeroes_rates_for_single_class() -> None:
    """One-class targets have no positives or negatives, so the rates are 0."""
    metrics = compute_classification_metrics(["a", "a"], ["a", "a"])

    assert metrics["tpr"] == 0.0
    assert metrics["fpr"] == 0.0
    assert metrics["mcc"] == 0.0
    assert metrics["kappa"] == 0.0


def test_compute_classification_metrics_scores_binary_positive_class() -> None:
    """A binary target reports the positive class instead of averaging classes."""
    labels = ["ei"] * 95 + ["jah"] * 165
    preds = ["jah"] * 260

    metrics = compute_classification_metrics(labels, preds, positive_label="jah")

    precision = 165 / 260
    assert metrics["precision"] == pytest.approx(precision)
    assert metrics["recall"] == pytest.approx(1.0)
    assert metrics["f1"] == pytest.approx(2 * precision / (precision + 1))
    assert metrics["tpr"] == pytest.approx(1.0)
    assert metrics["fpr"] == pytest.approx(1.0)


def test_compute_classification_metrics_positive_with_other_prediction() -> None:
    """An unseen prediction stays out of the positive class's one-vs-rest score."""
    metrics = compute_classification_metrics(
        ["ei", "ei", "jah", "jah"],
        ["ei", "other", "jah", "jah"],
        positive_label="jah",
    )

    assert metrics["precision"] == pytest.approx(1.0)
    assert metrics["recall"] == pytest.approx(1.0)
    assert metrics["f1"] == pytest.approx(1.0)
    assert metrics["tpr"] == pytest.approx(1.0)
    assert metrics["fpr"] == pytest.approx(0.0)


def test_compute_classification_metrics_binary_scores_zero_when_all_negative() -> None:
    """All-negative predictions score the positive class zero."""
    metrics = compute_classification_metrics(
        ["ei", "ei", "jah", "jah"],
        ["ei", "ei", "ei", "ei"],
        positive_label="jah",
    )

    assert metrics["precision"] == pytest.approx(0.0)
    assert metrics["recall"] == pytest.approx(0.0)
    assert metrics["f1"] == pytest.approx(0.0)
    assert metrics["tpr"] == pytest.approx(0.0)
    assert metrics["fpr"] == pytest.approx(0.0)


def test_compute_classification_metrics_scores_positive_one_vs_rest() -> None:
    """A supplied positive class is scored one-vs-rest rather than as macro."""
    metrics = compute_classification_metrics(
        ["a", "a", "b", "b", "c", "c"],
        ["a", "b", "b", "b", "c", "c"],
        positive_label="a",
    )

    assert metrics["precision"] == pytest.approx(1.0)
    assert metrics["recall"] == pytest.approx(0.5)
    assert metrics["f1"] == pytest.approx(2 / 3)


def test_compute_classification_metrics_falls_back_when_positive_absent() -> None:
    """A positive class missing from the gold labels leaves scoring as macro."""
    metrics = compute_classification_metrics(
        ["ei", "ei"], ["ei", "jah"], positive_label="jah"
    )

    assert metrics["recall"] == pytest.approx(0.25)


def test_encoder_accumulator_scores_binary_positive_class() -> None:
    """The encoder accumulator applies the positive class to a binary target."""
    accumulator = MetricsAccumulator(
        cast("PreTrainedTokenizerFast", None), "encoder", positive_label=0
    )
    logits = np.array([[0.9, 0.1]] * 4)
    labels = np.array([0, 0, 1, 1])

    metrics = accumulator(EvalPrediction(predictions=logits, label_ids=labels))

    assert metrics["precision"] == pytest.approx(0.5)
    assert metrics["recall"] == pytest.approx(1.0)


def test_sequence_classification_accumulator_scores_decoder_logits() -> None:
    """A decoder classification head reports class logits, not token sequences."""
    accumulator = MetricsAccumulator(
        cast("PreTrainedTokenizerFast", None),
        "decoder",
        sequence_classification=True,
    )
    logits = np.array([[0.9, 0.1]] * 4)
    labels = np.array([0, 0, 1, 1])

    metrics = accumulator(EvalPrediction(predictions=logits, label_ids=labels))

    assert metrics["accuracy"] == pytest.approx(0.5)
    assert accumulator.last_predictions == [0, 0, 0, 0]


def test_compute_classification_metrics_reports_binary_auc() -> None:
    labels = np.array([0, 0, 1, 1])
    preds = np.array([0, 1, 1, 1])
    scores = np.array([[0.9, 0.1], [0.6, 0.4], [0.3, 0.7], [0.2, 0.8]])

    metrics = compute_classification_metrics(labels, preds, scores=scores)

    assert metrics["auc"] == pytest.approx(1.0)


def test_compute_classification_metrics_reports_macro_auc() -> None:
    """Multiclass AUC averages one-vs-rest over the present classes only."""
    labels = np.array([1, 1, 2, 2])
    scores = np.array(
        [[0.1, 0.8, 0.1], [0.1, 0.7, 0.2], [0.1, 0.2, 0.7], [0.2, 0.5, 0.3]]
    )

    metrics = compute_classification_metrics(labels, labels, scores=scores)

    assert metrics["auc"] == pytest.approx(1.0)


def test_compute_classification_metrics_auc_undefined_for_single_class() -> None:
    labels = np.array([0, 0])
    scores = np.array([[0.7, 0.3], [0.6, 0.4]])

    metrics = compute_classification_metrics(labels, labels, scores=scores)

    assert metrics["auc"] == 0.0


def test_compute_classification_metrics_omits_auc_without_scores() -> None:
    metrics = compute_classification_metrics(["a", "b"], ["a", "b"])

    assert "auc" not in metrics


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


def test_streaming_scores_binary_positive_class() -> None:
    """Streaming evaluation honors the binary positive class."""
    accumulator = StreamingClassificationMetrics(positive_label="jah")
    for label, pred in [("ei", "jah"), ("jah", "jah")]:
        accumulator.update(label, pred)

    metrics = accumulator.finalize()

    assert metrics["precision"] == pytest.approx(0.5)
    assert metrics["recall"] == pytest.approx(1.0)


def test_streaming_finalize_without_updates_returns_zeros() -> None:
    assert StreamingClassificationMetrics().finalize() == {
        "accuracy": 0.0,
        "precision": 0.0,
        "recall": 0.0,
        "f1": 0.0,
        "tpr": 0.0,
        "fpr": 0.0,
        "mcc": 0.0,
        "kappa": 0.0,
    }

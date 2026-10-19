"""Streaming metric computation for encoder, decoder, and encoder-decoder Trainers."""

from collections import Counter
from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch import Tensor
from tqdm.rich import tqdm
from transformers import EvalPrediction  # noqa: TC002

from eh30pt.constants import SENTINEL_TOKEN
from eh30pt.logging import format_counts, logger

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerFast

    from eh30pt.types import Architecture


class MetricsAccumulator(Callable):
    """Collect per-batch eval outputs and computes metrics on the final call."""

    def __init__(self, tokenizer: PreTrainedTokenizerFast, arch: Architecture) -> None:
        """Store the tokenizer and architecture."""
        self._tokenizer = tokenizer
        self._arch = arch
        self._labels: list[np.ndarray] = []
        self._preds: list[np.ndarray] = []
        self._probs: list[np.ndarray] = []
        self._bar: tqdm | None = None
        self.last_labels: list[str | int] = []
        self.last_predictions: list[str | int] = []

    def __call__(
        self,
        eval_pred: EvalPrediction,
        *,
        compute_result: bool = True,
    ) -> dict[str, float]:
        """Accumulate one batch, computing metrics when `compute_result`."""
        collected = len(self._labels)
        self._extend(eval_pred)

        if self._bar is None:
            self._bar = tqdm(
                desc="                             evaluating",
                unit="samples",
            )

        self._bar.update(len(self._labels) - collected)

        if not compute_result:
            return {}

        try:
            return self._compute()
        finally:
            self.reset()

    def reset(self) -> None:
        """Drop all accumulated batches."""
        self._labels.clear()
        self._preds.clear()
        self._probs.clear()
        if self._bar is not None:
            self._bar.close()
            self._bar = None

    def _extend(self, eval_pred: EvalPrediction) -> None:
        if self._arch == "encoder-decoder":
            batch_labels = eval_pred.label_ids
            batch_preds = eval_pred.predictions

            if isinstance(batch_labels, tuple):
                batch_labels = batch_labels[0]

            if isinstance(batch_preds, tuple):
                batch_preds = batch_preds[0]

            self._labels.extend(batch_labels)
            self._preds.extend(batch_preds)
            return

        labels, preds, logits = _batch_to_numpy(eval_pred)
        if self._arch == "decoder":
            _, label_dim = labels.shape
            _, pred_dim = preds.shape
            if label_dim < pred_dim:
                preds = preds[:, pred_dim - label_dim :]

            self._labels.extend(labels)
            self._preds.extend(preds)
            return

        self._labels.extend(labels)
        self._preds.extend(preds)
        self._probs.extend(_softmax(logits))

    def _compute(self) -> dict[str, float]:
        if self._arch == "encoder":
            all_labels = np.array(self._labels)
            all_preds = np.array(self._preds)
            all_probs = np.array(self._probs)
            mask = all_labels != SENTINEL_TOKEN

            _labels_count = Counter(all_labels[mask].tolist())
            _preds_count = Counter(all_preds[mask].tolist())

            logger.debug("labels: %s", format_counts(_labels_count))
            logger.debug("predictions: %s", format_counts(_preds_count))

            self.last_labels = [int(value) for value in all_labels[mask].tolist()]
            self.last_predictions = [int(value) for value in all_preds[mask].tolist()]

            return compute_classification_metrics(
                all_labels[mask],
                all_preds[mask],
                scores=all_probs[mask],
            )

        labels = []
        preds = []
        for label, pred in zip(self._labels, self._preds, strict=True):
            if self._arch == "decoder":
                shifted_label = label[1:]
                shifted_pred = pred[:-1]
            else:
                shifted_label = label
                shifted_pred = pred

            label_mask = shifted_label != SENTINEL_TOKEN
            labels.append(shifted_label[label_mask])

            if self._arch == "decoder":
                preds.append(shifted_pred[label_mask])
            else:
                pred_mask = shifted_pred != self._tokenizer.pad_token_id
                preds.append(shifted_pred[pred_mask])

        references = self._tokenizer.batch_decode(labels, skip_special_tokens=True)
        predictions = self._tokenizer.batch_decode(preds, skip_special_tokens=True)
        predictions = _filter_other(references, predictions)

        _refs_count = Counter(references)
        _preds_count = Counter(predictions)

        logger.debug("references: %s", format_counts(_refs_count))
        logger.debug("predictions: %s", format_counts(_preds_count))

        self.last_labels = list(references)
        self.last_predictions = list(predictions)

        return compute_classification_metrics(references, predictions)


class StreamingClassificationMetrics:
    """Accumulate prediction pairs and derive metrics with scikit-learn."""

    def __init__(self) -> None:
        """Initialize empty label and prediction lists."""
        self._labels: list[str] = []
        self._predictions: list[str] = []

    def update(self, label: str, prediction: str) -> None:
        """Record one gold label and prediction pair."""
        self._labels.append(label)
        self._predictions.append(prediction)

    def finalize(self) -> dict[str, float]:
        """Derive the classification metrics for the accumulated pairs."""
        if not self._labels:
            return {
                "accuracy": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
                "tpr": 0.0,
                "fpr": 0.0,
                "mcc": 0.0,
                "kappa": 0.0,
            }

        return compute_classification_metrics(self._labels, self._predictions)


def _batch_to_numpy(
    eval_pred: EvalPrediction,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    batch_labels = eval_pred.label_ids
    if isinstance(batch_labels, tuple):
        batch_labels = batch_labels[0]

    if isinstance(batch_labels, Tensor):
        batch_labels = batch_labels.detach().cpu().numpy()

    batch_logits = eval_pred.predictions
    if isinstance(batch_logits, tuple):
        batch_logits = batch_logits[0]

    if isinstance(batch_logits, Tensor):
        batch_logits = batch_logits.detach().cpu().numpy()

    batch_preds = np.argmax(batch_logits, axis=-1)
    return batch_labels, batch_preds, batch_logits


def _softmax(logits: np.ndarray) -> np.ndarray:
    """Normalize logits into per-position class probabilities."""
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exponentiated = np.exp(shifted)
    return exponentiated / exponentiated.sum(axis=-1, keepdims=True)


def _filter_other(references: list[str], predictions: list[str]) -> list[str]:
    refset = set(references)
    return [pred if pred in refset else "other" for pred in predictions]


def _safe_rate(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    """Divide elementwise, yielding 0 wherever the denominator is zero."""
    return np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator, dtype=float),
        where=denominator != 0,
    )


def _hard_rates(labels: np.ndarray, preds: np.ndarray) -> dict[str, float]:
    """
    Macro one-vs-rest rates and agreement scores from hard predictions.

    A single-class target has no false positives or negatives to speak of, so
    every rate is 0. Calling scikit-learn there would only emit warnings about
    the degenerate confusion matrix.
    """
    classes = np.union1d(labels, preds)
    if classes.size == 1:
        return {"tpr": 0.0, "fpr": 0.0, "mcc": 0.0, "kappa": 0.0}

    matrix = confusion_matrix(labels, preds, labels=classes).astype(float)
    true_positives = np.diag(matrix)
    false_negatives = matrix.sum(axis=1) - true_positives
    false_positives = matrix.sum(axis=0) - true_positives
    true_negatives = matrix.sum() - true_positives - false_positives - false_negatives

    return {
        "tpr": float(
            _safe_rate(true_positives, true_positives + false_negatives).mean()
        ),
        "fpr": float(
            _safe_rate(false_positives, false_positives + true_negatives).mean()
        ),
        "mcc": float(matthews_corrcoef(labels, preds)),
        "kappa": float(cohen_kappa_score(labels, preds)),
    }


def _macro_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    """
    Macro one-vs-rest AUC over the class probabilities.

    Returns 0 when only one class is present, since AUC is undefined for a
    single-class target. Class ids index the score columns.
    """
    y = np.asarray(labels)
    classes = np.unique(y)
    if classes.size == 1:
        return 0.0

    columns = scores[:, classes.astype(int)]
    per_class = [
        roc_auc_score((y == label).astype(int), columns[:, index])
        for index, label in enumerate(classes)
    ]

    return float(np.mean(per_class))


def compute_classification_metrics(
    labels: np.ndarray | list,
    preds: np.ndarray | list,
    scores: np.ndarray | None = None,
) -> dict[str, float]:
    """Compute hard-label metrics plus AUC when class scores are given."""
    logger.debug("computing classification metrics")

    metrics = {
        "accuracy": accuracy_score(labels, preds),
        "precision": precision_score(labels, preds, average="macro", zero_division=0),
        "recall": recall_score(labels, preds, average="macro", zero_division=0),
        "f1": f1_score(labels, preds, average="macro", zero_division=0),
        **_hard_rates(np.asarray(labels), np.asarray(preds)),
    }

    if scores is not None:
        metrics["auc"] = _macro_auc(np.asarray(labels), scores)

    return metrics

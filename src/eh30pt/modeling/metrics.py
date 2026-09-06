"""Streaming metric computation for encoder, decoder, and encoder-decoder Trainers."""

from collections import Counter
from typing import TYPE_CHECKING

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch import Tensor
from transformers import EvalPrediction  # noqa: TC002

from eh30pt.constants import SENTINEL_TOKEN
from eh30pt.logging import format_counts, logger

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerFast

    from eh30pt.types import Architecture


class MetricsAccumulator:
    """
    Collect per-batch eval outputs and computes metrics on the final call.

    Parameters
    ----------
    tokenizer:
        Tokenizer used to decode generated ids (unused for encoders).
    arch:
        Model architecture, selecting the accumulation format.

    """

    def __init__(self, tokenizer: PreTrainedTokenizerFast, arch: Architecture) -> None:
        """Store the tokenizer and architecture."""
        self._tokenizer = tokenizer
        self._arch = arch
        self._labels: list[np.ndarray] = []
        self._preds: list[np.ndarray] = []

    def __call__(
        self,
        eval_pred: EvalPrediction,
        *,
        compute_result: bool = True,
    ) -> dict[str, float]:
        """Accumulate one batch, computing metrics when `compute_result`."""
        self._extend(eval_pred)
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

        labels, preds = _batch_to_numpy(eval_pred)
        if self._arch == "decoder":
            _, label_dim = labels.shape
            _, pred_dim = preds.shape
            if label_dim < pred_dim:
                preds = preds[:, pred_dim - label_dim :]

        self._labels.extend(labels)
        self._preds.extend(preds)

    def _compute(self) -> dict[str, float]:
        if self._arch == "encoder":
            all_labels = np.array(self._labels)
            all_preds = np.array(self._preds)
            mask = all_labels != SENTINEL_TOKEN

            _labels_count = Counter(all_labels[mask].tolist())
            _preds_count = Counter(all_preds[mask].tolist())

            logger.debug("labels: %s", format_counts(_labels_count))
            logger.debug("predictions: %s", format_counts(_preds_count))

            return _compute_classification_metrics(all_labels[mask], all_preds[mask])

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
        predictions = _filter_gibberish(references, predictions)

        _refs_count = Counter(references)
        _preds_count = Counter(predictions)

        logger.debug("references: %s", format_counts(_refs_count))
        logger.debug("predictions: %s", format_counts(_preds_count))

        return _compute_classification_metrics(references, predictions)


def _batch_to_numpy(eval_pred: EvalPrediction) -> tuple[np.ndarray, np.ndarray]:
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
    return batch_labels, batch_preds


def _filter_gibberish(references: list[str], predictions: list[str]) -> list[str]:
    refset = set(references)
    return [pred if pred in refset else "<gibberish>" for pred in predictions]


def _compute_classification_metrics(
    labels: np.ndarray | list,
    preds: np.ndarray | list,
) -> dict[str, float]:
    logger.debug("computing classification metrics")

    accuracy = accuracy_score(labels, preds)
    precision = precision_score(labels, preds, average="macro", zero_division=0)
    recall = recall_score(labels, preds, average="macro", zero_division=0)
    f1 = f1_score(labels, preds, average="macro", zero_division=0)
    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}

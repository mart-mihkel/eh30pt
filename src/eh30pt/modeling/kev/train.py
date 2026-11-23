"""Evaluation and calibration helpers for Kev models."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from torch.nn import functional

from eh30pt.modeling.kev.trainer import _collate as _trainer_collate
from eh30pt.modeling.metrics import compute_classification_metrics

if TYPE_CHECKING:
    from collections.abc import Sequence

    from eh30pt.modeling.kev.model import KevModel
    from eh30pt.modeling.kev.prompt import EncodedSample


@dataclass(frozen=True)
class DecisionBatch:
    """Padded tensors of one forward pass over Kev rows."""

    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    position_ids: torch.Tensor
    decide_index: torch.Tensor
    option_index: torch.Tensor
    option_mask: torch.Tensor
    labels: torch.Tensor


def _collate(
    samples: Sequence[EncodedSample],
    pad_id: int,
    device: torch.device,
) -> DecisionBatch:
    """Create a padded batch for direct model evaluation."""
    values = _trainer_collate(
        [
            {
                "input_ids": sample.encoded.input_ids,
                "decide_index": sample.encoded.decide_index,
                "option_index": sample.encoded.option_indices,
                "labels": sample.label,
            }
            for sample in samples
        ],
        pad_id,
    )

    return DecisionBatch(**{name: value.to(device) for name, value in values.items()})


@dataclass(frozen=True)
class EvaluationMetrics:
    """Decision quality on one set of rows."""

    loss: float
    accuracy: float
    f1: float
    count: int


def kev_logits(
    model: KevModel,
    samples: Sequence[EncodedSample],
    *,
    batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return option logits and labels for encoded Kev rows."""
    device = next(model.parameters()).device
    logits: list[torch.Tensor] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(samples), batch_size):
            batch = _collate(samples[start : start + batch_size], model.pad_id, device)
            scores = model(
                batch.input_ids,
                batch.attention_mask,
                batch.position_ids,
                batch.decide_index,
                batch.option_index,
            )
            logits.append(
                scores.masked_fill(~batch.option_mask, torch.finfo(scores.dtype).min)
            )

    return torch.cat(logits), torch.tensor(
        [sample.label for sample in samples], device=device
    )


def _metrics(logits: torch.Tensor, labels: torch.Tensor) -> EvaluationMetrics:
    metrics = compute_classification_metrics(
        labels.cpu().numpy(),
        logits.argmax(dim=-1).cpu().numpy(),
        classes=list(range(logits.shape[-1])),
    )
    return EvaluationMetrics(
        loss=float(functional.cross_entropy(logits, labels)),
        accuracy=metrics["accuracy"],
        f1=metrics["f1"],
        count=len(labels),
    )


def evaluate_kev(
    model: KevModel,
    samples: Sequence[EncodedSample],
    *,
    batch_size: int,
    temperature: float = 1.0,
) -> EvaluationMetrics:
    """Score encoded rows at the selected calibration temperature."""
    logits, labels = kev_logits(model, samples, batch_size=batch_size)
    return _metrics(logits / temperature, labels)


def fit_temperature(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """Find the scalar temperature with minimum held-out cross-entropy."""
    if len(labels) == 0:
        return 1.0

    candidates = torch.exp(torch.linspace(math.log(0.05), math.log(20.0), steps=601))
    losses = torch.stack(
        [functional.cross_entropy(logits / value, labels) for value in candidates]
    )
    return float(candidates[int(losses.argmin())])


def calibrate_kev(
    model: KevModel,
    samples: Sequence[EncodedSample],
    *,
    batch_size: int,
) -> float:
    """Fit a decision temperature using the held-out evaluation rows."""
    logits, labels = kev_logits(model, samples, batch_size=batch_size)
    return fit_temperature(logits, labels)

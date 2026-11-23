"""Unit tests for the Kev-style decision model primitives."""

from __future__ import annotations

from typing import Final

import pytest
import torch

from eh30pt.modeling.kev.model import PointerHead, resolve_device
from eh30pt.modeling.kev.prompt import EncodedKevRow, EncodedSample
from eh30pt.modeling.kev.records import KevRecord
from eh30pt.modeling.kev.train import (
    DecisionBatch,
    _collate,
    _metrics,
    fit_temperature,
)

_hidden_size: Final = 8
_head_dim: Final = 4


def test_decision_record_rejects_missing_options() -> None:
    with pytest.raises(ValueError, match="option"):
        KevRecord(state="state", instructions="question", options=(), label=0)


def test_decision_record_rejects_label_outside_options() -> None:
    with pytest.raises(ValueError, match="outside"):
        KevRecord(
            state="state",
            instructions="question",
            options=("a", "b"),
            label=2,
        )


def test_pointer_head_scores_every_option() -> None:
    head = PointerHead(hidden_size=_hidden_size, head_dim=_head_dim)
    decide = torch.randn(2, _hidden_size)
    options = torch.randn(2, 5, _hidden_size)

    logits = head(decide, options)

    assert logits.shape == (2, 5)
    assert logits.dtype == torch.float32


def test_fit_temperature_recovers_the_sampling_sharpness() -> None:
    generator = torch.Generator().manual_seed(0)
    logits = torch.randn(512, 3, generator=generator)
    probabilities = torch.softmax(logits / 2.0, dim=-1)
    labels = torch.multinomial(probabilities, 1, generator=generator).squeeze(-1)

    temperature = fit_temperature(logits, labels)

    assert temperature == pytest.approx(2.0, rel=0.15)


def test_fit_temperature_ignores_empty_labels() -> None:
    assert fit_temperature(torch.empty(0, 3), torch.empty(0, dtype=torch.long)) == 1.0


def test_macro_f1_averages_over_the_full_option_set() -> None:
    labels = torch.tensor([0, 1, 2])
    assert _metrics(torch.eye(3), labels).f1 == pytest.approx(1.0)
    logits = torch.tensor([[1.0, 0, 0]] * 3)
    assert _metrics(logits, labels).f1 == pytest.approx(1 / 6)
    absent = _metrics(torch.tensor([[1.0, 0, 0]]), torch.tensor([0]))
    assert absent.f1 == pytest.approx(1 / 3)


def test_collate_pads_rows_and_marks_options() -> None:
    samples = (
        _sample(input_ids=(1, 2, 3), decide_index=2, option_indices=(2,)),
        _sample(
            input_ids=(4, 5, 6, 7, 8),
            decide_index=4,
            option_indices=(3, 4),
        ),
    )

    batch = _collate(samples, pad_id=0, device=torch.device("cpu"))

    assert isinstance(batch, DecisionBatch)
    assert batch.input_ids.tolist() == [
        [1, 2, 3, 0, 0],
        [4, 5, 6, 7, 8],
    ]
    assert batch.attention_mask.tolist() == [
        [1, 1, 1, 0, 0],
        [1, 1, 1, 1, 1],
    ]
    assert batch.position_ids.tolist() == [
        [0, 1, 2, 0, 0],
        [0, 1, 2, 3, 4],
    ]
    assert batch.decide_index.tolist() == [2, 4]
    assert batch.option_index.tolist() == [[2, 0], [3, 4]]
    assert batch.option_mask.tolist() == [[True, False], [True, True]]
    assert batch.labels.tolist() == [0, 1]


def test_resolve_device_accepts_cpu() -> None:
    assert resolve_device("cpu") == torch.device("cpu")


def _sample(
    *,
    input_ids: tuple[int, ...],
    decide_index: int,
    option_indices: tuple[int, ...],
) -> EncodedSample:
    """Build a labelled encoded row without a tokenizer."""
    encoded = EncodedKevRow(
        input_ids=input_ids,
        position_ids=tuple(range(len(input_ids))),
        decide_index=decide_index,
        option_indices=option_indices,
        state_tokens=1,
        truncated=False,
    )
    return EncodedSample(encoded=encoded, label=len(option_indices) - 1)

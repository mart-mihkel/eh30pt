"""Tests for the binary jah/ei discontinuation dataset."""

from pathlib import Path
from typing import TYPE_CHECKING, cast

import pytest

from eh30pt.datasets.discont.binary import DiscontBinary
from eh30pt.datasets.discont.classify import DiscontClassifier
from eh30pt.datasets.registry import (
    get_few_shots,
    get_response_schema,
    get_sys_prompt,
    get_task,
    load_data,
)

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerFast

    from eh30pt.datasets.discont.binary import DiscontBinaryRow

_dataset = "discont-binary/extract-labels"


def test_total_counts_the_split_rows() -> None:
    rows: list[DiscontBinaryRow] = [
        {"id": 1, "text": "t", "phrase": "p", "drug": "d", "label": "jah"},
        {"id": 2, "text": "t", "phrase": "p", "drug": "d", "label": "ei"},
    ]
    tokenizer = cast("PreTrainedTokenizerFast", None)

    assert DiscontBinary(rows, tokenizer, "decoder").total() == len(rows)


def test_registry_exposes_it_as_a_classifier() -> None:
    assert get_task(_dataset) == "classification"
    assert get_sys_prompt(_dataset) == DiscontBinary.system_prompt
    assert get_response_schema(_dataset)["json_schema"]["schema"] == {
        "type": "string",
        "enum": ["jah", "ei"],
    }

    assert get_few_shots(_dataset, 3) == DiscontBinary.examples[:3]


def test_every_studio_reason_folds_onto_a_label() -> None:
    """The labels partition cleanly into the two answers."""
    stopping = (
        "Adverse reactions",
        "Treatment success",
        "Treatment inefficacy",
        "Contraindication/unsafe co-usage",
        "Other",
    )

    continued = (
        "Adverse reaction without discontinuation",
        "Treatment inefficacy without discontinuation",
        "Dosage change",
        "Other non-discontinuation",
    )

    assert set(DiscontBinary.reason2label) == set(DiscontClassifier.en2et)
    assert set(DiscontBinary.reason2label.values()) == {"jah", "ei"}

    for reason in stopping:
        assert DiscontBinary.reason2label[reason] == "jah"

    for reason in continued:
        assert DiscontBinary.reason2label[reason] == "ei"


def test_few_shots_agree_with_the_folding() -> None:
    """Every few-shot example must match what its reason folds onto."""
    cls_shots = DiscontClassifier.examples
    bin_shots = DiscontBinary.examples
    margend2reason: dict[str, str] = {
        margend: reason for reason, margend in DiscontClassifier.en2et.items()
    }

    assert len(cls_shots) == len(bin_shots)
    for cls_shot, bin_shot in zip(cls_shots, bin_shots, strict=True):
        assert (
            bin_shot.label == DiscontBinary.reason2label[margend2reason[cls_shot.label]]
        )


def test_load_data_requires_a_data_path_for_the_binary_dataset() -> None:
    tokenizer = cast("PreTrainedTokenizerFast", None)

    with pytest.raises(ValueError, match="needs a labels file"):
        load_data(tokenizer, _dataset, "decoder")


def test_load_data_rejects_a_data_path_on_a_dataset_without_labels() -> None:
    tokenizer = cast("PreTrainedTokenizerFast", None)

    with pytest.raises(ValueError, match="labels file is only read by"):
        load_data(tokenizer, "oblique", "decoder", data_path=Path("labels.parquet"))

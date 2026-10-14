"""Tests for the binary yes/no discontinuation dataset."""

from pathlib import Path
from typing import TYPE_CHECKING, cast

import pytest

from eh30pt.datasets.discontinuation.binary import DiscontBinary
from eh30pt.datasets.discontinuation.classify import DiscontClassifier
from eh30pt.datasets.registry import (
    get_few_shots,
    get_labels,
    get_response_schema,
    get_sys_prompt,
    get_task,
    load_data,
)

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerFast

_dataset = "discont-binary/extract-labels"


def test_labels_are_yes_and_no() -> None:
    assert get_labels(_dataset) == ("yes", "no")
    assert DiscontBinary.id2label == {0: "yes", 1: "no"}
    assert DiscontBinary.label2id == {"yes": 0, "no": 1}


def test_registry_exposes_it_as_a_classifier() -> None:
    assert get_task(_dataset) == "classification"
    assert get_sys_prompt(_dataset) == DiscontBinary.system_prompt
    assert get_response_schema(_dataset)["json_schema"]["schema"] == {
        "type": "string",
        "enum": ["yes", "no"],
    }
    assert get_few_shots(_dataset, 3) == DiscontBinary.few_shots[:3]


def test_prompt_only_asks_yes_or_no() -> None:
    assert "yes" in DiscontBinary.system_prompt
    assert "no" in DiscontBinary.system_prompt
    assert 'Respond only with "yes" or "no".' in DiscontBinary.system_prompt


def test_every_studio_reason_folds_onto_a_label() -> None:
    """The ten labels partition cleanly into the two answers."""
    stopping = (
        "Adverse reactions",
        "Treatment success",
        "Treatment inefficacy",
        "Contraindication/unsafe co-usage",
        "Other",
    )
    continued = (
        "Adverse reaction without discontinuation",
        "Dosage change",
        "Other non-discontinuation",
    )

    assert set(DiscontBinary.reason2label) == set(DiscontClassifier.id2label.values())
    assert set(DiscontBinary.reason2label.values()) == {"yes", "no"}

    for reason in stopping:
        assert DiscontBinary.reason2label[reason] == "yes"

    for reason in continued:
        assert DiscontBinary.reason2label[reason] == "no"


def test_few_shots_agree_with_the_folding() -> None:
    """Every few-shot example must match what its reason folds onto."""
    by_user = {shot.user: shot.label for shot in DiscontClassifier.few_shots}
    binary = {shot.user: shot.label for shot in DiscontBinary.few_shots}

    assert binary.keys() == by_user.keys()
    for user, label in binary.items():
        assert label == DiscontBinary.reason2label[by_user[user]]


def test_load_data_requires_a_data_path_for_the_binary_dataset() -> None:
    tokenizer = cast("PreTrainedTokenizerFast", None)

    with pytest.raises(ValueError, match="needs a labels file"):
        load_data(tokenizer, _dataset, "decoder")


def test_load_data_rejects_a_data_path_on_a_dataset_without_labels() -> None:
    tokenizer = cast("PreTrainedTokenizerFast", None)

    with pytest.raises(ValueError, match="labels file is only read by"):
        load_data(tokenizer, "oblique", "decoder", data_path=Path("labels.parquet"))

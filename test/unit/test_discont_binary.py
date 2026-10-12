"""Tests for the binary yes/no discontinuation dataset."""

from typing import TYPE_CHECKING, cast

import pytest

from eh30pt.datasets.discontinuation.binary import (
    DiscontBinary,
    DiscontBinaryLabel,
    _map_label,
)
from eh30pt.datasets.discontinuation.classify import DiscontClassifier
from eh30pt.datasets.registry import (
    get_few_shots,
    get_labels,
    get_response_schema,
    get_sys_prompt,
    get_task,
    load_data,
)
from eh30pt.labeling.votes import labeled_row, labeled_rows

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
    """The eight reasons partition cleanly into the two answers."""
    stopping = (
        "Adverse reactions",
        "Treatment success",
        "Treatment inefficacy",
        "Contraindication",
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


@pytest.mark.parametrize("control", ["discont", "other"])
def test_labeled_row_folds_a_reason_the_same_from_either_box(control: str) -> None:
    """Only the reason decides the answer, never which box it was chosen in."""
    discont = ("Adverse reactions",) if control == "discont" else ()
    other = ("Dosage change",) if control == "other" else ()
    row = labeled_row(
        task_id=1,
        text="text",
        phrase="phrase",
        drug="drug",
        discont=discont,
        other=other,
        meta=(),
        fold=_map_label,
    )
    expected: DiscontBinaryLabel = "yes" if control == "discont" else "no"

    assert row == {
        "id": 1,
        "text": "text",
        "phrase": "phrase",
        "drug": "drug",
        "label": expected,
    }


def test_labeled_row_keeps_distinct_reasons_apart() -> None:
    """Two reasons that disagree on yes/no must not collapse into one vote."""

    def ballot(annotation_id: int, reason: str) -> dict[str, object]:
        return {
            "id": annotation_id,
            "result": [
                {
                    "from_name": "discont",
                    "to_name": "text",
                    "type": "choices",
                    "value": {"choices": [reason]},
                }
            ],
        }

    tasks = [
        {
            "id": 1,
            "inner_id": 1,
            "data": {"text": "t", "phrase": "p", "drug": "d"},
            "annotations": [
                ballot(1, "Adverse reactions"),
                ballot(2, "Dosage change"),
            ],
        },
    ]

    # Both fold to different labels, so the task ties and trains nothing.
    rows = labeled_rows(tasks, _map_label)  # ty: ignore[invalid-argument-type]

    assert list(rows) == []


def test_map_label_drops_a_reason_the_folding_does_not_know() -> None:
    assert _map_label("Something new") is None


def test_load_data_requires_projects_for_the_binary_dataset() -> None:
    tokenizer = cast("PreTrainedTokenizerFast", None)

    with pytest.raises(ValueError, match="needs project ids"):
        load_data(tokenizer, _dataset, "decoder")


def test_load_data_rejects_projects_on_a_dataset_without_them() -> None:
    tokenizer = cast("PreTrainedTokenizerFast", None)

    with pytest.raises(ValueError, match="project ids are only read by"):
        load_data(tokenizer, "oblique", "decoder", labeling_projects=[1])

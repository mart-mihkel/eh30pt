"""Tests for silver prediction helpers."""

import json
from typing import TYPE_CHECKING

import pytest

from eh30pt.datasets.registry import get_labels
from eh30pt.labeling.silver.classify import (
    classification_choice,
    classification_controls,
    classification_sample,
)
from eh30pt.labeling.silver.extract import extraction_fields, extraction_spans

if TYPE_CHECKING:
    from eh30pt.labeling.studio import StudioTask


def _extracted_task(task_id: int) -> StudioTask:
    return {
        "id": task_id,
        "data": {
            "text": "t",
            "phrase": "p",
            "reason": "r",
            "drug": "d",
        },
        "annotations": [],
    }


def test_extraction_fields_parse_mapping_and_json() -> None:
    fields = {"phrase": "p", "reason": "r", "drug": "d"}

    assert extraction_fields(fields) == ("p", "r", "d")
    assert extraction_fields(json.dumps(fields)) == ("p", "r", "d")


@pytest.mark.parametrize(
    "prediction",
    ["not json", {"phrase": "p", "reason": 1, "drug": "d"}, ["p", "r", "d"]],
)
def test_extraction_fields_reject_invalid_predictions(prediction: object) -> None:
    assert extraction_fields(prediction) is None


def test_extraction_spans_locate_each_field() -> None:
    text = "katkestas Nebileti tarvitamise, vererõhk normaliseerus"
    prediction = {
        "phrase": text,
        "reason": "vererõhk normaliseerus",
        "drug": "Nebilet",
    }

    spans = extraction_spans(text, prediction)

    assert spans is not None
    assert [span["value"]["labels"] for span in spans] == [
        ["phrase"],
        ["reason"],
        ["drug"],
    ]
    assert all(span["readonly"] for span in spans)


def test_extraction_spans_reject_text_without_a_field() -> None:
    text = "katkestas Nebileti tarvitamise, vererõhk normaliseerus"
    prediction = {"phrase": text, "reason": "peavalu", "drug": "Nebilet"}

    assert extraction_spans(text, prediction) is None


def test_classification_sample_prompts_with_the_extracted_fields() -> None:
    task_id = 3
    sample = classification_sample(_extracted_task(task_id))

    assert sample.id == task_id
    assert sample.label == ""
    assert sample.user == "Anamnees: t\nFraas: p\nRavim: d"


def test_classification_controls_map_each_label_to_its_box() -> None:
    controls = classification_controls("discont-classify/extract-labels")

    assert set(controls) == set(get_labels("discont-classify/extract-labels"))
    assert controls["Kõrvaltoimed"] == ("discont", "Adverse reactions")
    assert controls["Muu põhjus"] == ("discont", "Other")
    assert controls["Annuse muutmine"] == ("other", "Dosage change")
    assert controls["Muu sündmus ilma ravi lõpetamiseta"] == (
        "other",
        "Other non-discontinuation",
    )


def test_classification_controls_reject_datasets_outside_the_interface() -> None:
    with pytest.raises(NotImplementedError, match="classification dataset"):
        classification_controls("multinerd")


def test_classification_choice_selects_the_box_of_the_label() -> None:
    controls = classification_controls("discont-classify/extract-labels")

    result = classification_choice(" Ravi edukas\n", controls)

    assert result is not None
    assert result["from_name"] == "discont"
    assert result["type"] == "choices"
    assert result["readonly"]
    assert result["value"] == {"choices": ["Treatment success"]}


@pytest.mark.parametrize(
    "prediction",
    ["", "Mitte ükski silt", None, {"choices": ["Kõrvaltoimed"]}],
)
def test_classification_choice_rejects_outputs_that_are_not_labels(
    prediction: object,
) -> None:
    controls = classification_controls("discont-classify/extract-labels")

    assert classification_choice(prediction, controls) is None

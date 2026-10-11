"""Tests for classification Label Studio task building."""

import polars as pl

from eh30pt.datasets.registry import get_labels
from eh30pt.labeling.classify import (
    choice_config,
    choice_task,
    classification_labeling,
)


def test_choice_config_offers_dataset_labels() -> None:
    config = choice_config(get_labels("oblique"))

    assert '<Text name="text" value="$text" />' in config
    assert (
        '<Choices name="label" toName="text" choice="single" showInLine="true">'
        in config
    )
    for label in get_labels("oblique"):
        assert f'<Choice value="{label}" />' in config


def test_choice_config_is_not_a_span_config() -> None:
    config = choice_config(get_labels("discont-classifier/extract-labels"))

    assert 'value="phrase" background="#ffe08a"' not in config
    assert '<View style="display: none">' not in config


def test_choice_task_preselects_label() -> None:
    task = choice_task("sõna on vaba", "vaba", "run-1")

    assert task == {
        "data": {"text": "sõna on vaba"},
        "predictions": [
            {
                "result": [
                    {
                        "from_name": "label",
                        "to_name": "text",
                        "type": "choices",
                        "readonly": True,
                        "hidden": False,
                        "value": {"choices": ["vaba"]},
                    }
                ],
                "model_version": "run-1",
            }
        ],
    }


def test_classification_labeling_skips_foreign_predictions() -> None:
    frame = pl.DataFrame(
        {
            "user": ["sõna", "teine", "kolmas"],
            "prediction": ["vaba", "ebaloomulik", "hallucinated"],
        }
    )

    labeling = classification_labeling("oblique", frame, "run-1")

    assert labeling.tasks == [
        choice_task("sõna", "vaba", "run-1"),
        choice_task("teine", "ebaloomulik", "run-1"),
    ]
    assert labeling.config == choice_config(get_labels("oblique"))


def test_classification_labeling_without_shards_is_empty() -> None:
    labeling = classification_labeling(
        "discont-classifier/extract-labels", pl.DataFrame(), "run-1"
    )

    assert labeling.tasks == []
    assert 'value="Other"' in labeling.config

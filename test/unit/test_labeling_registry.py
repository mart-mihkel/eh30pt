"""Tests for the labeling dispatch on dataset task type."""

from typing import TYPE_CHECKING, cast

import polars as pl
import pytest

from eh30pt.datasets.registry import get_labels
from eh30pt.labeling.registry import build_labeling, export_frame, label_controls
from eh30pt.labeling.tasks.classify import choice_config
from eh30pt.labeling.tasks.extract import extraction_config, extraction_controls

if TYPE_CHECKING:
    from eh30pt.types import DatasetName


def test_build_labeling_dispatches_on_task_type() -> None:
    frame = pl.DataFrame()

    extraction = build_labeling("discont-extract/trigger", frame, "run-1")
    classification = build_labeling("discont-classifier/extract-labels", frame, "run-1")

    assert extraction.tasks == []
    assert extraction.config == extraction_config()
    assert classification.tasks == []
    assert classification.config == choice_config(
        get_labels("discont-classifier/extract-labels")
    )


def test_label_controls_dispatches_on_task_type() -> None:
    assert label_controls("discont-extract/trigger") == extraction_controls()
    assert label_controls("discont-classifier/extract-labels") == ("label",)


def test_export_frame_dispatches_on_task_type() -> None:
    span_tasks = [
        {
            "id": 7,
            "inner_id": 1,
            "data": {"text": "t", "phrase": "p", "drug": "d", "reason": "r"},
            "annotations": [
                {
                    "id": 1,
                    "completed_by": 2,
                    "result": [
                        {
                            "id": "res-1",
                            "from_name": "label",
                            "to_name": "text",
                            "type": "choices",
                            "value": {"choices": ["Other"]},
                        },
                        {
                            "id": "res-2",
                            "from_name": "meta",
                            "to_name": "text",
                            "type": "choices",
                            "value": {"choices": ["Incomplete phrase"]},
                        },
                    ],
                }
            ],
        },
    ]
    span_frame = export_frame(span_tasks, "discont-extract/trigger", 5)  # ty: ignore[invalid-argument-type]

    assert span_frame["label"].to_list() == ["Other"]
    assert "discont" not in span_frame.columns

    choice_tasks = [
        {
            "id": 9,
            "inner_id": 2,
            "data": {"text": "t"},
            "annotations": [{"id": 3, "result": []}],
        },
    ]

    with pytest.raises(NotImplementedError, match="no implementation"):
        export_frame(choice_tasks, "oblique", 5)  # ty: ignore[invalid-argument-type]


def test_build_labeling_rejects_unknown_dataset() -> None:
    with pytest.raises(ValueError, match="invalid dataset"):
        build_labeling(cast("DatasetName", "unknown"), pl.DataFrame(), "run-1")

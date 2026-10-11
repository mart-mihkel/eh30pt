"""Tests for the labeling dispatch on dataset task type."""

from typing import TYPE_CHECKING, cast

import polars as pl
import pytest

from eh30pt.datasets.registry import get_labels
from eh30pt.labeling.classify import choice_config
from eh30pt.labeling.extract import span_config
from eh30pt.labeling.registry import build_labeling

if TYPE_CHECKING:
    from eh30pt.types import DatasetName


def test_build_labeling_dispatches_on_task_type() -> None:
    frame = pl.DataFrame()

    extraction = build_labeling("discont-extract/trigger", frame, "run-1")
    classification = build_labeling("discont-classifier/extract-labels", frame, "run-1")

    assert extraction.tasks == []
    assert extraction.config == span_config()
    assert classification.tasks == []
    assert classification.config == choice_config(
        get_labels("discont-classifier/extract-labels")
    )


def test_build_labeling_rejects_unknown_dataset() -> None:
    with pytest.raises(ValueError, match="invalid dataset"):
        build_labeling(cast("DatasetName", "unknown"), pl.DataFrame(), "run-1")

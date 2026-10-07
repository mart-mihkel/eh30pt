"""Tests for the create-task script helpers."""

import pytest

from eh30pt.cli.scripts.labeling import _dataset_name


def test_dataset_name_returns_the_tracked_dataset() -> None:
    assert _dataset_name({"dataset": "discont-extract", "method": "x"}, "run-1") == (
        "discont-extract"
    )


def test_dataset_name_rejects_a_run_without_dataset() -> None:
    with pytest.raises(ValueError, match="records no dataset"):
        _dataset_name({"method": "few-shot"}, "run-1")


def test_dataset_name_rejects_an_unknown_dataset() -> None:
    with pytest.raises(ValueError, match="records unknown dataset 'nope'"):
        _dataset_name({"dataset": "nope"}, "run-1")

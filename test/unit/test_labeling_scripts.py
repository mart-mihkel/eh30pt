"""Tests for the labeling script helpers."""

from typing import TYPE_CHECKING

import pytest

from eh30pt.cli.scripts.create_task import _dataset_name
from eh30pt.datasets.registry import get_response_schema
from eh30pt.settings import ServerSettings

if TYPE_CHECKING:
    from eh30pt.types import DatasetName


def test_dataset_name_returns_the_tracked_dataset() -> None:
    assert _dataset_name(
        {"dataset": "discont-extract/trigger", "method": "x"}, "run-1"
    ) == ("discont-extract/trigger")


def test_dataset_name_rejects_a_run_without_dataset() -> None:
    with pytest.raises(ValueError, match="records no dataset"):
        _dataset_name({"method": "few-shot"}, "run-1")


def test_dataset_name_rejects_an_unknown_dataset() -> None:
    with pytest.raises(ValueError, match="records unknown dataset 'nope'"):
        _dataset_name({"dataset": "nope"}, "run-1")


@pytest.mark.parametrize(
    "dataset",
    ["discont-extract/trigger", "discont-classify/extract-labels"],
)
def test_structured_server_uses_the_dataset_response_schema(
    dataset: DatasetName,
) -> None:
    fmt = get_response_schema(dataset)
    server = ServerSettings(format=fmt)

    assert server.format == get_response_schema(dataset)

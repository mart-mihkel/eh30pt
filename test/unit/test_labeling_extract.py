"""Tests for extraction Label Studio task building."""

from typing import cast, get_args

import polars as pl
import pytest

from eh30pt.datasets.registry import get_task
from eh30pt.labeling.extract import (
    _assert_span_fields,
    extraction_labeling,
    span_config,
)
from eh30pt.types import DatasetName


def test_every_extraction_dataset_is_exportable() -> None:
    for name in cast("tuple[DatasetName, ...]", get_args(DatasetName)):
        if get_task(name) != "extraction":
            continue

        try:
            _assert_span_fields(name)
        except NotImplementedError as exc:
            pytest.fail(f"extraction dataset '{name}' is not exportable: {exc}")


def test_span_config_highlights_phrase_and_drug() -> None:
    config = span_config()

    assert 'value="phrase" background="#ffe08a"' in config
    assert 'value="drug" background="#a8e6a3"' in config
    assert config.index("<Choices ") < config.index('<View style="display: none">')


def test_extraction_labeling_without_shards_is_empty() -> None:
    labeling = extraction_labeling("discont-extract", pl.DataFrame(), "run-1")

    assert labeling.tasks == []
    assert labeling.config == span_config()

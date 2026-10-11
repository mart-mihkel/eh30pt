"""Tests for extraction Label Studio task building."""

import re
from typing import cast, get_args

import polars as pl
import pytest

from eh30pt.datasets.registry import get_task
from eh30pt.labeling.extract import (
    _assert_span_fields,
    extraction_labeling,
    span_config,
    span_controls,
)
from eh30pt.types import DatasetName

_row_style = "display: flex; flex-wrap: wrap; gap: 16px; align-items: flex-start"
_col_style = "flex: 1; min-width: 240px"
_box_cls = "eh30pt-box"
_title_cls = "eh30pt-title"


def test_every_extraction_dataset_is_exportable() -> None:
    for name in cast("tuple[DatasetName, ...]", get_args(DatasetName)):
        if get_task(name) != "extraction":
            continue

        try:
            _assert_span_fields(name)
        except NotImplementedError as exc:
            pytest.fail(f"extraction dataset '{name}' is not exportable: {exc}")


def test_span_config_highlights_phrase_reason_and_drug() -> None:
    config = span_config()

    assert 'value="phrase" background="#ffe08a"' in config
    assert 'value="reason" background="#a9c6ff"' in config
    assert 'value="drug" background="#a8e6a3"' in config
    assert config.index("<Choices ") < config.index('<View style="display: none">')


def test_span_config_has_three_choice_boxes() -> None:
    config = span_config()

    assert '<Header value="Drug discontinuation" />' in config
    assert '<Header value="Not discontinuation" />' in config
    assert '<Header value="Meta" />' in config

    common = 'toName="text" choice="multiple" showInLine="false"'
    for name in ("discont", "other", "meta"):
        assert f'<Choices name="{name}" {common}>' in config

    for label in (
        "Treatment success",
        "Treatment inefficacy",
        "Contraindication",
        "Adverse reactions",
        "Adverse reaction without discontinuation",
        "Dosage change",
        "Incomplete phrase",
        "Incorrect phrase",
    ):
        assert f'value="{label}" hint=' in config

    assert 'value="Drug discontinuation (DD)"' not in config
    assert 'value="Dosage change (DC)"' not in config
    assert 'value="Complex fact"' not in config
    assert 'value="Adverse reactions (ADR)"' not in config


def test_span_config_boxes_lie_side_by_side() -> None:
    """Boxes only render in a row when they share one flex parent."""
    config = span_config()

    _boxes = 3

    row = f'\n  <View style="{_row_style}">'
    box = f'\n    <View className="{_box_cls}" style="{_col_style}">'

    assert config.count(row) == 1
    assert config.count(box) == config.count("<Choices ") == _boxes
    assert config.index(row) < config.index(box)
    assert config.rindex(box) < config.index('<View style="display: none">')


def test_span_config_highlights_the_header_of_a_used_box() -> None:
    """Only a colour changes when a choice is picked, so the layout cannot move."""
    config = span_config()

    _boxes = 3

    rule = f".{_box_cls}:has(input:checked) .{_title_cls} {{"
    assert rule in config

    assert "::before" not in config
    assert "visibleWhen" not in config
    assert f".{_title_cls} {{" in config
    assert "border: 1px solid transparent" in config
    assert "border-color: #1f9d55" in config

    assert config.count(f'className="{_box_cls}"') == _boxes
    assert config.count(f'className="{_title_cls}"') == _boxes

    title = f'<View className="{_title_cls}">'
    for header in ("Drug discontinuation", "Not discontinuation", "Meta"):
        assert f'{title}\n        <Header value="{header}" />\n      </View>' in config


def test_span_controls_match_the_rendered_boxes() -> None:
    config = span_config()

    assert span_controls() == ("discont", "other", "meta")
    for name in span_controls():
        assert f'<Choices name="{name}" toName="text"' in config


def test_span_config_boxes_do_not_share_a_label() -> None:
    """Two boxes offering one identical string would export as one label."""
    config = span_config()

    labels = re.findall(r'<Choice value="([^"]+)"', config)

    assert len(labels) == len(set(labels))


def test_extraction_labeling_without_shards_is_empty() -> None:
    labeling = extraction_labeling("discont-extract/trigger", pl.DataFrame(), "run-1")

    assert labeling.tasks == []
    assert labeling.config == span_config()

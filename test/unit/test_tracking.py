"""Tests for tracking config verification and run naming."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import polars as pl
import pytest
from mlflow.data.code_dataset_source import CodeDatasetSource
from mlflow.data.http_dataset_source import HTTPDatasetSource
from mlflow.data.huggingface_dataset_source import HuggingFaceDatasetSource

from eh30pt.tracking.datasets import dataset_input
from eh30pt.tracking.resume import _identity_view, build_run_name
from eh30pt.tracking.shards import write_shard
from eh30pt.types import GenerateRecord

if TYPE_CHECKING:
    from pathlib import Path
    from typing import Any

    from eh30pt.types import RunConfig


@pytest.fixture
def run_config() -> RunConfig:
    """Return a run config with representative identity and volatile fields."""
    return {
        "method": "fine-tune",
        "dataset": "oblique",
        "architecture": "encoder",
        "base_model": "some-model",
        "n_shot": 0,
        "batch_size": 8,
        "source": "trigger",
        "system_prompt": "original prompt",
        "total_parameters": 100,
        "trainable_parameters": 10,
    }


def test_identity_view_drops_volatile_fields(run_config: RunConfig) -> None:
    assert "system_prompt" not in _identity_view(run_config)
    assert "total_parameters" not in _identity_view(run_config)
    assert "trainable_parameters" not in _identity_view(run_config)
    assert _identity_view(run_config)["batch_size"] == run_config["batch_size"]


def test_identity_view_ignores_volatile_drift(run_config: RunConfig) -> None:
    edited = dict(run_config)
    edited.update(
        {
            "system_prompt": "a different prompt",
            "total_parameters": 999,
            "trainable_parameters": 1,
        }
    )

    assert _identity_view(run_config) == _identity_view(cast("RunConfig", edited))


def test_build_run_name_uses_short_model_name_and_labels() -> None:
    """Shorten repo paths and append distinguishing labels."""
    assert build_run_name("org/t5gemma-2/", "qlora", "r16") == "t5gemma-2-qlora-r16"


def test_build_run_name_without_labels() -> None:
    """Join model and method when no labels are given."""
    assert build_run_name("some-model", "fine-tune") == "some-model-fine-tune"


def test_build_run_name_strips_weight_file_extension() -> None:
    """Drop a trailing weight-file extension from the model name."""
    name = build_run_name("models/Llama-3.2-1B-Instruct-Q8_0.gguf", "few-shot", "8shot")

    assert name == "Llama-3.2-1B-Instruct-Q8_0-few-shot-8shot"


def test_build_run_name_keeps_version_dots() -> None:
    """Keep dots that are part of the model version."""
    assert build_run_name("org/Llama-3.2", "fine-tune") == "Llama-3.2-fine-tune"


def test_dataset_input_names_match_datasets() -> None:
    assert dataset_input("multinerd").name == "multinerd"
    assert dataset_input("oblique").name == "oblique"
    assert dataset_input("discont-classifier").name == "discont-classifier"
    assert dataset_input("discont-decider").name == "discont-decider"


def test_dataset_input_sources_point_at_origins() -> None:
    assert isinstance(dataset_input("multinerd").source, HuggingFaceDatasetSource)
    assert isinstance(dataset_input("oblique").source, HTTPDatasetSource)
    assert isinstance(dataset_input("discont-classifier").source, CodeDatasetSource)
    assert isinstance(dataset_input("discont-decider").source, CodeDatasetSource)


def test_identity_view_keeps_identity_fields(run_config: RunConfig) -> None:
    faster = dict(run_config)
    faster["learning_rate"] = 1e-4

    other_method = dict(run_config)
    other_method["method"] = "cls-head"

    other_source = dict(run_config)
    other_source["source"] = "observed"

    assert _identity_view(run_config) != _identity_view(cast("RunConfig", faster))
    assert _identity_view(run_config) != _identity_view(cast("RunConfig", other_method))
    assert _identity_view(run_config) != _identity_view(cast("RunConfig", other_source))


def _read_records(path: Path) -> list[Any]:
    return pl.read_parquet(path).to_dicts()


def test_write_shard_numbers_files(tmp_path: Path) -> None:
    first = write_shard(
        tmp_path, "predictions", 0, [GenerateRecord(id=0, user="a", prediction="x")]
    )
    second = write_shard(
        tmp_path, "predictions", 1, [GenerateRecord(id=1, user="b", prediction="y")]
    )

    assert first.name == "predictions-00000.parquet"
    assert second.name == "predictions-00001.parquet"


def test_write_shard_round_trips_batch(tmp_path: Path) -> None:
    batch = [
        GenerateRecord(id=0, user="a", prediction="x"),
        GenerateRecord(id=1, user="b", prediction="y"),
    ]

    written = _read_records(write_shard(tmp_path, "predictions", 0, batch))

    assert [GenerateRecord(**record) for record in written] == batch

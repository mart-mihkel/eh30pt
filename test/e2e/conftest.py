"""Shared fixtures and helpers for end to end CLI tests."""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

import mlflow
import pytest

if TYPE_CHECKING:
    from pathlib import Path

    from mlflow.entities import Run
    from transformers import PreTrainedTokenizerFast

    from eh30pt.types import Architecture, DataSplits

_test_rows: Final = 64


@pytest.fixture
def tracking_uri(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> str:
    """Isolated sqlite tracking store in a fresh working directory."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MLFLOW_DISABLE_TELEMETRY", "true")
    return f"sqlite:///{tmp_path}/mlflow.db"


def single_run(tracking_uri: str, experiment: str) -> Run:
    """Return the only run of `experiment`, asserting exactly one exists."""
    mlflow.set_tracking_uri(tracking_uri)
    tracked = mlflow.get_experiment_by_name(experiment)
    assert tracked is not None

    runs = mlflow.search_runs(
        experiment_ids=[tracked.experiment_id],
        output_format="list",
    )
    assert isinstance(runs, list)
    assert len(runs) == 1

    return runs[0]


@pytest.fixture(autouse=True)
def _small_test_split(monkeypatch: pytest.MonkeyPatch) -> None:
    """Truncate registry test splits to keep end to end evaluation cheap."""
    from eh30pt.datasets import registry

    real_obl = registry.load_obl
    real_multinerd = registry.load_multinerd

    def truncate(splits: DataSplits) -> DataSplits:
        test = splits.test
        test._data = test._data.select(range(_test_rows))  # ty: ignore[unresolved-attribute, invalid-assignment]
        return splits

    def small_obl(
        tokenizer: PreTrainedTokenizerFast,
        arch: Architecture,
        n_shot: int = 0,
        n_virtual: int = 0,
    ) -> DataSplits:
        return truncate(real_obl(tokenizer, arch, n_shot, n_virtual))

    def small_multinerd(
        tokenizer: PreTrainedTokenizerFast,
        arch: Architecture,
        n_shot: int = 0,
        n_virtual: int = 0,
    ) -> DataSplits:
        return truncate(real_multinerd(tokenizer, arch, n_shot, n_virtual))

    monkeypatch.setattr(registry, "load_obl", small_obl)
    monkeypatch.setattr(registry, "load_multinerd", small_multinerd)

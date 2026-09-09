"""Shared fixtures and helpers for end to end CLI tests."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pathlib import Path

    from mlflow.entities import Run
    from mlflow.tracking import MlflowClient


@pytest.fixture
def tracking_uri(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> str:
    """Isolated sqlite tracking store in a fresh working directory."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MLFLOW_DISABLE_TELEMETRY", "true")
    return f"sqlite:///{tmp_path}/mlflow.db"


def single_run(client: MlflowClient, experiment: str) -> Run:
    """Return the only run of `experiment`, asserting exactly one exists."""
    tracked = client.get_experiment_by_name(experiment)
    assert tracked is not None

    runs = client.search_runs([tracked.experiment_id])
    assert len(runs) == 1

    return runs[0]

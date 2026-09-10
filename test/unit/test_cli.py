"""Tests for CLI environment variable fallback."""

from typing import TYPE_CHECKING

from typer.testing import CliRunner

from eh30pt.cli.cli import app

if TYPE_CHECKING:
    import pytest


def test_mlflow_options_document_env_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """MLflow flags must advertise their environment variable fallback."""
    monkeypatch.setenv("COLUMNS", "200")
    result = CliRunner().invoke(app, ["fine-tune", "--help"])

    assert result.exit_code == 0
    assert "MLFLOW_TRACKING_URI" in result.output
    assert "MLFLOW_EXPERIMENT" in result.output
    assert "MLFLOW_RUN_NAME" in result.output
    assert "MLFLOW_RUN_ID" in result.output

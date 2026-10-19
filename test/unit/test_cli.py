"""Tests for CLI environment variable fallback."""

from typing import TYPE_CHECKING

from click.testing import CliRunner

from eh30pt.cli.cli import app, log_accelerator, set_seed

if TYPE_CHECKING:
    from pathlib import Path

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


def test_set_seed_without_seed_warns(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level("WARNING", logger="eh30pt"):
        set_seed(None)

    assert "randomness is not fixed" in caplog.text


def test_set_seed_with_seed_is_deterministic() -> None:
    import torch

    set_seed(42)
    first = torch.rand(1)

    set_seed(42)
    second = torch.rand(1)

    assert torch.equal(first, second)


def test_log_accelerator_runs() -> None:
    log_accelerator()


def test_all_commands_render_help() -> None:
    """Every command must render help without warnings or errors."""
    runner = CliRunner()
    for name in app.commands:
        result = runner.invoke(app, [name, "--help"])

        assert result.exit_code == 0, name


def test_training_commands_expose_epochs_and_train_size() -> None:
    """Fine-tune, prompt-tune and qlora take --epochs and --train-size."""
    runner = CliRunner()
    for name in ("fine-tune", "prompt-tune", "qlora"):
        result = runner.invoke(app, [name, "--help"])

        assert result.exit_code == 0, name
        assert "--epochs" in result.output, name
        assert "--train-size" in result.output, name
        assert "--save-model" in result.output, name
        assert "--train-steps" not in result.output, name


def test_hf_commands_expose_system_prompt() -> None:
    """Training and evaluation commands accept a --system prompt override."""
    runner = CliRunner()
    for name in ("fine-tune", "prompt-tune", "qlora", "few-shot"):
        result = runner.invoke(app, [name, "--help"])

        assert result.exit_code == 0, name
        assert "--system" in result.output, name


def test_prompt_tune_exposes_num_virtual() -> None:
    """Only prompt-tune takes a virtual token count override."""
    runner = CliRunner()
    result = runner.invoke(app, ["prompt-tune", "--help"])

    assert result.exit_code == 0
    assert "--num-virtual" in result.output

    for name in ("fine-tune", "qlora", "few-shot"):
        other = runner.invoke(app, [name, "--help"])

        assert "--num-virtual" not in other.output, name


def test_epochs_rejects_zero() -> None:
    result = CliRunner().invoke(
        app,
        [
            "fine-tune",
            "--model",
            "some-model",
            "--dataset",
            "oblique",
            "--epochs",
            "0",
        ],
    )

    assert result.exit_code != 0


def test_slurm_rejects_non_training_command() -> None:
    result = CliRunner().invoke(app, ["slurm", "generate", "--", "--model", "x"])

    assert result.exit_code != 0
    assert "cannot submit" in result.output


def test_completion_requires_system_prompt() -> None:
    result = CliRunner().invoke(app, ["completion", "hello"])

    assert result.exit_code != 0
    assert "system prompt is required" in result.output


def test_slurm_dry_run_writes_script(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path
    monkeypatch.chdir(root)
    result = CliRunner().invoke(
        app,
        [
            "slurm",
            "--dry-run",
            "--job-name",
            "test-job",
            "fine-tune",
            "--",
            "--model",
            "some-model",
        ],
    )

    assert result.exit_code == 0
    assert (root / "log" / "slurm" / "test-job.sbatch").is_file()

"""End to end CLI tests on tiny-random models with small data subsets."""

from typing import TYPE_CHECKING, Final

import pytest
import torch
from click.testing import CliRunner

from eh30pt.cli.cli import app
from test.conftest import models
from test.e2e.conftest import single_run

if TYPE_CHECKING:
    from pathlib import Path

    from click.testing import Result


_model_params: Final = [pytest.param(name, id=name) for name in models]


def _invoke(args: list[str]) -> Result:
    return CliRunner().invoke(app, args)


def test_help_lists_commands() -> None:
    result = _invoke(["--help"])

    assert result.exit_code == 0
    assert "fine-tune" in result.output
    assert "prompt-tune" in result.output
    assert "qlora" in result.output
    assert "slurm" in result.output
    assert "few-shot" in result.output
    assert "completion" in result.output
    assert "evaluate" in result.output
    assert "generate" in result.output


def test_rejects_unknown_dataset() -> None:
    result = _invoke(
        [
            "fine-tune",
            "--model",
            "hf-internal-testing/tiny-random-bert",
            "--dataset",
            "unknown",
        ],
    )

    assert result.exit_code != 0


def test_rejects_extraction_dataset_for_training() -> None:
    result = _invoke(
        [
            "fine-tune",
            "--model",
            "hf-internal-testing/tiny-random-bert",
            "--dataset",
            "discont-extract/trigger",
            "--epochs",
            "1",
        ],
    )

    assert result.exit_code != 0
    assert "extraction dataset" in str(result.exception)


def test_rejects_classification_dataset_for_langextract() -> None:
    result = _invoke(
        [
            "langextract",
            "--dataset",
            "multinerd",
        ],
    )

    assert result.exit_code != 0
    assert "classification dataset" in str(result.exception)


def test_rejects_unknown_provider() -> None:
    result = _invoke(
        [
            "evaluate",
            "--dataset",
            "oblique",
            "--provider",
            "unknown",
        ],
    )

    assert result.exit_code != 0


def test_rejects_classifier_without_projects() -> None:
    result = _invoke(["generate", "--dataset", "discont-classifier/extract-labels"])

    assert result.exit_code != 0


def test_rejects_unknown_dataset_source() -> None:
    result = _invoke(["generate", "--dataset", "discont-extract/extract-labels"])

    assert result.exit_code != 0


@pytest.mark.slow
@pytest.mark.parametrize("model_name", _model_params)
def test_few_shot_obl(
    tracking_uri: str,
    model_name: str,
) -> None:
    experiment: Final = f"e2e-few-shot-{model_name}"
    result = _invoke(
        [
            "few-shot",
            "--model",
            models[model_name],
            "--dataset",
            "oblique",
            "--batch-size",
            "8",
            "--seed",
            "0",
            "--mlflow-experiment",
            experiment,
            "--mlflow-tracking-uri",
            tracking_uri,
        ],
    )

    assert result.exit_code == 0, result.output

    run = single_run(tracking_uri, experiment)
    assert run.data.params["method"] == "few-shot"
    assert "test_f1" in run.data.metrics


@pytest.mark.slow
@pytest.mark.parametrize("model_name", _model_params)
def test_fine_tune_obl(
    tmp_path: Path,
    tracking_uri: str,
    model_name: str,
) -> None:
    experiment: Final = f"e2e-fine-tune-{model_name}"
    result = _invoke(
        [
            "fine-tune",
            "--model",
            models[model_name],
            "--dataset",
            "oblique",
            "--epochs",
            "1",
            "--train-size",
            "16",
            "--save-model",
            "--batch-size",
            "8",
            "--seed",
            "0",
            "--mlflow-experiment",
            experiment,
            "--mlflow-tracking-uri",
            tracking_uri,
        ],
    )

    assert result.exit_code == 0, result.output

    run = single_run(tracking_uri, experiment)
    assert "test_f1" in run.data.metrics

    run_dir = tmp_path / "log" / run.info.run_id
    assert run_dir.is_dir()
    assert any(
        child.is_dir() and child.name.startswith("checkpoint-")
        for child in run_dir.iterdir()
    )


@pytest.mark.slow
@pytest.mark.parametrize("model_name", _model_params)
def test_qlora_obl(
    tracking_uri: str,
    model_name: str,
) -> None:
    if not torch.cuda.is_available() and not torch.mps.is_available():
        pytest.xfail(reason="qlora requires CUDA or MPS for bitsandbytes 4-bit")

    if model_name == "t5" and torch.mps.is_available():
        pytest.xfail(reason="mps sdpa does not support dropout in training")

    if model_name in {"bert", "modernbert", "gemma"}:
        pytest.xfail(reason="qlora unsupported without custom target modules")

    experiment: Final = f"e2e-qlora-{model_name}"
    result = _invoke(
        [
            "qlora",
            "--model",
            models[model_name],
            "--dataset",
            "oblique",
            "--epochs",
            "1",
            "--train-size",
            "16",
            "--batch-size",
            "8",
            "--seed",
            "0",
            "--mlflow-experiment",
            experiment,
            "--mlflow-tracking-uri",
            tracking_uri,
        ],
    )

    assert result.exit_code == 0, result.output

    run = single_run(tracking_uri, experiment)
    assert run.data.params["method"] == "qlora"
    assert "test_f1" in run.data.metrics


@pytest.mark.slow
@pytest.mark.parametrize("model_name", _model_params)
def test_prompt_tune_obl(
    tracking_uri: str,
    model_name: str,
) -> None:
    if model_name == "t5" and torch.mps.is_available():
        pytest.xfail(reason="mps sdpa does not support dropout in training")

    experiment: Final = f"e2e-prompt-tune-{model_name}"
    result = _invoke(
        [
            "prompt-tune",
            "--model",
            models[model_name],
            "--dataset",
            "oblique",
            "--epochs",
            "1",
            "--train-size",
            "16",
            "--batch-size",
            "8",
            "--seed",
            "0",
            "--mlflow-experiment",
            experiment,
            "--mlflow-tracking-uri",
            tracking_uri,
        ],
    )

    assert result.exit_code == 0, result.output

    run = single_run(tracking_uri, experiment)
    assert run.data.params["method"] == "prompt-tune"
    assert "test_f1" in run.data.metrics

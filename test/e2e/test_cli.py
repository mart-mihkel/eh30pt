"""End to end CLI tests on tiny-random models with small data subsets."""

from typing import TYPE_CHECKING, Final

import pytest
from mlflow.tracking import MlflowClient
from typer.testing import CliRunner

from eh30pt.cli import app
from test.conftest import models

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

    from mlflow.entities import Run
    from typer.testing import Result

pytestmark: Final = pytest.mark.filterwarnings(
    "ignore:'pin_memory' argument is set as true.*:UserWarning"
)


# TODO: resolve these problems  # noqa: FIX002, TD002, TD003
_xfail_reasons: Final[Mapping[str, str]] = {
    "gpt2": "oblique decoder prompts tokenize past tiny-random-gpt2's 512 positions",
    "t5": "seq2seq generate rejects the 'truncated' batch column",
}

_model_params: Final = [
    pytest.param(
        name,
        id=name,
        marks=pytest.mark.xfail(reason=_xfail_reasons[name], strict=True)
        if name in _xfail_reasons
        else (),
    )
    for name in sorted(models)
]


def _tracking_uri(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> str:
    monkeypatch.chdir(tmp_path)
    return f"sqlite:///{tmp_path}/mlflow.db"


def _invoke(args: list[str]) -> Result:
    return CliRunner().invoke(app, args)


def _single_run(client: MlflowClient, experiment: str) -> Run:
    tracked = client.get_experiment_by_name(experiment)
    assert tracked is not None

    runs = client.search_runs([tracked.experiment_id])
    assert len(runs) == 1

    return runs[0]


def test_help_lists_commands() -> None:
    result = _invoke(["--help"])

    assert result.exit_code == 0
    assert "fine-tune" in result.output
    assert "prompt-tune" in result.output
    assert "few-shot" in result.output


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


@pytest.mark.slow
@pytest.mark.parametrize("model_name", _model_params)
def test_few_shot_oblique(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    model_name: str,
) -> None:
    uri = _tracking_uri(tmp_path, monkeypatch)
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
            uri,
        ],
    )

    assert result.exit_code == 0, result.output

    run = _single_run(MlflowClient(uri), experiment)
    assert run.data.params["method"] == "0-shot"
    assert "test_f1" in run.data.metrics


@pytest.mark.slow
@pytest.mark.parametrize("model_name", _model_params)
def test_fine_tune_oblique(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    model_name: str,
) -> None:
    uri = _tracking_uri(tmp_path, monkeypatch)
    experiment: Final = f"e2e-fine-tune-{model_name}"
    result = _invoke(
        [
            "fine-tune",
            "--model",
            models[model_name],
            "--dataset",
            "oblique",
            "--train-samples",
            "8",
            "--epochs",
            "1",
            "--batch-size",
            "8",
            "--seed",
            "0",
            "--mlflow-experiment",
            experiment,
            "--mlflow-tracking-uri",
            uri,
        ],
    )

    assert result.exit_code == 0, result.output

    run = _single_run(MlflowClient(uri), experiment)
    assert "test_f1" in run.data.metrics

    run_dir = tmp_path / "log" / run.info.run_id
    assert run_dir.is_dir()
    assert any(
        child.is_dir() and child.name.startswith("checkpoint-")
        for child in run_dir.iterdir()
    )

    collected = _invoke(
        [
            "collect",
            "--mlflow-experiment",
            experiment,
            "--mlflow-tracking-uri",
            uri,
            "--metric",
            "test_f1",
        ],
    )

    assert collected.exit_code == 0, collected.output
    assert (tmp_path / "log" / "metrics" / f"{experiment}.csv").is_file()
    assert (tmp_path / "log" / "metrics" / f"{experiment}-history.csv").is_file()


@pytest.mark.slow
@pytest.mark.parametrize("model_name", _model_params)
def test_prompt_tune_oblique(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    model_name: str,
) -> None:
    uri = _tracking_uri(tmp_path, monkeypatch)
    experiment: Final = f"e2e-prompt-tune-{model_name}"
    result = _invoke(
        [
            "prompt-tune",
            "--model",
            models[model_name],
            "--dataset",
            "oblique",
            "--train-samples",
            "8",
            "--epochs",
            "1",
            "--batch-size",
            "8",
            "--seed",
            "0",
            "--mlflow-experiment",
            experiment,
            "--mlflow-tracking-uri",
            uri,
        ],
    )

    assert result.exit_code == 0, result.output

    run = _single_run(MlflowClient(uri), experiment)
    assert run.data.params["method"] == "prompt-tune-pretrained"
    assert "test_f1" in run.data.metrics

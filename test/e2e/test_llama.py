"""End to end llama-server CLI tests backed by recorded cassettes."""

from typing import TYPE_CHECKING, Final

import pytest
from mlflow.tracking import MlflowClient
from typer.testing import CliRunner

from eh30pt.cli.llama import app
from test.e2e.conftest import single_run

if TYPE_CHECKING:
    from typer.testing import Result

    from eh30pt.cli.scripts.infer import EvalData
    from eh30pt.types import DatasetName


@pytest.fixture(scope="module")
def vcr_config() -> dict[str, object]:
    return {
        "filter_headers": ["authorization"],
        "ignore_hosts": [
            "api.mlflow-telemetry.io",
            "config.mlflow-telemetry.io",
        ],
    }


def _invoke(args: list[str]) -> Result:
    return CliRunner().invoke(app, args)


def _stub_eval_data(dataset: DatasetName, n_shot: int) -> EvalData:
    from eh30pt.cli.scripts.infer import EvalData
    from eh30pt.datasets.registry import get_labels
    from eh30pt.types import EvalExample

    assert dataset == "multinerd"
    assert n_shot == 1

    example = EvalExample(
        user="Sentence: Barack Obama was president.\nEntity: Barack Obama\nTag:",
        label="PER",
    )

    samples: Final = [
        example,
        EvalExample(
            user="Sentence: Tallinn is the capital of Estonia.\nEntity: Tallinn\nTag:",
            label="LOC",
        ),
    ]

    return EvalData(
        examples=[example],
        samples=samples,
        labels=get_labels(dataset),
    )


def test_help_lists_commands() -> None:
    result = _invoke(["--help"])

    assert result.exit_code == 0
    assert "completion" in result.output
    assert "infer" in result.output


def test_rejects_unknown_dataset() -> None:
    result = _invoke(
        [
            "completion",
            "hello",
            "--system",
            "You are terse.",
            "--dataset",
            "unknown",
        ],
    )

    assert result.exit_code != 0


@pytest.mark.vcr
def test_completion_prints_completion(caplog: pytest.LogCaptureFixture) -> None:
    result = _invoke(
        [
            "completion",
            "What is 2+2? Answer in one word.",
            "--system",
            "You are a terse calculator, answer with the number.",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "4" in caplog.text


@pytest.mark.vcr
def test_completion_dataset_prints_label(
    caplog: pytest.LogCaptureFixture,
) -> None:
    result = _invoke(
        [
            "completion",
            "Sentence: Barack Obama was president.\nEntity: Barack Obama\nTag:",
            "--system",
            "You are a NER tagger.",
            "--dataset",
            "multinerd",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "PER" in caplog.text


@pytest.mark.vcr
def test_infer_one_conv(
    monkeypatch: pytest.MonkeyPatch,
    tracking_uri: str,
) -> None:
    monkeypatch.setattr(
        "eh30pt.cli.scripts.infer.load_eval_data",
        _stub_eval_data,
    )

    experiment: Final = "e2e-infer-one-conv-multinerd"
    result = _invoke(
        [
            "infer",
            "--system",
            "Classify the text.",
            "--dataset",
            "multinerd",
            "--n-shot",
            "1",
            "--one-conv",
            "--seed",
            "7",
            "--mlflow-experiment",
            experiment,
            "--mlflow-tracking-uri",
            tracking_uri,
        ],
    )

    assert result.exit_code == 0, result.output

    run = single_run(MlflowClient(tracking_uri), experiment)
    assert run.data.params["method"] == "1-shot"
    assert run.data.params["one_conv"] == "True"
    assert "test_f1" in run.data.metrics

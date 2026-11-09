"""End to end llama-server CLI tests backed by recorded cassettes."""

from typing import TYPE_CHECKING, Final

import mlflow
import mlflow.artifacts
import polars as pl
import pytest
from click.testing import CliRunner
from mlflow.tracking import MlflowClient

from eh30pt.cli.cli import app
from eh30pt.datasets.registry import get_labels
from eh30pt.types import GenerateRecord
from test.e2e.conftest import single_run

if TYPE_CHECKING:
    from pathlib import Path

    from click.testing import Result


@pytest.fixture(scope="module")
def vcr_config() -> dict[str, object]:
    return {
        "filter_headers": ["authorization"],
        "ignore_hosts": [
            "api.mlflow-telemetry.io",
            "config.mlflow-telemetry.io",
            "huggingface.co",
            "cas-bridge.xethub.huggingface.co",
            "datasets-server.huggingface.co",
            "s3.amazonaws.com",
        ],
    }


def _invoke(args: list[str]) -> Result:
    return CliRunner().invoke(app, args)


@pytest.mark.parametrize("command", ["evaluate", "generate"])
def test_systemone_flag_rejects_vllm(command: str) -> None:
    result = _invoke([command, "-d", "oblique", "--provider", "vllm", "--systemone"])
    assert result.exit_code != 0
    assert "System One requires llama" in result.output


@pytest.mark.parametrize(
    "command",
    ["evaluate", "generate", "silver-labels"],
)
@pytest.mark.parametrize("port", ["0", "65536"])
def test_server_commands_reject_invalid_port(command: str, port: str) -> None:
    result = _invoke([command, "--port", port])
    assert result.exit_code != 0
    assert "--port" in result.output


@pytest.mark.vcr
def test_systemone_evaluate_reports_metrics(
    tracking_uri: str,
    caplog: pytest.LogCaptureFixture,
) -> None:
    experiment = "e2e-systemone-evaluate-oblique"
    result = _invoke(
        [
            "evaluate",
            "--dataset",
            "oblique",
            "--systemone",
            "--port",
            "9932",
            "--concurrency",
            "1",
            "--seed",
            "7",
            "--mlflow-experiment",
            experiment,
            "--mlflow-tracking-uri",
            tracking_uri,
        ]
    )

    assert result.exit_code == 0, result.output
    run = single_run(tracking_uri, experiment)
    assert run.data.params["method"] == "systemone"
    assert run.data.params["dataset"] == "oblique"
    assert 0.0 <= run.data.metrics["test_accuracy"] <= 1.0
    assert "invalid responses" not in caplog.text


@pytest.mark.vcr
def test_systemone_generate_saves_predictions(tracking_uri: str) -> None:
    experiment = "e2e-systemone-generate-oblique"
    result = _invoke(
        [
            "generate",
            "--dataset",
            "oblique",
            "--n-shot",
            "1",
            "--systemone",
            "--port",
            "9932",
            "--concurrency",
            "1",
            "--seed",
            "7",
            "--mlflow-experiment",
            experiment,
            "--mlflow-tracking-uri",
            tracking_uri,
        ]
    )

    assert result.exit_code == 0, result.output
    run = single_run(tracking_uri, experiment)
    assert run.data.params["method"] == "systemone"
    shards = MlflowClient(tracking_uri).list_artifacts(run.info.run_id, "predictions")
    assert len(shards) == 1
    downloaded = mlflow.artifacts.download_artifacts(
        run_id=run.info.run_id,
        artifact_path=shards[0].path,
        tracking_uri=tracking_uri,
    )
    predictions = pl.read_parquet(downloaded)["prediction"].to_list()
    assert predictions
    assert set(predictions) <= set(get_labels("oblique"))


@pytest.mark.vcr
def test_evaluate_reports_metrics(tracking_uri: str) -> None:
    experiment: Final = "e2e-llama-evaluate-multinerd"
    result = _invoke(
        [
            "evaluate",
            "--dataset",
            "multinerd",
            "--n-shot",
            "1",
            "--system",
            "Classify the text.",
            "--seed",
            "7",
            "--mlflow-experiment",
            experiment,
            "--mlflow-tracking-uri",
            tracking_uri,
        ],
    )

    assert result.exit_code == 0, result.output

    run = single_run(tracking_uri, experiment)
    assert run.data.params["method"] == "few-shot"
    assert run.data.params["dataset"] == "multinerd"

    for metric in ("test_accuracy", "test_precision", "test_recall", "test_f1"):
        assert metric in run.data.metrics
        assert 0.0 <= run.data.metrics[metric] <= 1.0


@pytest.mark.vcr
def test_generate_saves_predictions(
    tmp_path: Path,
    tracking_uri: str,
) -> None:
    experiment: Final = "e2e-llama-generate-multinerd"
    result = _invoke(
        [
            "generate",
            "--dataset",
            "multinerd",
            "--n-shot",
            "1",
            "--system",
            "Classify the text.",
            "--seed",
            "7",
            "--shard-size",
            "16",
            "--mlflow-experiment",
            experiment,
            "--mlflow-tracking-uri",
            tracking_uri,
        ],
    )

    assert result.exit_code == 0, result.output

    client = MlflowClient(tracking_uri)
    run = single_run(tracking_uri, experiment)
    assert run.data.params["method"] == "few-shot"

    shards = client.list_artifacts(run.info.run_id, "predictions")
    shard_paths = sorted(info.path for info in shards if info.path.endswith(".parquet"))
    assert len(shard_paths) > 1

    shard_sizes: list[int] = []
    records: list[GenerateRecord] = []
    for shard in shard_paths:
        downloaded = mlflow.artifacts.download_artifacts(
            run_id=run.info.run_id,
            artifact_path=shard,
            dst_path=str(tmp_path),
            tracking_uri=tracking_uri,
        )

        rows = pl.read_parquet(downloaded).to_dicts()
        shard_sizes.append(len(rows))
        records.extend(GenerateRecord(**row) for row in rows)

    assert shard_sizes[:-1] == [16] * (len(shard_sizes) - 1)

    records.sort(key=lambda record: record.id)
    assert [record.id for record in records] == list(range(len(records)))
    assert len(records) > 0

    labels = set(get_labels("multinerd")) | {"other"}
    for record in records:
        assert record.prediction in labels
        assert isinstance(record.user, str)

    logs = client.list_artifacts(run.info.run_id, "logs")
    assert [info.path for info in logs if info.path.endswith("run.log")]

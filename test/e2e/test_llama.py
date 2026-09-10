"""End to end llama-server CLI tests backed by recorded cassettes."""

import re
from typing import TYPE_CHECKING, Final

import mlflow
import mlflow.artifacts
import polars as pl
import pytest
from mlflow.tracking import MlflowClient
from typer.testing import CliRunner

from eh30pt.cli.cli import app
from eh30pt.datasets.registry import get_labels
from eh30pt.types import GenerateRecord
from test.e2e.conftest import single_run

if TYPE_CHECKING:
    from pathlib import Path

    from typer.testing import Result


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

    match = re.search(r"completion: (\w+)", caplog.text)
    assert match is not None
    assert match.group(1) in get_labels("multinerd")


@pytest.mark.vcr
@pytest.mark.slow
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

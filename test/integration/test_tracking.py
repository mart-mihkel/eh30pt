"""Tracking integration tests against a real sqlite-backed MLflow store."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Final, cast

import mlflow
import mlflow.artifacts
import polars as pl
import pytest
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient

from eh30pt.logging import logger
from eh30pt.settings import MlflowSettings
from eh30pt.tracking.context import RunContext
from eh30pt.tracking.resume import (
    ConfigMismatchError,
    _identity_view,
    _verify_config,
)
from eh30pt.tracking.shards import log_shard
from eh30pt.types import GenerateRecord

if TYPE_CHECKING:
    from collections.abc import Iterator

    from transformers import PreTrainedTokenizerFast

    from eh30pt.types import RunConfig


_winning_f1: Final = 0.7
_second_version: Final = 2


@pytest.fixture
def mlflow_client(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[MlflowClient]:
    """Point fluent tracking at an isolated sqlite store, keeping a client for src."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MLFLOW_DISABLE_TELEMETRY", "true")
    uri = f"sqlite:///{tmp_path}/mlflow.db"
    mlflow.set_tracking_uri(uri)
    mlflow.set_registry_uri(uri)
    client = MlflowClient(tracking_uri=uri, registry_uri=uri)

    yield client

    if mlflow.active_run() is not None:
        mlflow.end_run()


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
        "system_prompt": "original prompt",
        "total_parameters": 100,
        "trainable_parameters": 10,
    }


def _track_experiment(name: str = "experiment") -> str:
    mlflow.set_experiment(name)
    experiment = mlflow.get_experiment_by_name(name)
    assert experiment is not None

    return experiment.experiment_id


def _stored_run(
    client: MlflowClient,
    config: RunConfig,
    *,
    with_artifact: bool = True,
) -> str:
    _ = client
    _track_experiment()
    with mlflow.start_run() as run:
        for key, value in _identity_view(config).items():
            mlflow.log_param(key, str(value))

        if with_artifact:
            mlflow.log_dict(dict(config), "config.json")

        return run.info.run_id


def test_verify_accepts_volatile_drift(
    mlflow_client: MlflowClient,
    run_config: RunConfig,
) -> None:
    run_id = _stored_run(mlflow_client, run_config)
    edited = cast(
        "RunConfig",
        dict(
            run_config,
            system_prompt="edited prompt",
            total_parameters=555,
            trainable_parameters=55,
        ),
    )

    _verify_config(mlflow_client, run_id, edited)


def test_verify_rejects_identity_mismatch(
    mlflow_client: MlflowClient,
    run_config: RunConfig,
) -> None:
    run_id = _stored_run(mlflow_client, run_config)
    edited = cast("RunConfig", dict(run_config, learning_rate=1e-4))

    with pytest.raises(ConfigMismatchError, match="learning_rate"):
        _verify_config(mlflow_client, run_id, edited)


def test_verify_falls_back_to_params_without_artifact(
    mlflow_client: MlflowClient,
    run_config: RunConfig,
) -> None:
    run_id = _stored_run(mlflow_client, run_config, with_artifact=False)

    _verify_config(mlflow_client, run_id, run_config)


def _best_metric(client: MlflowClient, name: str) -> float | None:
    try:
        current = client.get_model_version_by_alias(name, "best")
    except MlflowException:
        return None

    return float(current.tags["test_f1"]) if "test_f1" in current.tags else None


def test_register_best_model_max_beats(
    mlflow_client: MlflowClient, run_config: RunConfig
) -> None:
    mlflow_settings = MlflowSettings(
        tracking_uri=mlflow_client.tracking_uri, experiment="registry-exp"
    )
    with RunContext(mlflow_settings, config=run_config) as ctx:
        version = ctx.register_best_model("model", {"test_f1": _winning_f1})

    assert version == 1
    assert _best_metric(mlflow_client, "model") == _winning_f1


def test_register_best_model_max_skips_ties_and_losses(
    mlflow_client: MlflowClient, run_config: RunConfig
) -> None:
    mlflow_settings = MlflowSettings(
        tracking_uri=mlflow_client.tracking_uri, experiment="registry-exp"
    )
    with RunContext(mlflow_settings, config=run_config) as ctx:
        ctx.register_best_model("model", {"test_f1": 0.7})

        assert ctx.register_best_model("model", {"test_f1": 0.7}) is None
        assert ctx.register_best_model("model", {"test_f1": 0.6}) is None


def test_register_best_model_min_beats_lower(
    mlflow_client: MlflowClient, run_config: RunConfig
) -> None:
    mlflow_settings = MlflowSettings(
        tracking_uri=mlflow_client.tracking_uri, experiment="registry-exp"
    )
    with RunContext(mlflow_settings, config=run_config) as ctx:
        ctx.register_best_model("model", {"test_loss": 0.5}, "test_loss", "min")

        assert (
            ctx.register_best_model("model", {"test_loss": 0.3}, "test_loss", "min")
            == _second_version
        )
        assert (
            ctx.register_best_model("model", {"test_loss": 0.9}, "test_loss", "min")
            is None
        )


def test_register_best_model_without_metric_skips(
    mlflow_client: MlflowClient, run_config: RunConfig
) -> None:
    mlflow_settings = MlflowSettings(
        tracking_uri=mlflow_client.tracking_uri, experiment="registry-exp"
    )
    with RunContext(mlflow_settings, config=run_config) as ctx:
        ctx.register_best_model("model", {"test_f1": 0.5})

        assert ctx.register_best_model("model", {"test_accuracy": 0.9}) is None


def test_run_context_logs_dataset_input(
    mlflow_client: MlflowClient,
    run_config: RunConfig,
) -> None:
    mlflow.set_experiment("dataset-exp")

    mlflow_settings = MlflowSettings(
        tracking_uri=mlflow_client.tracking_uri, experiment="dataset-exp"
    )
    with RunContext(mlflow_settings, config=run_config) as ctx:
        run_id = ctx.run_id

    run_inputs = mlflow.get_run(run_id).inputs

    assert run_inputs is not None
    inputs = run_inputs.dataset_inputs

    assert len(inputs) == 1
    assert inputs[0].dataset.name == run_config["dataset"]


def test_log_model_artifacts_links_flavored_model_to_run(
    mlflow_client: MlflowClient,
    run_config: RunConfig,
) -> None:
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    mlflow.set_experiment("model-exp")
    experiment = mlflow.get_experiment_by_name("model-exp")
    assert experiment is not None
    experiment_id = experiment.experiment_id
    model = AutoModelForSequenceClassification.from_pretrained(
        "hf-internal-testing/tiny-random-bert"
    )
    tokenizer = cast(
        "PreTrainedTokenizerFast",
        AutoTokenizer.from_pretrained("hf-internal-testing/tiny-random-bert"),
    )

    mlflow_settings = MlflowSettings(
        tracking_uri=mlflow_client.tracking_uri, experiment="model-exp"
    )
    with RunContext(mlflow_settings, config=run_config) as ctx:
        ctx.log_model_artifacts(model, tokenizer, task="text-classification")
        run_id = ctx.run_id

    models = mlflow.search_logged_models(
        experiment_ids=[experiment_id],
        output_format="list",
    )
    assert isinstance(models, list)

    assert len(models) == 1
    assert models[0].source_run_id == run_id


def test_run_context_opens_fresh_run(
    mlflow_client: MlflowClient,
    run_config: RunConfig,
) -> None:
    mlflow.set_experiment("fresh-exp")

    mlflow_settings = MlflowSettings(
        tracking_uri=mlflow_client.tracking_uri, experiment="fresh-exp"
    )
    with RunContext(mlflow_settings, config=run_config) as ctx:
        assert ctx.resumed is False
        run_id = ctx.run_id

    assert mlflow.get_run(run_id).info.status == "FINISHED"


def test_run_context_reattaches_by_explicit_id(
    mlflow_client: MlflowClient,
    run_config: RunConfig,
) -> None:
    mlflow.set_experiment("explicit-exp")

    mlflow_settings = MlflowSettings(
        tracking_uri=mlflow_client.tracking_uri, experiment="explicit-exp"
    )
    with RunContext(mlflow_settings, config=run_config) as ctx:
        run_id = ctx.run_id

    mlflow_settings = MlflowSettings(
        tracking_uri=mlflow_client.tracking_uri,
        experiment="explicit-exp",
        run_id=run_id,
    )
    with RunContext(mlflow_settings, config=run_config) as second:
        assert second.run_id == run_id
        assert second.resumed is True


def test_run_context_rejects_config_mismatch(
    mlflow_client: MlflowClient,
    run_config: RunConfig,
) -> None:
    mlflow.set_experiment("mismatch-exp")

    mlflow_settings = MlflowSettings(
        tracking_uri=mlflow_client.tracking_uri, experiment="mismatch-exp"
    )
    with RunContext(mlflow_settings, config=run_config) as ctx:
        run_id = ctx.run_id

    other: RunConfig = {**run_config, "n_shot": run_config["n_shot"] + 1}

    with (
        pytest.raises(ConfigMismatchError, match="different config"),
        RunContext(
            MlflowSettings(
                tracking_uri=mlflow_client.tracking_uri,
                experiment="mismatch-exp",
                run_id=run_id,
            ),
            config=other,
        ),
    ):
        pass


def test_run_context_rejects_foreign_experiment(
    mlflow_client: MlflowClient,
    run_config: RunConfig,
) -> None:
    mlflow.set_experiment("home-exp")

    mlflow_settings = MlflowSettings(
        tracking_uri=mlflow_client.tracking_uri, experiment="home-exp"
    )
    with RunContext(mlflow_settings, config=run_config) as ctx:
        run_id = ctx.run_id

    with (
        pytest.raises(ConfigMismatchError, match="not in experiment"),
        RunContext(
            MlflowSettings(
                tracking_uri=mlflow_client.tracking_uri,
                experiment="away-exp",
                run_id=run_id,
            ),
            config=run_config,
        ),
    ):
        pass


def test_run_context_writes_log_file_and_artifact(
    mlflow_client: MlflowClient,
    run_config: RunConfig,
    tmp_path: Path,
) -> None:
    mlflow.set_experiment("log-exp")
    previous = logger.level
    logger.setLevel(logging.INFO)

    mlflow_settings = MlflowSettings(
        tracking_uri=mlflow_client.tracking_uri, experiment="log-exp"
    )
    try:
        with RunContext(mlflow_settings, config=run_config) as ctx:
            run_id = ctx.run_id
            logger.info("hello run log")
    finally:
        logger.setLevel(previous)

    lines = (tmp_path / "log" / run_id / "run.log").read_text(encoding="utf-8")
    assert "hello run log" in lines
    assert "\x1b" not in lines

    downloaded = mlflow.artifacts.download_artifacts(
        run_id=run_id,
        artifact_path="logs/run.log",
        dst_path=str(tmp_path / "download"),
    )
    assert "hello run log" in Path(downloaded).read_text(encoding="utf-8")


def test_run_context_continuation_appends_to_log(
    mlflow_client: MlflowClient,
    run_config: RunConfig,
    tmp_path: Path,
) -> None:
    mlflow.set_experiment("log-continue-exp")
    previous = logger.level
    logger.setLevel(logging.INFO)

    mlflow_settings = MlflowSettings(
        tracking_uri=mlflow_client.tracking_uri, experiment="log-continue-exp"
    )
    try:
        with RunContext(mlflow_settings, config=run_config) as ctx:
            run_id = ctx.run_id
            logger.info("first chunk")

        mlflow_settings = MlflowSettings(
            tracking_uri=mlflow_client.tracking_uri,
            experiment="log-continue-exp",
            run_id=run_id,
        )
        with RunContext(mlflow_settings, config=run_config):
            logger.info("second chunk")
    finally:
        logger.setLevel(previous)

    downloaded = mlflow.artifacts.download_artifacts(
        run_id=run_id,
        artifact_path="logs/run.log",
        dst_path=str(tmp_path / "download"),
    )
    text = Path(downloaded).read_text(encoding="utf-8")

    assert text.index("first chunk") < text.index("second chunk")


def test_failed_run_log_captures_traceback(
    mlflow_client: MlflowClient,
    run_config: RunConfig,
    tmp_path: Path,
) -> None:
    mlflow.set_experiment("log-failure-exp")
    seen: list[str] = []

    def _record_and_boom(run_id: str) -> None:
        seen.append(run_id)
        msg = "boom"
        raise RuntimeError(msg)

    with (
        pytest.raises(RuntimeError, match="boom"),
        RunContext(
            MlflowSettings(
                tracking_uri=mlflow_client.tracking_uri, experiment="log-failure-exp"
            ),
            config=run_config,
        ) as ctx,
    ):
        _record_and_boom(ctx.run_id)

    text = (tmp_path / "log" / seen[0] / "run.log").read_text(encoding="utf-8")

    assert "Traceback" in text
    assert "boom" in text
    assert "\x1b" not in text
    assert mlflow.get_run(seen[0]).info.status == "FAILED"


def test_enter_failure_marks_run_failed_and_saves_log(
    mlflow_client: MlflowClient,
    run_config: RunConfig,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    experiment_id = _track_experiment("enter-failure-exp")

    def _boom(dictionary: dict[str, object], artifact_file: str) -> None:
        _ = dictionary, artifact_file
        msg = "boom"
        raise RuntimeError(msg)

    monkeypatch.setattr(mlflow, "log_dict", _boom)
    handlers_before = list(logger.handlers)
    mlflow_settings = MlflowSettings(
        tracking_uri=mlflow_client.tracking_uri, experiment="enter-failure-exp"
    )

    with (
        pytest.raises(RuntimeError, match="boom"),
        RunContext(mlflow_settings, config=run_config),
    ):
        pass

    assert logger.handlers == handlers_before

    [run] = mlflow.search_runs(
        experiment_ids=[experiment_id],
        output_format="list",
    )

    assert run.info.status == "FAILED"

    downloaded = mlflow.artifacts.download_artifacts(
        run_id=run.info.run_id,
        artifact_path="logs/run.log",
        dst_path=str(tmp_path / "download"),
    )
    text = Path(downloaded).read_text(encoding="utf-8")

    assert "Traceback" in text
    assert "boom" in text
    assert "\x1b" not in text


def _read_downloaded(path: str) -> list[GenerateRecord]:
    return [GenerateRecord(**record) for record in pl.read_parquet(path).to_dicts()]


def test_log_shard_logs_and_removes_local_copy(
    mlflow_client: MlflowClient,
    run_config: RunConfig,
    tmp_path: Path,
) -> None:
    mlflow.set_experiment("shard-exp")
    batch: list[GenerateRecord] = [GenerateRecord(id=0, user="a", prediction="x")]
    index = 3

    mlflow_settings = MlflowSettings(
        tracking_uri=mlflow_client.tracking_uri, experiment="shard-exp"
    )
    with RunContext(mlflow_settings, config=run_config) as ctx:
        nxt = log_shard(ctx, tmp_path, "predictions", index, batch)
        run_id = ctx.run_id

        assert nxt == index + 1
        assert (tmp_path / "predictions-00003.parquet").exists() is False

    downloaded = mlflow.artifacts.download_artifacts(
        run_id=run_id,
        artifact_path="predictions/predictions-00003.parquet",
        dst_path=str(tmp_path / "download"),
    )

    assert _read_downloaded(downloaded) == batch


def test_log_shard_removes_local_copy_on_log_failure(
    mlflow_client: MlflowClient,
    run_config: RunConfig,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mlflow.set_experiment("shard-failure-exp")
    batch: list[GenerateRecord] = [GenerateRecord(id=0, user="a", prediction="x")]

    def _boom(local_path: str, artifact_path: str | None = None) -> None:
        _ = local_path, artifact_path
        msg = "boom"
        raise RuntimeError(msg)

    mlflow_settings = MlflowSettings(
        tracking_uri=mlflow_client.tracking_uri, experiment="shard-failure-exp"
    )
    with RunContext(mlflow_settings, config=run_config) as ctx:
        monkeypatch.setattr(mlflow, "log_artifact", _boom)

        with pytest.raises(RuntimeError, match="boom"):
            log_shard(ctx, tmp_path, "predictions", 0, batch)

    assert (tmp_path / "predictions-00000.parquet").exists() is False


def test_next_shard_index_continues_after_logged_shards(
    mlflow_client: MlflowClient,
    run_config: RunConfig,
    tmp_path: Path,
) -> None:
    mlflow.set_experiment("shard-index-exp")
    batch: list[GenerateRecord] = [GenerateRecord(id=5, user="a", prediction="x")]

    mlflow_settings = MlflowSettings(
        tracking_uri=mlflow_client.tracking_uri, experiment="shard-index-exp"
    )
    with RunContext(mlflow_settings, config=run_config) as ctx:
        assert ctx.next_shard_index() == 0

        log_shard(ctx, tmp_path, "predictions", 0, batch)
        log_shard(ctx, tmp_path, "predictions", 1, batch)

        expected = 2
        assert ctx.next_shard_index() == expected


def test_run_context_reattaches_to_finished_run(
    mlflow_client: MlflowClient,
    run_config: RunConfig,
) -> None:
    mlflow.set_experiment("reattach-exp")

    mlflow_settings = MlflowSettings(
        tracking_uri=mlflow_client.tracking_uri, experiment="reattach-exp"
    )
    with RunContext(mlflow_settings, config=run_config) as ctx:
        run_id = ctx.run_id

    mlflow_settings = MlflowSettings(
        tracking_uri=mlflow_client.tracking_uri,
        experiment="reattach-exp",
        run_id=run_id,
    )
    with RunContext(mlflow_settings, config=run_config) as ctx:
        assert ctx.run_id == run_id
        assert ctx.resumed is True
        ctx.record_from_id(100)
        ctx.record_from_id(200)

    assert mlflow.get_run(run_id).data.tags["eh30pt.from_id"] == "100,200"

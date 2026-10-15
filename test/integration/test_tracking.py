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

    from transformers import (
        PreTrainedModel,
        PreTrainedTokenizerFast,
    )

    from eh30pt.settings import TrainingSettings
    from eh30pt.types import DataSplits, RunConfig


_winning_f1: Final = 0.7
_second_version: Final = 2
_finish_f1: Final = 0.8


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
    from eh30pt.datasets.registry import get_sys_prompt
    from test.conftest import models

    return {
        "method": "fine-tune",
        "dataset": "oblique",
        "architecture": "encoder",
        "base_model": models["bert"],
        "n_shot": 0,
        "batch_size": 8,
        "system_prompt": get_sys_prompt("oblique"),
        "total_parameters": 100,
        "trainable_parameters": 10,
    }


def _real_records(n: int = 1, start_id: int = 0) -> list[GenerateRecord]:
    from eh30pt.datasets.registry import get_few_shots

    shots = get_few_shots("oblique", 2)
    return [
        GenerateRecord(
            id=start_id + idx,
            user=shots[idx % len(shots)].user,
            prediction=shots[idx % len(shots)].label,
        )
        for idx in range(n)
    ]


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
        tracking_uri=mlflow_client.tracking_uri,
        experiment="model-exp",
        run_name="named-run",
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
    from eh30pt.tracking.context import logged_model_name

    assert models[0].name == logged_model_name(run_config["base_model"])


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


def test_run_context_routes_mlflow_logs_to_rich(
    mlflow_client: MlflowClient,
    run_config: RunConfig,
    caplog: pytest.LogCaptureFixture,
) -> None:
    from rich.logging import RichHandler

    mlflow.set_experiment("rich-exp")

    mlflow_settings = MlflowSettings(
        tracking_uri=mlflow_client.tracking_uri, experiment="rich-exp"
    )
    with RunContext(mlflow_settings, config=run_config) as ctx:
        assert ctx.resumed is False
        mlflow_logger = logging.getLogger("mlflow")
        assert mlflow_logger.propagate
        assert all(
            isinstance(handler, RichHandler) for handler in mlflow_logger.handlers
        )
        with caplog.at_level("INFO", logger="mlflow"):
            mlflow_logger.info("rich mlflow probe")

    assert "rich mlflow probe" in caplog.text


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
    batch = _real_records(1)
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
    batch = _real_records(1)

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
    batch = _real_records(1, start_id=5)

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


def test_unopened_context_has_no_run_id(run_config: RunConfig) -> None:
    ctx = RunContext(MlflowSettings(experiment="unopened-exp"), config=run_config)

    with pytest.raises(RuntimeError, match="before use"):
        _ = ctx.run_id


def test_helpers_require_run_name(
    mlflow_client: MlflowClient,
    run_config: RunConfig,
    bert_tokenizer: PreTrainedTokenizerFast,
    oblique_bert_model: PreTrainedModel,
    oblique_bert_dataset: DataSplits,
) -> None:
    from eh30pt.modeling.collator import get_collator
    from eh30pt.modeling.metrics import MetricsAccumulator

    _ = mlflow_client
    mlflow.set_experiment("noname-exp")
    settings = MlflowSettings(
        tracking_uri=mlflow_client.tracking_uri, experiment="noname-exp"
    )
    metrics = MetricsAccumulator(bert_tokenizer, "encoder")
    with RunContext(settings, config=run_config) as ctx:
        ctx.settings.run_name = None

        with pytest.raises(ValueError, match="no mlflow run name"):
            ctx.log_prompt_version("oblique", "decoder", "prompt")

        with pytest.raises(ValueError, match="no mlflow run name"):
            ctx.evaluate_and_log(
                model=oblique_bert_model,
                tokenizer=bert_tokenizer,
                data=oblique_bert_dataset,
                arch="encoder",
                collate_fn=get_collator(bert_tokenizer, "encoder"),
                metrics_fn=metrics,
                batch_size=8,
                constrain=False,
            )

        with pytest.raises(ValueError, match="no mlflow run name"):
            ctx.train_and_evaluate(
                model=oblique_bert_model,
                tokenizer=bert_tokenizer,
                data=oblique_bert_dataset,
                arch="encoder",
                collate_fn=get_collator(bert_tokenizer, "encoder"),
                metrics_fn=metrics,
                registered_name="name",
                training=cast("TrainingSettings", None),
            )


def test_finish_training_logs_registers_model(
    mlflow_client: MlflowClient,
    run_config: RunConfig,
    bert_tokenizer: PreTrainedTokenizerFast,
    oblique_bert_model: PreTrainedModel,
) -> None:
    _ = mlflow_client
    mlflow.set_experiment("finish-exp")
    settings = MlflowSettings(
        tracking_uri=mlflow_client.tracking_uri, experiment="finish-exp"
    )
    with RunContext(settings, config=run_config) as ctx:
        version = ctx.register_best_model("finish-model", {"test_f1": _finish_f1})

        ctx.finish_training(
            "finish-model",
            model=oblique_bert_model,
            tokenizer=bert_tokenizer,
            task="text-classification",
            test_metrics={"test_f1": _finish_f1},
        )
        run_id = ctx.run_id

    assert version == 1
    assert mlflow.get_run(run_id).data.metrics["test_f1"] == _finish_f1


def test_finish_training_skips_model_without_saving(
    mlflow_client: MlflowClient,
    run_config: RunConfig,
    bert_tokenizer: PreTrainedTokenizerFast,
    oblique_bert_model: PreTrainedModel,
) -> None:
    _ = mlflow_client
    mlflow.set_experiment("finish-skip-exp")
    settings = MlflowSettings(
        tracking_uri=mlflow_client.tracking_uri, experiment="finish-skip-exp"
    )
    with RunContext(settings, config=run_config) as ctx:
        ctx.finish_training(
            "finish-skip-model",
            model=oblique_bert_model,
            tokenizer=bert_tokenizer,
            task="text-classification",
            test_metrics={"test_f1": _finish_f1},
            save_model=False,
        )
        run_id = ctx.run_id

    assert mlflow.get_run(run_id).data.metrics["test_f1"] == _finish_f1
    assert mlflow_client.list_artifacts(run_id, "model") == []


def test_get_experiment_id_rejects_missing(
    mlflow_client: MlflowClient,
) -> None:
    from eh30pt.tracking.resume import _get_experiment_id

    with pytest.raises(RuntimeError, match="not found after activation"):
        _get_experiment_id(mlflow_client, "definitely-missing-exp")


def test_next_shard_index_skips_unrelated_artifacts(
    mlflow_client: MlflowClient,
    run_config: RunConfig,
    tmp_path: Path,
) -> None:
    _ = mlflow_client
    mlflow.set_experiment("shard-skip-exp")
    settings = MlflowSettings(
        tracking_uri=mlflow_client.tracking_uri, experiment="shard-skip-exp"
    )
    with RunContext(settings, config=run_config) as ctx:
        note = tmp_path / "notes.txt"
        note.write_text("hi", encoding="utf-8")
        ctx.log_artifact(note, artifact_path="predictions")

        subdir = tmp_path / "subdir"
        subdir.mkdir()
        (subdir / "file.txt").write_text("x", encoding="utf-8")
        mlflow.log_artifacts(str(subdir), artifact_path="predictions/subdir")

        assert ctx.next_shard_index() == 0


def test_existing_max_step_tracks_logged_metrics(
    mlflow_client: MlflowClient, run_config: RunConfig
) -> None:
    _ = mlflow_client
    mlflow.set_experiment("max-step-exp")
    settings = MlflowSettings(
        tracking_uri=mlflow_client.tracking_uri, experiment="max-step-exp"
    )
    with RunContext(settings, config=run_config) as ctx:
        assert ctx.existing_max_step() == 0

        _later, _earlier = 5, 3
        ctx.log_metrics({"a": 1.0}, step=_later)
        ctx.log_metrics({"b": 2.0}, step=_earlier)

        assert ctx.existing_max_step() == _later


def test_register_prompt_version_reuses_unchanged_template(
    mlflow_client: MlflowClient, run_config: RunConfig
) -> None:
    from eh30pt.datasets.registry import get_sys_prompt

    _ = mlflow_client
    mlflow.set_experiment("prompt-reuse-exp")
    settings = MlflowSettings(
        tracking_uri=mlflow_client.tracking_uri, experiment="prompt-reuse-exp"
    )
    template = get_sys_prompt("oblique")
    with RunContext(settings, config=run_config) as ctx:
        first = ctx.register_prompt_version("reused", template, "first")
        second = ctx.register_prompt_version("reused", template, "second")

        assert first.version == second.version


def test_log_prompt_version_accepts_slashed_dataset(
    mlflow_client: MlflowClient, run_config: RunConfig
) -> None:
    """Register a prompt for datasets whose name carries a source slash."""
    from eh30pt.datasets.registry import get_sys_prompt

    _ = mlflow_client
    mlflow.set_experiment("prompt-slash-exp")
    settings = MlflowSettings(
        tracking_uri=mlflow_client.tracking_uri,
        experiment="prompt-slash-exp",
        run_name="slash-run",
    )

    with RunContext(settings, config=run_config) as ctx:
        ctx.log_prompt_version(
            "discont-extract/trigger",
            "decoder",
            get_sys_prompt("discont-extract/trigger"),
        )

        prompt_version = mlflow_client.search_prompt_versions(
            "discont-extract-trigger-decoder"
        )

        assert len(prompt_version) == 1
        assert mlflow.get_run(ctx.run_id).data.params["prompt_version"] == "1"


def test_checkpoint_names_empty_without_directory(
    run_config: RunConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    ctx = RunContext(MlflowSettings(run_id="missing-run"), config=run_config)

    assert ctx._checkpoint_names() == []


def test_artifact_uri_without_location_raises(
    mlflow_client: MlflowClient,
    run_config: RunConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from mlflow.entities import Run, RunData, RunInfo

    _ = mlflow_client
    mlflow.set_experiment("nouri-exp")
    settings = MlflowSettings(
        tracking_uri=mlflow_client.tracking_uri, experiment="nouri-exp"
    )
    with RunContext(settings, config=run_config) as ctx:
        missing = Run(
            run_info=RunInfo(
                run_id="missing",
                experiment_id="0",
                user_id="test",
                status="RUNNING",
                start_time=0,
                end_time=None,
                lifecycle_stage="active",
                artifact_uri=None,
            ),
            run_data=RunData(),
        )
        monkeypatch.setattr(
            ctx._client,
            "get_run",
            lambda _run_id: missing,
        )

        with pytest.raises(RuntimeError, match="no artifact location"):
            _ = ctx.artifact_uri


def test_reattach_without_prior_log_succeeds(
    mlflow_client: MlflowClient, run_config: RunConfig
) -> None:
    run_id = _stored_run(mlflow_client, run_config)
    settings = MlflowSettings(
        tracking_uri=mlflow_client.tracking_uri,
        experiment="experiment",
        run_id=run_id,
    )
    with RunContext(settings, config=run_config) as ctx:
        assert ctx.resumed is True


@pytest.mark.filterwarnings(
    "ignore:'pin_memory' argument is set as true but not supported on MPS.*:UserWarning"
)
def test_evaluate_and_log_saves_test_predictions(
    mlflow_client: MlflowClient,
    run_config: RunConfig,
    bert_tokenizer: PreTrainedTokenizerFast,
    oblique_bert_model: PreTrainedModel,
    oblique_bert_dataset: DataSplits,
) -> None:
    from tempfile import TemporaryDirectory

    from eh30pt.datasets.registry import get_labels
    from eh30pt.modeling.collator import get_collator
    from eh30pt.modeling.metrics import MetricsAccumulator
    from eh30pt.types import EvalRecord

    _ = mlflow_client
    mlflow.set_experiment("eval-preds-exp")
    settings = MlflowSettings(
        tracking_uri=mlflow_client.tracking_uri,
        experiment="eval-preds-exp",
        run_name="eval-preds-run",
    )
    metrics = MetricsAccumulator(bert_tokenizer, "encoder")
    with RunContext(settings, config=run_config) as ctx:
        ctx.evaluate_and_log(
            model=oblique_bert_model,
            tokenizer=bert_tokenizer,
            data=oblique_bert_dataset,
            arch="encoder",
            collate_fn=get_collator(bert_tokenizer, "encoder"),
            metrics_fn=metrics,
            batch_size=8,
            constrain=False,
        )
        run_id = ctx.run_id

    with TemporaryDirectory(prefix="eh30pt-eval-preds-") as tmp:
        downloaded = mlflow.artifacts.download_artifacts(
            run_id=run_id,
            artifact_path="predictions/predictions-00000.parquet",
            dst_path=tmp,
        )
        rows = pl.read_parquet(downloaded).to_dicts()

    records = [EvalRecord(**row) for row in rows]
    samples = list(oblique_bert_dataset.test.samples())

    assert len(records) == len(samples)
    assert {record.prediction for record in records} <= set(get_labels("oblique"))
    for record, sample in zip(records, samples, strict=True):
        assert record.id == sample.id
        assert record.user == sample.user
        assert record.label == sample.label


def test_enter_without_active_run_raises(
    mlflow_client: MlflowClient,
    run_config: RunConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _ = mlflow_client
    mlflow.set_experiment("noactive-exp")
    settings = MlflowSettings(
        tracking_uri=mlflow_client.tracking_uri, experiment="noactive-exp"
    )
    monkeypatch.setattr(mlflow, "active_run", lambda: None)

    with (
        pytest.raises(RuntimeError, match="no active mlflow run"),
        RunContext(settings, config=run_config),
    ):
        pass

    monkeypatch.undo()
    if mlflow.active_run() is not None:
        mlflow.end_run()

"""Tests for MLflow run management and metrics collection."""

from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock

import pytest

from eh30pt.tracking import (
    ConfigMismatchError,
    RunContext,
    _checkpoint_dir,
    _conf_hash_tag,
    _find_checkpoint,
    _hash_config,
    build_run_name,
    collect_runs,
    open_run,
)

if TYPE_CHECKING:
    from pathlib import Path

    from eh30pt.types import FewShotRunConfig


def _fake_search_run(
    run_id: str,
    status: str = "RUNNING",
    run_name: str | None = None,
) -> MagicMock:
    run = MagicMock()
    run.info.run_id = run_id
    run.info.run_name = run_name or f"run-{run_id}"
    run.info.status = status
    return run


def _ctx(client: MagicMock, *, resumed: bool = False) -> RunContext:
    return RunContext(
        client=client,
        experiment="exp",
        run_id="abc123",
        run_name="my-run",
        tracking_uri="sqlite:///unused.db",
        resumed=resumed,
    )


def _mock_backends(
    monkeypatch: pytest.MonkeyPatch,
    client: MagicMock,
) -> MagicMock:
    monkeypatch.setattr(
        "eh30pt.tracking.MlflowClient",
        lambda *_, **__: client,
    )
    fluent = MagicMock()
    monkeypatch.setattr("eh30pt.tracking.mlflow", fluent)
    return fluent


def test_run_name_strips_model_org() -> None:
    name = build_run_name(
        "multinerd", "meta-llama/Llama-3.1-8B-Instruct", "fine-tune", 20000
    )

    assert name == "multinerd-Llama-3.1-8B-Instruct-fine-tune-20000"


def test_run_name_without_org() -> None:
    name = build_run_name("multinerd", "some-model", "fine-tune", 20000)

    assert name == "multinerd-some-model-fine-tune-20000"


def test_run_name_all_samples() -> None:
    name = build_run_name("oblique", "google/flan-t5-xxl", "prompt-tune-random", None)

    assert name == "oblique-flan-t5-xxl-prompt-tune-random-all"


def test_run_name_zero_samples_is_not_all() -> None:
    name = build_run_name("oblique", "google/flan-t5-xxl", "5-shot", 0)

    assert name == "oblique-flan-t5-xxl-5-shot-0"


def test_config_hash_is_deterministic() -> None:
    config: dict[str, Any] = {"lr": 5e-5, "epochs": 5, "seed": None}

    assert _hash_config(config) == _hash_config(dict(config))


def test_config_hash_changes_with_config() -> None:
    assert _hash_config({"lr": 5e-5}) != _hash_config({"lr": 1e-3})


def test_config_hash_ignores_key_order() -> None:
    assert _hash_config({"a": 1, "b": 2}) == _hash_config({"b": 2, "a": 1})


def test_checkpoint_dir_is_keyed_by_run_id(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr("eh30pt.tracking.LOGDIR", tmp_path)

    assert _checkpoint_dir("abc123") == tmp_path / "abc123"


def test_find_checkpoint_returns_none_for_empty_dir(tmp_path: Path) -> None:
    assert _find_checkpoint(tmp_path) is None


def _config(*, batch_size: int = 8) -> FewShotRunConfig:
    return {
        "method": "3-shot",
        "dataset": "multinerd",
        "architecture": "encoder",
        "base_model": "some-model",
        "n_shot": 3,
        "batch_size": batch_size,
        "seed": None,
        "system_prompt": "prompt",
        "train_samples": 0,
        "validation_samples": 0,
        "test_samples": 10,
        "total_parameters": 100,
        "trainable_parameters": 0,
    }


def test_open_run_without_resume_creates_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = MagicMock()
    client.get_experiment_by_name.return_value = MagicMock(experiment_id="1")
    created = _fake_search_run("new-id")
    client.create_run.return_value = created
    fluent = _mock_backends(monkeypatch, client)
    config = _config()

    with open_run(
        "exp", "my-run", "sqlite:///unused.db", resume=False, config=config
    ) as ctx:
        assert ctx.run_id == "new-id"
        assert ctx.resumed is False

    client.search_runs.assert_not_called()
    client.create_run.assert_called_once()
    _, kwargs = client.create_run.call_args
    assert kwargs["tags"] == {_conf_hash_tag: _hash_config(config)}
    assert client.log_batch.call_count == 1
    client.set_terminated.assert_called_once_with("new-id", "FINISHED")
    fluent.start_run.assert_called_once_with(run_id="new-id")
    fluent.end_run.assert_called_once_with("FINISHED")


def test_open_run_creates_missing_experiment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = MagicMock()
    client.get_experiment_by_name.return_value = None
    client.create_experiment.return_value = "9"
    client.create_run.return_value = _fake_search_run("new-id")
    _mock_backends(monkeypatch, client)

    with open_run(
        "exp", "my-run", "sqlite:///unused.db", resume=False, config=_config()
    ):
        pass

    client.create_experiment.assert_called_once_with("exp")
    client.create_run.assert_called_once()
    assert client.create_run.call_args.args[0] == "9"


def test_open_run_resume_reattaches_to_unfinished_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = MagicMock()
    client.get_experiment_by_name.return_value = MagicMock(experiment_id="1")
    client.search_runs.return_value = [_fake_search_run("old-id", status="RUNNING")]
    config = _config()
    stored = MagicMock()
    stored.data.params = {key: str(value) for key, value in config.items()}
    client.get_run.return_value = stored
    fluent = _mock_backends(monkeypatch, client)

    with open_run(
        "exp", "my-run", "sqlite:///unused.db", resume=True, config=config
    ) as ctx:
        assert ctx.run_id == "old-id"
        assert ctx.resumed is True

    client.create_run.assert_not_called()
    fluent.start_run.assert_called_once_with(run_id="old-id")


def test_open_run_resume_starts_new_run_when_previous_finished(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = MagicMock()
    client.get_experiment_by_name.return_value = MagicMock(experiment_id="1")
    client.search_runs.return_value = [_fake_search_run("old-id", status="FINISHED")]
    client.create_run.return_value = _fake_search_run("new-id")
    _mock_backends(monkeypatch, client)

    with open_run(
        "exp", "my-run", "sqlite:///unused.db", resume=True, config=_config()
    ) as ctx:
        assert ctx.run_id == "new-id"
        assert ctx.resumed is False

    client.create_run.assert_called_once()


def test_open_run_resume_starts_new_run_without_previous(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = MagicMock()
    client.get_experiment_by_name.return_value = MagicMock(experiment_id="1")
    client.search_runs.return_value = []
    client.create_run.return_value = _fake_search_run("new-id")
    _mock_backends(monkeypatch, client)

    with open_run(
        "exp", "my-run", "sqlite:///unused.db", resume=True, config=_config()
    ) as ctx:
        assert ctx.resumed is False


def test_open_run_resume_rejects_config_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = MagicMock()
    client.get_experiment_by_name.return_value = MagicMock(experiment_id="1")
    client.search_runs.return_value = [_fake_search_run("old-id", status="RUNNING")]
    stored = MagicMock()
    stored.data.params = {"batch_size": "16"}
    client.get_run.return_value = stored
    fluent = _mock_backends(monkeypatch, client)

    with (
        pytest.raises(ConfigMismatchError, match="batch_size"),
        open_run("exp", "my-run", "sqlite:///unused.db", resume=True, config=_config()),
    ):
        pass

    fluent.start_run.assert_not_called()


def test_open_run_marks_failed_on_error(monkeypatch: pytest.MonkeyPatch) -> None:
    client = MagicMock()
    client.get_experiment_by_name.return_value = MagicMock(experiment_id="1")
    client.create_run.return_value = _fake_search_run("new-id")
    fluent = _mock_backends(monkeypatch, client)

    def _boom() -> None:
        msg = "boom"
        raise RuntimeError(msg)

    with (
        pytest.raises(RuntimeError, match="boom"),
        open_run(
            "exp", "my-run", "sqlite:///unused.db", resume=False, config=_config()
        ),
    ):
        _boom()

    client.set_terminated.assert_called_once_with("new-id", "FAILED")
    fluent.end_run.assert_called_once_with("FAILED")


def _fake_run(
    run_id: str,
    metrics: dict,
    params: dict,
    status: str = "FINISHED",
) -> MagicMock:
    run = MagicMock()
    run.info.run_id = run_id
    run.info.run_name = f"run-{run_id}"
    run.info.status = status
    run.info.start_time = 0
    run.info.end_time = 1
    run.data.metrics = metrics
    run.data.params = params
    return run


def test_collect_metrics_raises_when_experiment_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = MagicMock()
    client.get_experiment_by_name.return_value = None
    monkeypatch.setattr(
        "eh30pt.tracking.MlflowClient",
        lambda *_, **__: client,
    )

    with pytest.raises(RuntimeError, match="missing-experiment"):
        collect_runs("missing-experiment", "sqlite:///unused.db", ())


def test_collect_metrics_builds_dataframe_and_writes_csv(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    client = MagicMock()
    experiment = MagicMock(experiment_id="1")
    client.get_experiment_by_name.return_value = experiment
    client.search_runs.return_value = [
        _fake_run("a", {"accuracy": 0.9}, {"dataset": "boolq"}),
        _fake_run("b", {"accuracy": 0.8}, {"dataset": "wic"}),
    ]
    monkeypatch.setattr(
        "eh30pt.tracking.MlflowClient",
        lambda *_, **__: client,
    )
    monkeypatch.setattr("eh30pt.tracking.LOGDIR", tmp_path)

    df = collect_runs("exp", "sqlite:///unused.db", (), write_csv=True).metrics
    expected_rows = 2

    assert df.shape[0] == expected_rows
    assert set(df["run_id"]) == {"a", "b"}
    assert set(df["accuracy"]) == {0.9, 0.8}
    assert (tmp_path / "metrics" / "exp.csv").exists()


def test_collect_metrics_sums_train_runtime_over_resumes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    client = MagicMock()
    client.get_experiment_by_name.return_value = MagicMock(experiment_id="1")
    client.search_runs.return_value = [_fake_run("a", {"train_runtime": 30.0}, {})]
    client.get_metric_history.return_value = [
        MagicMock(value=100.0),
        MagicMock(value=30.0),
    ]
    monkeypatch.setattr(
        "eh30pt.tracking.MlflowClient",
        lambda *_, **__: client,
    )
    monkeypatch.setattr("eh30pt.tracking.LOGDIR", tmp_path)

    df = collect_runs("exp", "sqlite:///unused.db", ()).metrics

    client.get_metric_history.assert_called_once_with("a", "train_runtime")
    assert df["train_runtime"].to_list() == [130.0]


def test_collect_metrics_skips_history_without_train_runtime(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    client = MagicMock()
    client.get_experiment_by_name.return_value = MagicMock(experiment_id="1")
    client.search_runs.return_value = [_fake_run("a", {"test_runtime": 5.0}, {})]
    monkeypatch.setattr(
        "eh30pt.tracking.MlflowClient",
        lambda *_, **__: client,
    )
    monkeypatch.setattr("eh30pt.tracking.LOGDIR", tmp_path)

    collect_runs("exp", "sqlite:///unused.db", ())

    client.get_metric_history.assert_not_called()


def _fake_metric(step: int, value: float) -> MagicMock:
    metric = MagicMock()
    metric.step = step
    metric.value = value
    metric.timestamp = 1778465526011
    return metric


def test_collect_history_flattens_every_logged_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = MagicMock()
    client.get_experiment_by_name.return_value = MagicMock(experiment_id="1")
    client.search_runs.return_value = [
        _fake_run("a", {}, {"method": "fine-tune", "dataset": "multinerd"}),
    ]
    client.get_metric_history.side_effect = [
        [_fake_metric(100, 0.3), _fake_metric(200, 0.27)],
        [_fake_metric(0, 0.14)],
    ]
    monkeypatch.setattr(
        "eh30pt.tracking.MlflowClient",
        lambda *_, **__: client,
    )

    df = collect_runs("exp", "sqlite:///unused.db", ("loss", "eval_loss")).history

    expected_rows = 3
    assert df.shape[0] == expected_rows
    assert df["metric"].to_list() == ["loss", "loss", "eval_loss"]
    assert df["step"].to_list() == [100, 200, 0]
    assert set(df["method"]) == {"fine-tune"}
    assert set(df["run_id"]) == {"a"}


def test_collect_history_uses_requested_metrics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = MagicMock()
    client.get_experiment_by_name.return_value = MagicMock(experiment_id="1")
    client.search_runs.return_value = [_fake_run("a", {}, {})]
    client.get_metric_history.return_value = [_fake_metric(1, 0.5)]
    monkeypatch.setattr(
        "eh30pt.tracking.MlflowClient",
        lambda *_, **__: client,
    )

    collect_runs("exp", "sqlite:///unused.db", ("eval_f1",))

    client.get_metric_history.assert_called_once_with("a", "eval_f1")


def test_collect_history_writes_csv(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    client = MagicMock()
    client.get_experiment_by_name.return_value = MagicMock(experiment_id="1")
    client.search_runs.return_value = [_fake_run("a", {}, {})]
    client.get_metric_history.return_value = [_fake_metric(1, 0.5)]
    monkeypatch.setattr(
        "eh30pt.tracking.MlflowClient",
        lambda *_, **__: client,
    )
    monkeypatch.setattr("eh30pt.tracking.LOGDIR", tmp_path)

    collect_runs("exp", "sqlite:///unused.db", ("loss",), write_csv=True)

    assert (tmp_path / "metrics" / "exp-history.csv").exists()


def test_collect_runs_searches_runs_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = MagicMock()
    client.get_experiment_by_name.return_value = MagicMock(experiment_id="1")
    client.search_runs.return_value = [_fake_run("a", {}, {})]
    client.get_metric_history.return_value = [_fake_metric(1, 0.5)]
    monkeypatch.setattr(
        "eh30pt.tracking.MlflowClient",
        lambda *_, **__: client,
    )

    collect_runs("exp", "sqlite:///unused.db", ("loss", "eval_loss"))

    client.get_experiment_by_name.assert_called_once_with("exp")
    client.search_runs.assert_called_once()


def test_collect_runs_skips_csv_by_default(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    client = MagicMock()
    experiment = MagicMock(experiment_id="1")
    client.get_experiment_by_name.return_value = experiment
    client.search_runs.return_value = [_fake_run("a", {"accuracy": 0.9}, {})]
    monkeypatch.setattr(
        "eh30pt.tracking.MlflowClient",
        lambda *_, **__: client,
    )
    monkeypatch.setattr("eh30pt.tracking.LOGDIR", tmp_path)

    collect_runs("exp", "sqlite:///unused.db", ())

    assert not (tmp_path / "metrics" / "exp.csv").exists()


def _prompt_version(version: str, template: str) -> MagicMock:
    prompt_version = MagicMock()
    prompt_version.version = version
    prompt_version.template = template
    return prompt_version


def test_log_model_artifacts_saves_and_logs() -> None:
    client = MagicMock()
    model = MagicMock()
    tokenizer = MagicMock()

    _ctx(client).log_model_artifacts(model, tokenizer)

    model.save_pretrained.assert_called_once()
    tokenizer.save_pretrained.assert_called_once()
    client.log_artifacts.assert_called_once()
    _, kwargs = client.log_artifacts.call_args
    assert kwargs["artifact_path"] == "model"


def test_register_prompt_version_skips_unchanged() -> None:
    client = MagicMock()
    prompt_version = _prompt_version("3", "template")
    client.search_prompt_versions.return_value = [prompt_version]

    result = _ctx(client).register_prompt_version("name", "template", "msg")

    assert result is prompt_version
    client.register_prompt.assert_not_called()
    client.link_prompt_version_to_run.assert_called_once_with("abc123", prompt_version)


def test_register_prompt_version_commits_changed() -> None:
    client = MagicMock()
    client.search_prompt_versions.return_value = [_prompt_version("3", "old")]
    prompt_version = _prompt_version("4", "new")
    client.register_prompt.return_value = prompt_version

    result = _ctx(client).register_prompt_version("name", "new", "msg")

    assert result is prompt_version
    client.register_prompt.assert_called_once_with(
        name="name", template="new", commit_message="msg"
    )
    client.link_prompt_version_to_run.assert_called_once_with("abc123", prompt_version)


def _model_version(version: str, tags: dict) -> MagicMock:
    model_version = MagicMock()
    model_version.version = version
    model_version.tags = tags
    return model_version


def test_register_prompt_version_commits_when_missing() -> None:
    from mlflow.exceptions import MlflowException

    client = MagicMock()
    client.search_prompt_versions.side_effect = MlflowException("not found")
    prompt_version = _prompt_version("1", "new")
    client.register_prompt.return_value = prompt_version

    result = _ctx(client).register_prompt_version("name", "new", "msg")

    assert result is prompt_version
    client.register_prompt.assert_called_once_with(
        name="name", template="new", commit_message="msg"
    )
    client.link_prompt_version_to_run.assert_called_once_with("abc123", prompt_version)


def test_register_best_model_registers_first_version() -> None:
    from mlflow.exceptions import MlflowException

    client = MagicMock()
    client.create_model_version.return_value = _model_version("1", {})
    client.get_model_version_by_alias.side_effect = MlflowException("missing")

    version = _ctx(client).register_best_model(
        "name", {"test_f1": 0.8, "test_accuracy": 0.9}
    )

    assert version == "1"
    client.set_model_version_tag.assert_any_call("name", "1", "test_f1", "0.8")
    client.set_registered_model_alias.assert_called_once_with("name", "best", "1")


def test_register_best_model_registers_when_beaten() -> None:
    client = MagicMock()
    client.create_model_version.return_value = _model_version("2", {})
    client.get_model_version_by_alias.return_value = _model_version(
        "1", {"test_f1": "0.5"}
    )

    version = _ctx(client).register_best_model("name", {"test_f1": 0.9})

    assert version == "2"
    client.set_registered_model_alias.assert_called_once_with("name", "best", "2")


def test_register_best_model_skips_when_not_beaten() -> None:
    client = MagicMock()
    client.get_model_version_by_alias.return_value = _model_version(
        "1", {"test_f1": "0.95"}
    )

    version = _ctx(client).register_best_model("name", {"test_f1": 0.9})

    assert version is None
    client.create_model_version.assert_not_called()
    client.set_registered_model_alias.assert_not_called()


def test_register_best_model_skips_without_metric() -> None:
    client = MagicMock()

    version = _ctx(client).register_best_model("name", {})

    assert version is None
    client.create_model_version.assert_not_called()
    client.set_registered_model_alias.assert_not_called()


def test_finish_training_wires_helpers() -> None:
    client = MagicMock()
    client.get_model_version_by_alias.return_value = _model_version("1", {})
    client.create_model_version.return_value = _model_version("2", {})
    model = MagicMock()
    tokenizer = MagicMock()

    _ctx(client).finish_training(
        "name",
        model=model,
        tokenizer=tokenizer,
        test_metrics={"test_f1": 0.8},
    )

    assert client.log_batch.call_count == 1
    client.log_artifacts.assert_called_once()
    client.create_model_version.assert_called_once()

"""Tests for exporting MLflow experiment runs to a dataframe/csv."""

from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import pytest

from eh30pt.scripts.tracking import collect_runs, run_name, start_run

if TYPE_CHECKING:
    from pathlib import Path


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


def test_run_name_strips_model_org() -> None:
    name = run_name("multinerd", "meta-llama/Llama-3.1-8B-Instruct", "fine-tune", 20000)

    assert name == "multinerd/Llama-3.1-8B-Instruct/fine-tune/20000"


def test_run_name_without_org() -> None:
    name = run_name("multinerd", "some-model", "fine-tune", 20000)

    assert name == "multinerd/some-model/fine-tune/20000"


def test_run_name_all_samples() -> None:
    name = run_name("oblique", "google/flan-t5-xxl", "prompt-tune-random", None)

    assert name == "oblique/flan-t5-xxl/prompt-tune-random/all"


def test_run_name_zero_samples_is_not_all() -> None:
    name = run_name("oblique", "google/flan-t5-xxl", "5-shot", 0)

    assert name == "oblique/flan-t5-xxl/5-shot/0"


def test_start_run_without_resume_starts_new_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = MagicMock()
    monkeypatch.setattr("eh30pt.scripts.tracking.mlflow", fake)

    start_run("exp", "my-run", "sqlite:///unused.db", resume=False)

    fake.set_tracking_uri.assert_called_once_with("sqlite:///unused.db")
    fake.set_experiment.assert_called_once_with("exp")
    fake.search_runs.assert_not_called()
    fake.start_run.assert_called_once_with(run_name="my-run")


def test_start_run_reattaches_to_previous_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = MagicMock()
    fake.search_runs.return_value = [_fake_run("a", {}, {}, status="RUNNING")]
    monkeypatch.setattr("eh30pt.scripts.tracking.mlflow", fake)

    start_run("exp", "my-run", "sqlite:///unused.db", resume=True)

    fake.start_run.assert_called_once_with(run_id="a")


def test_start_run_does_not_reattach_to_finished_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = MagicMock()
    fake.search_runs.return_value = [_fake_run("a", {}, {})]
    monkeypatch.setattr("eh30pt.scripts.tracking.mlflow", fake)

    start_run("exp", "my-run", "sqlite:///unused.db", resume=True)

    fake.start_run.assert_called_once_with(run_name="my-run")


def test_start_run_resume_without_previous_run_starts_new_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = MagicMock()
    fake.search_runs.return_value = []
    monkeypatch.setattr("eh30pt.scripts.tracking.mlflow", fake)

    start_run("exp", "my-run", "sqlite:///unused.db", resume=True)

    fake.start_run.assert_called_once_with(run_name="my-run")


def test_collect_metrics_raises_when_experiment_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = MagicMock()
    client.get_experiment_by_name.return_value = None
    monkeypatch.setattr(
        "eh30pt.scripts.tracking.MlflowClient",
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
        "eh30pt.scripts.tracking.MlflowClient",
        lambda *_, **__: client,
    )
    monkeypatch.setattr("eh30pt.scripts.tracking.LOGDIR", tmp_path)

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
        "eh30pt.scripts.tracking.MlflowClient",
        lambda *_, **__: client,
    )
    monkeypatch.setattr("eh30pt.scripts.tracking.LOGDIR", tmp_path)

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
        "eh30pt.scripts.tracking.MlflowClient",
        lambda *_, **__: client,
    )
    monkeypatch.setattr("eh30pt.scripts.tracking.LOGDIR", tmp_path)

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
        "eh30pt.scripts.tracking.MlflowClient",
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
        "eh30pt.scripts.tracking.MlflowClient",
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
        "eh30pt.scripts.tracking.MlflowClient",
        lambda *_, **__: client,
    )
    monkeypatch.setattr("eh30pt.scripts.tracking.LOGDIR", tmp_path)

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
        "eh30pt.scripts.tracking.MlflowClient",
        lambda *_, **__: client,
    )

    collect_runs("exp", "sqlite:///unused.db", ("loss", "eval_loss"))

    client.get_experiment_by_name.assert_called_once_with("exp")
    client.search_runs.assert_called_once_with(["1"], "")


def test_collect_runs_skips_csv_by_default(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    client = MagicMock()
    experiment = MagicMock(experiment_id="1")
    client.get_experiment_by_name.return_value = experiment
    client.search_runs.return_value = [_fake_run("a", {"accuracy": 0.9}, {})]
    monkeypatch.setattr(
        "eh30pt.scripts.tracking.MlflowClient",
        lambda *_, **__: client,
    )
    monkeypatch.setattr("eh30pt.scripts.tracking.LOGDIR", tmp_path)

    collect_runs("exp", "sqlite:///unused.db", ())

    assert not (tmp_path / "metrics" / "exp.csv").exists()

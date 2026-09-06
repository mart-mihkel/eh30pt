"""Export MLflow experiment runs as metrics and per-step history dataframes."""

from typing import TYPE_CHECKING, NamedTuple, TypedDict

import mlflow
from mlflow.tracking import MlflowClient
from polars import DataFrame

from eh30pt.constants import LOGDIR
from eh30pt.logging import logger

if TYPE_CHECKING:
    from mlflow.entities import Run

    from eh30pt.types import DatasetName


class Collected(NamedTuple):
    """Final metrics per run, and the per-step history of the chosen metrics."""

    metrics: DataFrame
    history: DataFrame


class _HistoryRow(TypedDict):
    """One logged value of a per-step metric."""

    run_id: str
    run_name: str | None
    status: str
    dataset: str | None
    base_model: str | None
    method: str | None
    metric: str
    step: int
    timestamp: int
    value: float


def run_name(
    dataset: DatasetName,
    model_path: str,
    method: str,
    train_samples: int | None,
) -> str:
    """Build the default run name, also used as the checkpoint directory."""
    model = model_path.rstrip("/").split("/")[-1]
    samples = "all" if train_samples is None else train_samples
    return f"{dataset}/{model}/{method}/{samples}"


def start_run(
    experiment: str,
    run_name: str,
    mlflow_tracking_uri: str,
    *,
    resume: bool,
) -> None:
    """Start tracking, reattaching to the latest run of `run_name` when resuming."""
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    mlflow.set_experiment(experiment)

    if not resume:
        mlflow.start_run(run_name=run_name)
        return

    previous = mlflow.search_runs(
        experiment_names=[experiment],
        filter_string=f"tags.mlflow.runName = '{run_name}'",
        order_by=["start_time DESC"],
        max_results=1,
        output_format="list",
    )

    if not previous:
        logger.warning("no previous run named '%s', tracking a new one", run_name)
        mlflow.start_run(run_name=run_name)
        return

    info = previous[0].info
    if info.status == "FINISHED":
        logger.warning("run '%s' already finished, tracking a new one", info.run_id)
        mlflow.start_run(run_name=run_name)
        return

    logger.info("reattaching to run '%s'", info.run_id)
    mlflow.start_run(run_id=info.run_id)


def _train_runtime(client: MlflowClient, run_id: str) -> float:
    """
    Sum `train_runtime` over a run's history.

    Each `trainer.train()` call logs the runtime of that call alone, so a run
    resumed from a checkpoint reports only its last segment in `data.metrics`.
    """
    history = client.get_metric_history(run_id, "train_runtime")
    return sum(metric.value for metric in history)


def _search_runs(client: MlflowClient, experiment: str) -> list[Run]:
    """Find every run of `experiment`, erroring if the experiment is missing."""
    logger.info("finding experiment %s", experiment)
    exp = client.get_experiment_by_name(experiment)

    if exp is None:
        msg = f"experiment '{experiment}' not found"
        raise RuntimeError(msg)

    return client.search_runs([exp.experiment_id], "")


def _write_csv(df: DataFrame, name: str) -> None:
    """Write `df` under the metrics log directory, creating it if needed."""
    metricdir = LOGDIR / "metrics"
    path = metricdir / f"{name}.csv"
    metricdir.mkdir(parents=True, exist_ok=True)

    df.write_csv(path)
    logger.info("saved '%s'", path)


def _metrics_frame(client: MlflowClient, runs: list[Run]) -> DataFrame:
    """Build one row per run from its final metric values and params."""
    rows = []
    for run in runs:
        run_data = {
            "run_id": run.info.run_id,
            "run_name": run.info.run_name,
            "status": run.info.status,
            "start_time": run.info.start_time,
            "end_time": run.info.end_time,
        }

        run_data |= run.data.metrics  # ty: ignore[unsupported-operator]
        run_data |= run.data.params

        if "train_runtime" in run.data.metrics:
            run_data["train_runtime"] = _train_runtime(client, run.info.run_id)

        rows.append(run_data)

    df = DataFrame(rows)
    logger.info("found %d runs with %d params", df.shape[0], df.shape[1])
    return df


def _history_rows(client: MlflowClient, run: Run, metric: str) -> list[_HistoryRow]:
    """Read every logged value of `metric` for one run."""
    params = run.data.params
    return [
        _HistoryRow(
            run_id=run.info.run_id,
            run_name=run.info.run_name,
            status=run.info.status,
            dataset=params.get("dataset"),
            base_model=params.get("base_model"),
            method=params.get("method"),
            metric=metric,
            step=logged.step,
            timestamp=logged.timestamp,
            value=logged.value,
        )
        for logged in client.get_metric_history(run.info.run_id, metric)
    ]


def _history_frame(
    client: MlflowClient,
    runs: list[Run],
    metrics: tuple[str, ...],
) -> DataFrame:
    """Build one row per logged value of each metric, across all runs."""
    rows: list[_HistoryRow] = []
    for run in runs:
        for metric in metrics:
            rows.extend(_history_rows(client, run, metric))

    df = DataFrame(rows)
    logger.info("found %d logged values", df.shape[0])
    return df


def collect_runs(
    experiment: str,
    mlflow_tracking_uri: str,
    metrics: tuple[str, ...],
    *,
    write_csv: bool = False,
) -> Collected:
    """Collect final metrics and per-step history for every run of `experiment`."""
    logger.info("connecting to %s", mlflow_tracking_uri)
    client = MlflowClient(tracking_uri=mlflow_tracking_uri)
    runs = _search_runs(client, experiment)

    logger.info("collecting metrics")
    metrics_df = _metrics_frame(client, runs)

    logger.info("collecting history of %s", ", ".join(metrics))
    history_df = _history_frame(client, runs, metrics)

    if write_csv:
        _write_csv(metrics_df, experiment)
        _write_csv(history_df, f"{experiment}-history")

    return Collected(metrics=metrics_df, history=history_df)

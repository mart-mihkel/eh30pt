"""
MLflow experiment tracking.

Runs are the source of truth for resume: a run is reattached by its config
hash, and its checkpoint directory is derived from the run id. The local
``log/`` copy of checkpoints is a fast cache, not an identity.
"""

import hashlib
import json
from contextlib import contextmanager
from dataclasses import dataclass
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Any, NamedTuple, TypedDict

import mlflow
from mlflow.entities import Metric, Param
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient
from polars import DataFrame
from transformers.trainer_utils import get_last_checkpoint

from eh30pt.constants import LOGDIR
from eh30pt.logging import logger

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping
    from pathlib import Path

    from mlflow.entities import Run
    from mlflow.entities.model_registry.prompt_version import PromptVersion
    from peft import PeftModel
    from transformers import PreTrainedModel, PreTrainedTokenizerFast

    from eh30pt.types import DatasetName, RunConfig

_conf_hash_tag = "eh30pt.config_hash"
_articaft_path = "model"


class _Collected(NamedTuple):
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


class ConfigMismatchError(RuntimeError):
    """Raised when resuming a run whose stored config differs from the request."""


def build_run_name(
    dataset: DatasetName,
    model_path: str,
    method: str,
    train_samples: int | None,
) -> str:
    """
    Build the human-readable display name for a run.

    Parameters
    ----------
    dataset : DatasetName
        Dataset the run trains or evaluates on.
    model_path : str
        HuggingFace model id or checkpoint path (only the last segment is used).
    method : str
        Training or evaluation method, e.g. `fine-tune` or `3-shot`.
    train_samples : int | None
        Requested training subset size, shown as `all` when None.

    Returns
    -------
    str
        Display-only run name; carries no resume identity.

    """
    model = model_path.rstrip("/").split("/")[-1]
    samples = "all" if train_samples is None else train_samples
    return f"{dataset}-{model}-{method}-{samples}"


def _hash_config(config: Mapping[str, Any]) -> str:
    """Hash the full training configuration into a short resume key."""
    canonical = json.dumps(config, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


def _checkpoint_dir(run_id: str) -> Path:
    """Return the local checkpoint directory for a run id."""
    return LOGDIR / run_id


def _find_checkpoint(output_dir: Path | str) -> str | None:
    """Find the latest HF checkpoint in `output_dir`, if any exists."""
    checkpoint = get_last_checkpoint(str(output_dir))
    if checkpoint is None:
        logger.info("no checkpoint in '%s', training from scratch", output_dir)
        return None

    logger.info("resuming from checkpoint '%s'", checkpoint)
    return checkpoint


@dataclass
class RunContext:
    """
    An open MLflow run plus its local checkpoint directory.

    All tracking calls go through the client. The fluent mlflow module is
    only used to keep an active run for the HF Trainer autologging callback.

    Attributes
    ----------
    client : MlflowClient
        MLflow client bound to the run's tracking URI.
    experiment : str
        Name of the experiment owning the run.
    run_id : str
        MLflow run id; also keys the local checkpoint directory.
    run_name : str | None
        Human-readable display name.
    tracking_uri : str
        URI of the MLflow tracking server.
    resumed : bool
        Whether the run was reattached to a previous unfinished run.

    """

    client: MlflowClient
    experiment: str
    run_id: str
    run_name: str | None
    tracking_uri: str
    resumed: bool

    @property
    def dir(self) -> Path:
        """Return the local checkpoint directory for this run."""
        return _checkpoint_dir(self.run_id)

    def log_params(self, params: Mapping[str, Any]) -> None:
        """
        Log run params, stringifying values.

        Parameters
        ----------
        params : Mapping[str, Any]
            Param names to values; values are stored via `str`.

        """
        self.client.log_batch(
            self.run_id,
            params=[Param(key, str(value)) for key, value in params.items()],
        )

    def log_metrics(self, metrics: Mapping[str, float], *, step: int = 0) -> None:
        """
        Log run metrics at step.

        Parameters
        ----------
        metrics : Mapping[str, float]
            Metric names to values.
        step : int
            Training step the values belong to.

        """
        import time

        self.client.log_batch(
            self.run_id,
            metrics=[
                Metric(key, value, int(time.time() * 1000), step)
                for key, value in metrics.items()
            ],
        )

    def find_checkpoint(self) -> str | None:
        """Find the latest HF checkpoint of this run, if any exists."""
        return _find_checkpoint(self.dir)

    def log_model_artifacts(
        self,
        model: PreTrainedModel | PeftModel,
        tokenizer: PreTrainedTokenizerFast,
        artifact_path: str = _articaft_path,
    ) -> None:
        """
        Save model and tokenizer, logging them as run artifacts.

        Parameters
        ----------
        model : PreTrainedModel | PeftModel
            Trained model (or PEFT adapter) to save.
        tokenizer : PreTrainedTokenizerFast
            Tokenizer to save alongside the model.
        artifact_path : str
            Artifact directory inside the run.

        """
        with TemporaryDirectory() as tmp:
            model.save_pretrained(tmp)
            tokenizer.save_pretrained(tmp)
            self.client.log_artifacts(self.run_id, tmp, artifact_path=artifact_path)

    def register_prompt_version(
        self,
        name: str,
        template: str,
        commit_message: str,
    ) -> PromptVersion:
        """
        Commit template under name unless the latest version matches.

        Parameters
        ----------
        name : str
            Prompt registry name.
        template : str
            Prompt template text to commit.
        commit_message : str
            Message recorded with the new version.

        Returns
        -------
        PromptVersion
            The reused or newly committed prompt version, linked to the run.

        """
        try:
            versions = self.client.search_prompt_versions(name)
        except MlflowException:
            logger.debug("no prompt named '%s' yet", name)
            versions = []

        if versions and versions[0].template == template:
            prompt_version = versions[0]
            logger.debug(
                "prompt '%s' unchanged at version %s",
                name,
                versions[0].version,
            )
        else:
            logger.info("commiting prompt for '%s'", name)
            prompt_version = self.client.register_prompt(
                name=name,
                template=template,
                commit_message=commit_message,
            )

        self.client.link_prompt_version_to_run(self.run_id, prompt_version)
        return prompt_version

    def register_best_model(
        self,
        name: str,
        test_metrics: Mapping[str, float],
        metric: str = "test_f1",
    ) -> str | None:
        """
        Register the run's model only when it beats the best alias.

        Parameters
        ----------
        name : str
            Registered model name.
        test_metrics : Mapping[str, float]
            Test metrics to tag the version with; must contain `metric`.
        metric : str
            Metric deciding whether the run leads.

        Returns
        -------
        str | None
            The created model version number, or None when the run does not
            improve on the current best.

        """
        try:
            self.client.get_registered_model(name)
        except MlflowException:
            logger.info("creating registered model '%s'", name)
            self.client.create_registered_model(name)

        if metric not in test_metrics:
            logger.warning(
                "metric '%s' not in test metrics, skipping registration",
                metric,
            )

            return None

        current_best: float | None = None
        try:
            current = self.client.get_model_version_by_alias(name, "best")
            if metric in current.tags:
                current_best = float(current.tags[metric])
        except MlflowException:
            logger.debug("no best version of '%s' yet", name)

        if current_best is not None and test_metrics[metric] <= current_best:
            logger.info(
                "not registering '%s': %s=%f does not beat %f",
                name,
                metric,
                test_metrics[metric],
                current_best,
            )

            return None

        version = self.client.create_model_version(
            name,
            source=f"runs:/{self.run_id}/{_articaft_path}",
            run_id=self.run_id,
        ).version

        for key, value in test_metrics.items():
            self.client.set_model_version_tag(name, version, key, str(value))

        logger.info(
            "new best version %s of '%s' (%s=%f)",
            version,
            name,
            metric,
            test_metrics[metric],
        )

        self.client.set_registered_model_alias(name, "best", version)
        return version

    def finish_training(
        self,
        registered_name: str,
        *,
        model: PreTrainedModel | PeftModel,
        tokenizer: PreTrainedTokenizerFast,
        test_metrics: Mapping[str, float],
        metric: str = "test_f1",
    ) -> None:
        """
        Log test metrics and artifacts, registering the model when it leads.

        Parameters
        ----------
        registered_name : str
            Registered model name for the best-model comparison.
        model : PreTrainedModel | PeftModel
            Trained model (or PEFT adapter) to log.
        tokenizer : PreTrainedTokenizerFast
            Tokenizer to log alongside the model.
        test_metrics : Mapping[str, float]
            Test metrics to log and tag the version with.
        metric : str
            Metric deciding whether the run leads.

        """
        self.log_metrics(dict(test_metrics))
        self.log_model_artifacts(model, tokenizer)
        self.register_best_model(registered_name, test_metrics, metric)


def _verify_config(
    client: MlflowClient,
    run_id: str,
    config: Mapping[str, Any],
) -> None:
    """
    Raise ConfigMismatchError when the stored params differ from config.

    Parameters
    ----------
    client : MlflowClient
        MLflow client bound to the run's tracking URI.
    run_id : str
        Id of the run whose stored params are compared.
    config : Mapping[str, Any]
        Requested configuration; every value must match its stored param.

    Raises
    ------
    ConfigMismatchError
        If any configured value differs from the stored param.

    """
    stored = client.get_run(run_id).data.params
    mismatched = [key for key, value in config.items() if stored.get(key) != str(value)]

    if mismatched:
        msg = (
            f"run '{run_id}' was tracked with a different config "
            f"(mismatched keys: {', '.join(sorted(mismatched))}); "
            "pass --no-resume to start a fresh run"
        )

        raise ConfigMismatchError(msg)


def _get_or_create_experiment(client: MlflowClient, experiment: str) -> str:
    exp = client.get_experiment_by_name(experiment)
    if exp is None:
        return client.create_experiment(experiment)

    return exp.experiment_id


def _find_resumable(
    client: MlflowClient,
    experiment_id: str,
    digest: str,
) -> Run | None:
    runs = client.search_runs(
        [experiment_id],
        filter_string=f"tags.{_conf_hash_tag} = '{digest}'",
        order_by=["start_time DESC"],
        max_results=1,
    )

    if not runs:
        return None

    if runs[0].info.status == "FINISHED":
        logger.info(
            "previous run '%s' already finished, starting a new one",
            runs[0].info.run_id,
        )

        return None

    return runs[0]


@contextmanager
def open_run(
    experiment: str,
    run_name: str,
    tracking_uri: str,
    *,
    resume: bool,
    config: RunConfig,
) -> Iterator[RunContext]:
    """
    Open a run keyed by the hash of the config, logging params on creation.

    Parameters
    ----------
    experiment : str
        Name of the experiment owning the run.
    run_name : str
        Human-readable display name for newly created runs.
    tracking_uri : str
        URI of the MLflow tracking server.
    resume : bool
        Reattach to the latest unfinished run with an identical config hash.
    config : RunConfig
        Full run configuration.

    Yields
    ------
    RunContext
        The fresh or reattached run.

    Raises
    ------
    ConfigMismatchError
        If a reattached run's stored params diverge from config.

    """
    client = MlflowClient(tracking_uri)
    experiment_id = _get_or_create_experiment(client, experiment)
    digest = _hash_config(config)

    ctx: RunContext | None = None
    if resume:
        previous = _find_resumable(client, experiment_id, digest)
        if previous is not None:
            run_id = previous.info.run_id
            _verify_config(client, run_id, config)
            logger.info("reattaching to run '%s'", run_id)
            ctx = RunContext(
                client=client,
                experiment=experiment,
                run_id=run_id,
                run_name=previous.info.run_name,
                tracking_uri=tracking_uri,
                resumed=True,
            )

    if ctx is None:
        if resume:
            logger.info("no resumable run found, tracking a new one")

        run = client.create_run(
            experiment_id,
            run_name=run_name,
            tags={_conf_hash_tag: digest},
        )

        ctx = RunContext(
            client=client,
            experiment=experiment,
            run_id=run.info.run_id,
            run_name=run_name,
            tracking_uri=tracking_uri,
            resumed=False,
        )

        ctx.log_params(config)

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.start_run(run_id=ctx.run_id)

    try:
        yield ctx
    except BaseException:
        client.set_terminated(ctx.run_id, "FAILED")
        mlflow.end_run("FAILED")
        raise
    else:
        client.set_terminated(ctx.run_id, "FINISHED")
        mlflow.end_run("FINISHED")


def _train_runtime(client: MlflowClient, run_id: str) -> float:
    """Sum train runtime over a run's history."""
    history = client.get_metric_history(run_id, "train_runtime")
    return sum(metric.value for metric in history)


def _search_runs(
    client: MlflowClient,
    experiment: str,
    page_size: int = 500,
) -> list[Run]:
    """Find every run of experiment, erroring if the experiment is missing."""
    logger.info("finding experiment %s", experiment)
    exp = client.get_experiment_by_name(experiment)

    if exp is None:
        msg = f"experiment '{experiment}' not found"
        raise RuntimeError(msg)

    runs: list[Run] = []
    page_token: str | None = None
    while True:
        page = client.search_runs(
            [exp.experiment_id],
            "",
            max_results=page_size,
            page_token=page_token,
        )

        runs.extend(page)
        page_token = getattr(page, "token", None)
        if not page_token:
            return runs


def _write_csv(df: DataFrame, name: str) -> None:
    """Write dataframe under the metrics log directory, creating it if needed."""
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
    logger.info("found %d runs with %d columns", df.shape[0], df.shape[1])
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
) -> _Collected:
    """
    Collect final metrics and per-step history for every run of `experiment`.

    Parameters
    ----------
    experiment : str
        Name of the experiment to collect.
    mlflow_tracking_uri : str
        URI of the MLflow tracking server.
    metrics : tuple[str, ...]
        Per-step metrics to export the history of.
    write_csv : bool
        Write the frames to `log/metrics/` as CSV files.

    Returns
    -------
    _Collected
        Final metrics per run, and the per-step history of `metrics`.

    """
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

    return _Collected(metrics=metrics_df, history=history_df)

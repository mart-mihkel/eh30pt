"""Open MLflow runs keyed by config hash, with resume support."""

import hashlib
import json
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

import mlflow
from mlflow.tracking import MlflowClient

from eh30pt.logging import logger
from eh30pt.tracking.context import RunContext

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping

    from mlflow.entities import Run

    from eh30pt.types import DatasetName, RunConfig

_conf_hash_tag = "eh30pt.config_hash"


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

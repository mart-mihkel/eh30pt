"""Run configs and verification for run reattachment."""

import json
from pathlib import Path
from typing import TYPE_CHECKING, cast

from mlflow.exceptions import MlflowException

if TYPE_CHECKING:
    from mlflow.tracking import MlflowClient

    from eh30pt.types import Method, RunConfig


class ConfigMismatchError(RuntimeError):
    """Raised when continuing a run whose stored config differs from the request."""


def build_run_name(model: str, method: Method, *labels: str) -> str:
    """Build the human-readable display name for a run."""
    ext = ("gguf", "ggml", "safetensors", "bin", "pt", "pth", "onnx")
    short = model.rstrip("/").split("/")[-1]
    stem, dot, suffix = short.rpartition(".")
    if dot and stem and suffix.lower() in ext:
        short = stem

    return "-".join([short, method, *labels])


def _identity_view(config: RunConfig) -> RunConfig:
    """Return the config view used for params and verification."""
    view = dict(config)
    view.pop("system_prompt", None)
    view.pop("total_parameters", None)
    view.pop("trainable_parameters", None)
    view.pop("eval_steps", None)

    return cast("RunConfig", view)


def _stored_identity(client: MlflowClient, run_id: str) -> dict[str, str]:
    """Return the stored identity, preferring the config artifact."""
    try:
        local = client.download_artifacts(run_id, "config.json")
    except MlflowException:
        params = cast("RunConfig", client.get_run(run_id).data.params)
        return {key: str(value) for key, value in _identity_view(params).items()}

    raw: RunConfig = json.loads(Path(local).read_text())
    return {key: str(value) for key, value in _identity_view(raw).items()}


def _verify_config(
    client: MlflowClient,
    run_id: str,
    config: RunConfig,
) -> None:
    """Raise ConfigMismatchError when the stored identity differs from config."""
    expected = {key: str(value) for key, value in _identity_view(config).items()}
    stored = _stored_identity(client, run_id)
    mismatched = [key for key, value in expected.items() if stored.get(key) != value]

    if mismatched:
        msg = (
            f"run '{run_id}' was tracked with a different config "
            f"(mismatched keys: {', '.join(sorted(mismatched))}); "
            "omit --mlflow-run-id to start a fresh run"
        )

        raise ConfigMismatchError(msg)


def _get_experiment_id(client: MlflowClient, experiment: str) -> str:
    """Return the id of `experiment`, which must already be active."""
    exp = client.get_experiment_by_name(experiment)
    if exp is None:
        msg = f"experiment '{experiment}' not found after activation"
        raise RuntimeError(msg)

    return exp.experiment_id

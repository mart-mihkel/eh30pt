"""Tracking round-trips against an isolated store, without models."""

import json
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, cast

import mlflow
import mlflow.artifacts
import pytest

from eh30pt.settings import MlflowSettings
from eh30pt.tracking.context import RunContext
from eh30pt.tracking.resume import ConfigMismatchError

if TYPE_CHECKING:
    from eh30pt.types import RunConfig

type _ConfigValue = str | int | float | bool


def _config(**overrides: _ConfigValue) -> RunConfig:
    """Build a small run config with identity and volatile fields."""
    base: dict[str, _ConfigValue] = {
        "method": "few-shot",
        "dataset": "oblique",
        "architecture": "encoder",
        "base_model": "tiny-model",
        "n_shot": 0,
        "batch_size": 8,
        "system_prompt": "original prompt",
        "total_parameters": 100,
        "trainable_parameters": 0,
    }
    base.update(overrides)

    return cast("RunConfig", base)


def _boom() -> None:
    """Raise a simulated evaluation failure."""
    msg = "boom"
    raise ValueError(msg)


def test_fresh_run_logs_identity_and_config_artifact(tracking_uri: str) -> None:
    mlflow_settings = MlflowSettings(tracking_uri=tracking_uri, experiment="experiment")
    with RunContext(mlflow_settings, config=_config()) as ctx:
        assert mlflow.active_run() is not None
        assert ctx.resumed is False
        run_id = ctx.run_id

    assert mlflow.active_run() is None

    run = mlflow.get_run(run_id)
    assert run.info.status == "FINISHED"
    assert "system_prompt" not in run.data.params
    assert run.data.params["method"] == "few-shot"

    local = mlflow.artifacts.download_artifacts(
        run_id=run_id,
        artifact_path="config.json",
        tracking_uri=tracking_uri,
    )
    with Path(local).open() as handle:
        stored = json.load(handle)

    assert stored == dict(_config())


def test_reattach_ignores_prompt_and_count_drift(tracking_uri: str) -> None:
    mlflow_settings = MlflowSettings(tracking_uri=tracking_uri, experiment="experiment")
    with RunContext(mlflow_settings, config=_config()) as ctx:
        run_id = ctx.run_id

    resumed_settings = MlflowSettings(
        tracking_uri=tracking_uri, experiment="experiment", run_id=run_id
    )
    with RunContext(
        resumed_settings,
        config=_config(
            system_prompt="edited prompt",
            total_parameters=555,
            trainable_parameters=55,
        ),
    ) as ctx:
        assert ctx.resumed is True
        assert ctx.run_id == run_id

    assert mlflow.get_run(ctx.run_id).info.status == "FINISHED"


def test_identity_change_rejects_reattach(tracking_uri: str) -> None:
    mlflow_settings = MlflowSettings(tracking_uri=tracking_uri, experiment="experiment")
    with RunContext(mlflow_settings, config=_config()) as ctx:
        run_id = ctx.run_id

    with (
        pytest.raises(ConfigMismatchError, match="different config"),
        RunContext(
            MlflowSettings(
                tracking_uri=tracking_uri, experiment="experiment", run_id=run_id
            ),
            config=_config(batch_size=16),
        ),
    ):
        pass


def test_exception_marks_run_failed(tracking_uri: str) -> None:
    mlflow_settings = MlflowSettings(tracking_uri=tracking_uri, experiment="experiment")
    with (
        pytest.raises(ValueError, match="boom"),
        RunContext(mlflow_settings, config=_config()) as ctx,
    ):
        _boom()

    assert mlflow.get_run(ctx.run_id).info.status == "FAILED"
    assert mlflow.active_run() is None


def test_checkpoints_sync_and_restore(tracking_uri: str) -> None:
    mlflow_settings = MlflowSettings(tracking_uri=tracking_uri, experiment="experiment")
    with RunContext(mlflow_settings, config=_config()) as ctx:
        run_id = ctx.run_id
        checkpoint = ctx.dir / "checkpoint-100"
        checkpoint.mkdir(parents=True)
        (checkpoint / "pytorch_model.bin").write_bytes(b"fake weights")
        (checkpoint / "trainer_state.json").write_text("{}")
        ctx.sync_checkpoints()

    shutil.rmtree(ctx.dir)

    with RunContext(
        MlflowSettings(
            tracking_uri=tracking_uri, experiment="experiment", run_id=run_id
        ),
        config=_config(),
    ) as restored:
        assert restored.resumed is True
        assert restored.find_checkpoint() == str(restored.dir / "checkpoint-100")


def test_missing_checkpoints_resolve_to_scratch(tracking_uri: str) -> None:
    mlflow_settings = MlflowSettings(tracking_uri=tracking_uri, experiment="experiment")
    with RunContext(mlflow_settings, config=_config()) as ctx:
        assert (ctx.dir / "run.log").exists()
        assert ctx.find_checkpoint() is None
        ctx.sync_checkpoints()

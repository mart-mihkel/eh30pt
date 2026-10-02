"""Export prediction shards as Label Studio tasks with a labeling config."""

import json
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING

import mlflow
import polars as pl

from eh30pt.labeling.registry import build_labeling
from eh30pt.logging import logger
from eh30pt.settings import MlflowSettings

if TYPE_CHECKING:
    from pathlib import Path

    from eh30pt.types import DatasetName


def _read_shards(run: str) -> pl.DataFrame:
    """Download and concatenate the prediction shards logged by `run`."""
    artifacts = mlflow.artifacts.list_artifacts(
        artifact_path=MlflowSettings.predictions_path,
        run_id=run,
    )

    logger.info("found %d prediction shards", len(artifacts))
    shards: list[pl.DataFrame] = []
    with TemporaryDirectory(prefix="eh30pt-export-") as tmp:
        for artifact in artifacts:
            path = mlflow.artifacts.download_artifacts(
                artifact_path=artifact.path,
                run_id=run,
                dst_path=tmp,
            )

            logger.debug("downloaded shard '%s'", artifact.path)
            shards.append(pl.read_parquet(path))

    return pl.concat(shards) if shards else pl.DataFrame()


def main(
    *,
    settings: MlflowSettings,
    dataset: DatasetName,
    output: Path,
) -> None:
    """Export the run's prediction shards as Label Studio tasks."""
    mlflow.set_tracking_uri(settings.tracking_uri)

    run = settings.run_id
    if run is None:
        msg = "no mlflow run id to export"
        raise ValueError(msg)

    logger.info("exporting '%s' predictions of run '%s'", dataset, run)

    frame = _read_shards(run)
    logger.info("read %d prediction rows", frame.height)

    labeling = build_labeling(dataset, frame, run)

    output.parent.mkdir(parents=True, exist_ok=True)
    tasks = json.dumps(labeling.tasks, ensure_ascii=False, indent=2)
    output.write_text(tasks, encoding="utf-8")
    logger.info("wrote %d tasks to '%s'", len(labeling.tasks), output)

    config_path = output.with_suffix(".xml")
    config_path.write_text(labeling.config, encoding="utf-8")
    logger.info("wrote labeling config to '%s'", config_path)

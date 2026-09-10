"""Concat prediction shards into a Label Studio CSV."""

from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING

import mlflow
import polars as pl

from eh30pt.logging import logger
from eh30pt.settings import MlflowSettings

if TYPE_CHECKING:
    from pathlib import Path


def main(*, settings: MlflowSettings, output: Path) -> None:
    """Concat the run's prediction shards into a Label Studio CSV."""
    mlflow.set_tracking_uri(settings.tracking_uri)

    run = settings.run_id
    if run is None:
        msg = "no mlflow run id to export"
        raise ValueError(msg)

    shards = []
    artifacts = mlflow.artifacts.list_artifacts(
        artifact_path=MlflowSettings.predictions_path,
        run_id=run,
    )

    with TemporaryDirectory(prefix="eh30pt-export-") as tmp:
        for artifact in artifacts:
            path = mlflow.artifacts.download_artifacts(
                artifact_path=artifact.path,
                run_id=run,
                dst_path=tmp,
            )

            shard = pl.read_parquet(path)
            shards.append(shard)

    schema = {
        "phrase": pl.String,
        "reason": pl.String,
        "drug": pl.String,
    }

    dtype = pl.Struct(schema)

    df = (
        pl.concat(shards)
        .with_columns(pl.col("prediction").str.json_decode(dtype))
        .unnest("prediction")
        .filter(
            (pl.col("reason") != "") & (pl.col("phrase") != "") & (pl.col("drug") != "")
        )
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    df.write_csv(output)
    logger.info("wrote %d rows to '%s'", len(df), output)

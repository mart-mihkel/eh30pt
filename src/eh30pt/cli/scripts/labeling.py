"""Create a Label Studio project from the prediction shards of a run."""

from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, cast, get_args

import mlflow
import polars as pl
from mlflow.tracking import MlflowClient
from returns.result import Failure, Success

from eh30pt.constants import LOGDIR
from eh30pt.labeling.extract import span_controls
from eh30pt.labeling.registry import build_labeling
from eh30pt.labeling.studio import StudioClient, annotation_frame
from eh30pt.logging import logger
from eh30pt.settings import MlflowSettings
from eh30pt.tracking.resume import _stored_identity
from eh30pt.types import DatasetName

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from pathlib import Path

    from eh30pt.labeling.studio import ProjectRef
    from eh30pt.labeling.types import Labeling
    from eh30pt.settings import LabelStudioSettings


def read_shards(run: str) -> pl.DataFrame:
    """Download and concatenate the prediction shards logged by `run`."""
    artifacts = mlflow.artifacts.list_artifacts(
        artifact_path=MlflowSettings.predictions_path,
        run_id=run,
    )

    logger.info("found %d prediction shards", len(artifacts))
    shards: list[pl.DataFrame] = []
    with TemporaryDirectory(prefix="eh30pt-create-task-") as tmp:
        for artifact in artifacts:
            path = mlflow.artifacts.download_artifacts(
                artifact_path=artifact.path,
                run_id=run,
                dst_path=tmp,
            )

            logger.debug("downloaded shard '%s'", artifact.path)
            shards.append(pl.read_parquet(path))

    return pl.concat(shards) if shards else pl.DataFrame()


def _dataset_name(identity: Mapping[str, str], run: str) -> DatasetName:
    """Return the dataset of a stored run identity, raising when unusable."""
    names = cast("tuple[DatasetName, ...]", get_args(DatasetName.__value__))
    dataset = identity.get("dataset")
    if dataset is None:
        msg = f"run '{run}' records no dataset, pass --dataset explicitly"
        raise ValueError(msg)

    if dataset not in names:
        msg = (
            f"run '{run}' records unknown dataset '{dataset}', "
            "pass --dataset explicitly"
        )
        raise ValueError(msg)

    return dataset


def read_dataset(settings: MlflowSettings, run: str) -> DatasetName:
    """Return the dataset the run was tracked with."""
    client = MlflowClient(settings.tracking_uri)
    identity = _stored_identity(client, run)
    dataset = _dataset_name(identity, run)

    logger.info("inferred dataset '%s' from run '%s'", dataset, run)
    return dataset


def run_labeling(
    *,
    mlflow_settings: MlflowSettings,
    dataset: DatasetName | None = None,
) -> Labeling:
    """Build the tasks and labeling config of a run's prediction shards."""
    run = mlflow_settings.run_id
    if run is None:
        msg = "no mlflow run id to export"
        raise ValueError(msg)

    mlflow.set_tracking_uri(mlflow_settings.tracking_uri)
    if dataset is None:
        dataset = read_dataset(mlflow_settings, run)

    logger.info("reading '%s' predictions of run '%s'", dataset, run)

    frame = read_shards(run)
    logger.info("read %d prediction rows", frame.height)

    labeling = build_labeling(dataset, frame, run)
    logger.info("built %d tasks", len(labeling.tasks))

    return labeling


def main(
    *,
    settings: LabelStudioSettings,
    mlflow_settings: MlflowSettings,
    dataset: DatasetName | None = None,
    title: str | None = None,
) -> ProjectRef:
    """Create a Label Studio project holding the run's tasks and config."""
    if dataset is None:
        run = mlflow_settings.run_id
        if run is None:
            msg = "no mlflow run id to export"
            raise ValueError(msg)

        dataset = read_dataset(mlflow_settings, run)

    if title is None:
        title = f"eh30pt-{dataset}"
        logger.debug("inferred project title '%s'", title)

    labeling = run_labeling(mlflow_settings=mlflow_settings, dataset=dataset)

    with StudioClient(settings) as studio:
        match studio.create(title, labeling.config):
            case Success(project):
                pass
            case Failure(exc):
                msg = f"could not create project '{title}': {exc}"
                raise RuntimeError(msg) from exc

        match studio.validate(project.id, labeling.config):
            case Success(_):
                pass
            case Failure(exc):
                msg = f"project {project.id} rejected the labeling config: {exc}"
                raise RuntimeError(msg) from exc

        match studio.import_tasks(project.id, labeling.tasks):
            case Success(imported):
                pass
            case Failure(exc):
                msg = f"could not import tasks into project {project.id}: {exc}"
                raise RuntimeError(msg) from exc

    logger.info(
        "imported %d tasks and %d pre-annotations into project %d",
        imported.tasks,
        imported.predictions,
        project.id,
    )

    return project


def export_labels(
    *,
    settings: LabelStudioSettings,
    projects: Sequence[int],
    output: Path | None = None,
) -> Path:
    """
    Write projects' annotations to Parquet, one row per annotation.

    Task ids are unique per server, so rows from every project share one frame
    and the loader still keeps the first annotation per task.
    """
    controls = span_controls()
    frames: list[pl.DataFrame] = []
    with StudioClient(settings) as studio:
        for project in projects:
            match studio.tasks(project):
                case Success(tasks):
                    pass
                case Failure(exc):
                    msg = f"could not read tasks of project {project}: {exc}"
                    raise RuntimeError(msg) from exc

            frame = annotation_frame(tasks, controls)
            logger.info(
                "collected %d annotations from project %d",
                frame.height,
                project,
            )

            frames.append(frame)

    combined = pl.concat(frames, how="diagonal")
    if output is None:
        ids = f"labels-{'-'.join(str(project_id) for project_id in projects)}.parquet"
        output = LOGDIR / ids
        logger.debug("inferred output path '%s'", output)

    output.parent.mkdir(parents=True, exist_ok=True)
    combined.write_parquet(output)
    logger.info("wrote %d annotation rows to '%s'", combined.height, output)

    return output

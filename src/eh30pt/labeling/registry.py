"""Labeling dispatch: map a dataset to the tasks its predictions export as."""

from typing import TYPE_CHECKING

from eh30pt.datasets.registry import get_task
from eh30pt.labeling.export.extract import extraction_df
from eh30pt.labeling.silver.classify import predict_tasks as classify_tasks
from eh30pt.labeling.silver.extract import predict_tasks as extract_tasks
from eh30pt.labeling.tasks.classify import classification_labeling
from eh30pt.labeling.tasks.extract import (
    extraction_controls,
    extraction_labeling,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    import polars as pl

    from eh30pt.labeling.studio import StudioClient, StudioTask
    from eh30pt.labeling.types import Labeling, PredictionCounts
    from eh30pt.types import DatasetName, FewShot, Predictor


async def predict_silver_tasks(
    dataset: DatasetName,
    tasks: Sequence[StudioTask],
    predictor: Predictor[FewShot, str | None],
    studioc: StudioClient,
    *,
    model: str,
    concurrency: int,
) -> PredictionCounts:
    """Attach silver predictions of `model` to the tasks of `dataset`."""
    task = get_task(dataset)

    match task:
        case "extraction":
            return await extract_tasks(
                tasks,
                predictor,
                studioc,
                model=model,
                concurrency=concurrency,
            )
        case "classification":
            return await classify_tasks(
                tasks,
                dataset,
                predictor,
                studioc,
                model=model,
                concurrency=concurrency,
            )
        case _:
            msg = f"dataset '{dataset}' with task '{task}' is not implemented"
            raise NotImplementedError(msg)


def label_controls(dataset: DatasetName) -> tuple[str, ...]:
    """Return the choice controls of the interface that labeled `dataset`."""
    task = get_task(dataset)

    match task:
        case "extraction":
            return extraction_controls()
        case "classification":
            return ("label",)
        case _:
            msg = f"dataset '{dataset}' has task '{task}', which cannot be exported"
            raise NotImplementedError(msg)


def export_frame(
    tasks: Sequence[StudioTask],
    dataset: DatasetName,
    project: int,
) -> pl.DataFrame:
    """Build the annotation frame of the interface that labeled `dataset`."""
    task = get_task(dataset)

    match task:
        case "extraction":
            return extraction_df(tasks, project=project)
        case "classification":
            msg = f"dataset '{dataset}' has no implementation for '{task}'"
            raise NotImplementedError(msg)
        case _:
            msg = f"dataset '{dataset}' has task '{task}', which cannot be exported"
            raise NotImplementedError(msg)


def build_labeling(
    dataset: DatasetName,
    frame: pl.DataFrame,
    model_name: str,
) -> Labeling:
    """Build tasks and a label config matching the task type of `dataset`."""
    task = get_task(dataset)

    match task:
        case "extraction":
            return extraction_labeling(dataset, frame, model_name)
        case "classification":
            return classification_labeling(dataset, frame, model_name)
        case _:
            msg = f"dataset '{dataset}' has task '{task}', which cannot be exported"
            raise NotImplementedError(msg)

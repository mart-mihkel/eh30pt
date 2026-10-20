"""Labeling dispatch: map a dataset to the tasks its predictions export as."""

from typing import TYPE_CHECKING

from eh30pt.datasets.registry import get_task
from eh30pt.labeling.loaders.extract import extraction_df
from eh30pt.labeling.tasks.classify import classification_labeling
from eh30pt.labeling.tasks.extract import extraction_controls, extraction_labeling

if TYPE_CHECKING:
    from collections.abc import Sequence

    import polars as pl

    from eh30pt.labeling.studio import StudioTask
    from eh30pt.labeling.types import Labeling
    from eh30pt.types import DatasetName


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


def build_labeling(dataset: DatasetName, frame: pl.DataFrame, run: str) -> Labeling:
    """Build tasks and a label config matching the task type of `dataset`."""
    task = get_task(dataset)

    match task:
        case "extraction":
            return extraction_labeling(dataset, frame, run)
        case "classification":
            return classification_labeling(dataset, frame, run)
        case _:
            msg = f"dataset '{dataset}' has task '{task}', which cannot be exported"
            raise NotImplementedError(msg)

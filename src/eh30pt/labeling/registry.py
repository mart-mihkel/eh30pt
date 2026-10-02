"""Labeling dispatch: map a dataset to the tasks its predictions export as."""

from typing import TYPE_CHECKING

from eh30pt.datasets.registry import get_task
from eh30pt.labeling.classify import classification_labeling
from eh30pt.labeling.extract import extraction_labeling

if TYPE_CHECKING:
    import polars as pl

    from eh30pt.labeling.types import Labeling
    from eh30pt.types import DatasetName


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

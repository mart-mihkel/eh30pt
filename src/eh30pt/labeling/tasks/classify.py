"""Choice tasks for classification dataset predictions."""

import xml.etree.ElementTree as ET
from typing import TYPE_CHECKING

from eh30pt.datasets.registry import get_labels
from eh30pt.labeling.types import Labeling
from eh30pt.logging import logger

if TYPE_CHECKING:
    import polars as pl

    from eh30pt.labeling.types import ChoiceResult, ChoiceTask, Task, TextData
    from eh30pt.types import DatasetName


def choice_config(labels: tuple[str, ...]) -> str:
    """Build the Label Studio interface offering `labels` as choices."""
    view = ET.Element("View")
    ET.SubElement(view, "Text", name="text", value="$text")

    choices = ET.SubElement(
        view,
        "Choices",
        name="label",
        toName="text",
        choice="single",
        showInLine="true",
    )

    for label in labels:
        ET.SubElement(choices, "Choice", value=label)

    ET.indent(view)
    return ET.tostring(view, encoding="unicode")


def choice_task(user: str, label: str, model_version: str) -> ChoiceTask:
    """Build a task with the predicted label preselected."""
    result: ChoiceResult = {
        "from_name": "label",
        "to_name": "text",
        "type": "choices",
        "readonly": True,
        "hidden": False,
        "value": {"choices": [label]},
    }

    data: TextData = {"text": user}

    return {
        "data": data,
        "predictions": [{"result": [result], "model_version": model_version}],
    }


def classification_labeling(
    dataset: DatasetName,
    frame: pl.DataFrame,
    model_name: str,
) -> Labeling:
    """Build choice tasks from classification predictions, dropping unknown labels."""
    labels = get_labels(dataset)
    config = choice_config(labels)

    tasks: list[Task] = []
    if frame.height == 0:
        return Labeling(tasks=tasks, config=config)

    skipped = 0
    for user, prediction in frame.select("user", "prediction").iter_rows():
        if not isinstance(prediction, str) or prediction not in labels:
            skipped += 1
            continue

        tasks.append(choice_task(user, prediction, model_name))

    if skipped:
        logger.warning(
            "dropped %d rows whose prediction is not a '%s' label",
            skipped,
            dataset,
        )

    return Labeling(tasks=tasks, config=config)

"""Attach silver discontinuation labels to existing Label Studio tasks."""

from typing import TYPE_CHECKING, NamedTuple, cast

from returns.result import Failure, Success
from tqdm import tqdm

from eh30pt.datasets.discont.classify import DiscontClassifier
from eh30pt.datasets.registry import get_labels
from eh30pt.labeling.tasks.extract import extraction_control
from eh30pt.labeling.types import PredictionCounts
from eh30pt.logging import logger
from eh30pt.openai.stream import batched_concurrent, maybe_log_invalid
from eh30pt.types import FewShot

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from eh30pt.labeling.studio import StudioClient, StudioTask
    from eh30pt.labeling.types import ChoiceResult, ExtractionData
    from eh30pt.types import DatasetName, Predictor


class ClassificationControl(NamedTuple):
    """Choice control and interface value of one predicted label."""

    control: str
    choice: str


def classification_controls(dataset: DatasetName) -> dict[str, ClassificationControl]:
    """Map every model label of `dataset` to its extraction control and value."""
    controls: dict[str, ClassificationControl] = {}
    for label in get_labels(dataset):
        choice = DiscontClassifier.et2en.get(label)
        if choice is None:
            msg = (
                f"dataset '{dataset}' is a classification dataset whose labels "
                "the extraction interface does not offer"
            )
            raise NotImplementedError(msg)

        control = extraction_control(choice)
        if control is None:
            msg = (
                f"dataset '{dataset}' predicts '{choice}', but the extraction "
                "interface does not offer it"
            )
            raise NotImplementedError(msg)

        controls[label] = ClassificationControl(control=control, choice=choice)

    return controls


def classification_sample(task: StudioTask) -> FewShot:
    """Build the classifier prompt of one task carrying extracted fields."""
    data = cast("ExtractionData", task["data"])

    user = DiscontClassifier.format_user_prompt(
        data["text"],
        data["phrase"],
        data["drug"],
    )

    return FewShot(user=user, label="", id=task["id"])


def classification_choice(
    prediction: object,
    controls: Mapping[str, ClassificationControl],
) -> ChoiceResult | None:
    """Build the choice result of a predicted label, or None when it is not one."""
    if not isinstance(prediction, str):
        return None

    labelled = controls.get(prediction.strip())
    if labelled is None:
        return None

    return {
        "from_name": labelled.control,
        "to_name": "text",
        "type": "choices",
        "readonly": True,
        "hidden": False,
        "value": {"choices": [labelled.choice]},
    }


async def predict_tasks(
    tasks: Sequence[StudioTask],
    dataset: DatasetName,
    predictor: Predictor[FewShot, str | None],
    client: StudioClient,
    *,
    model: str,
    concurrency: int,
) -> PredictionCounts:
    """Classify each task and attach the predicted label to its task."""
    controls = classification_controls(dataset)
    samples = [classification_sample(task) for task in tasks]

    predicted = 0
    invalid = 0
    with tqdm(desc="Predicting", total=len(samples)) as bar:
        async for window, results in batched_concurrent(
            iter(samples),
            predictor,
            concurrency,
        ):
            for sample, prediction in zip(window, results, strict=True):
                task_id = sample.id
                if task_id is None:
                    msg = "silver label samples must carry task ids"
                    raise ValueError(msg)

                result = classification_choice(prediction, controls)
                if result is None:
                    invalid += 1
                    logger.warning("invalid prediction for task %d", task_id)
                    continue

                match client.predict(task_id, [result], model):
                    case Success(_):
                        predicted += 1
                    case Failure(exc):
                        msg = f"could not predict task {task_id}: {exc}"
                        raise RuntimeError(msg) from exc

            bar.update(len(window))

    maybe_log_invalid(invalid, len(samples))
    return PredictionCounts(predicted=predicted, invalid=invalid)

"""Attach silver extraction predictions to existing Label Studio tasks."""

import json
from typing import TYPE_CHECKING

from returns.result import Failure, Success
from tqdm import tqdm

from eh30pt.labeling.tasks.extract import extraction_results
from eh30pt.labeling.types import PredictionCounts
from eh30pt.logging import logger
from eh30pt.openai.stream import batched_concurrent, maybe_log_invalid
from eh30pt.types import FewShot

if TYPE_CHECKING:
    from collections.abc import Sequence

    from eh30pt.labeling.studio import StudioClient, StudioTask
    from eh30pt.labeling.types import SpanResult
    from eh30pt.types import Predictor


def extraction_fields(prediction: object) -> tuple[str, str, str] | None:
    """Parse a prediction into phrase, reason and drug, or None when invalid."""
    match prediction:
        case str():
            try:
                payload: object = json.loads(prediction)
            except json.JSONDecodeError:
                return None
        case _:
            payload = prediction

    if not isinstance(payload, dict):
        return None

    phrase = payload.get("phrase")
    reason = payload.get("reason")
    drug = payload.get("drug")
    if (
        not isinstance(phrase, str)
        or not isinstance(reason, str)
        or not isinstance(drug, str)
    ):
        return None

    return phrase, reason, drug


def extraction_spans(text: str, prediction: object) -> list[SpanResult] | None:
    """Build the spans of a prediction, or None when its fields are unusable."""
    fields = extraction_fields(prediction)
    if fields is None:
        return None

    return extraction_results(text, *fields, readonly=True)


async def predict_tasks(
    tasks: Sequence[StudioTask],
    predictor: Predictor[FewShot, str | None],
    client: StudioClient,
    *,
    model: str,
    concurrency: int,
) -> PredictionCounts:
    """Predict each task's text and attach the located spans to it."""
    samples = [
        FewShot(user=task["data"]["text"], label="", id=task["id"]) for task in tasks
    ]

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

                result = extraction_spans(sample.user, prediction)
                if result is None:
                    invalid += 1
                    logger.warning("invalid prediction for task %d", task_id)
                    continue

                match client.predict(task_id, result, model):
                    case Success(_):
                        predicted += 1
                    case Failure(exc):
                        msg = f"could not predict task {task_id}: {exc}"
                        raise RuntimeError(msg) from exc

            bar.update(len(window))

    maybe_log_invalid(invalid, len(samples))
    return PredictionCounts(predicted=predicted, invalid=invalid)

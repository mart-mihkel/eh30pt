"""Label Studio task payload shapes."""

from typing import NamedTuple, TypedDict


class TextData(TypedDict):
    """Label Studio task input holding the text to label."""

    text: str


class ExtractionData(TextData):
    """Label Studio task input showing the text with its extracted fields."""

    phrase: str
    reason: str
    drug: str


class SpanValue(TypedDict):
    """Character offsets and label of the highlighted phrase."""

    start: int
    end: int
    text: str
    labels: list[str]


class SpanResult(TypedDict):
    """Label Studio span pre-annotation pointing at a substring."""

    from_name: str
    to_name: str
    type: str
    readonly: bool
    hidden: bool
    value: SpanValue


class ChoiceValue(TypedDict):
    """Choice labels selected by a classification pre-annotation."""

    choices: list[str]


class ChoiceResult(TypedDict):
    """Label Studio choice pre-annotation carrying the predicted label."""

    from_name: str
    to_name: str
    type: str
    readonly: bool
    hidden: bool
    value: ChoiceValue


class SpanPrediction(TypedDict):
    """Label Studio pre-annotation holding the phrase highlight."""

    result: list[SpanResult]
    model_version: str


class ChoicePrediction(TypedDict):
    """Label Studio pre-annotation holding the predicted choice."""

    result: list[ChoiceResult]
    model_version: str


class ExtractionTask(TypedDict):
    """Label Studio span task for one extracted phrase."""

    data: ExtractionData
    predictions: list[SpanPrediction]


class ChoiceTask(TypedDict):
    """Label Studio text-classification task for one predicted label."""

    data: TextData
    predictions: list[ChoicePrediction]


type Task = ExtractionTask | ChoiceTask


class Labeling(NamedTuple):
    """Tasks together with the label config their pre-annotations reference."""

    tasks: list[Task]
    config: str


class PredictionCounts(NamedTuple):
    """Outcome of one silver prediction pass over the tasks."""

    predicted: int
    invalid: int

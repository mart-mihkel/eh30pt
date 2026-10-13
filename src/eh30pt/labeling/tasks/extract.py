"""Span tasks for extraction dataset predictions."""

import xml.etree.ElementTree as ET
from typing import TYPE_CHECKING, Final, NamedTuple

import polars as pl

from eh30pt.datasets.registry import get_extract_fields
from eh30pt.labeling.types import Labeling
from eh30pt.logging import logger

if TYPE_CHECKING:
    from eh30pt.labeling.types import (
        ExtractionData,
        ExtractionTask,
        SpanResult,
        SpanValue,
        Task,
    )
    from eh30pt.types import DatasetName


class _Choice(NamedTuple):
    """One selectable label and the hint shown next to it."""

    value: str
    hint: str


class _ChoiceBox(NamedTuple):
    """A headed block of choices annotators pick from together."""

    name: str
    header: str
    choices: tuple[_Choice, ...]


_box_class: Final = "eh30pt-box"
_title_class: Final = "eh30pt-title"


def _locate_span(text: str, span: str) -> tuple[int, int] | None:
    """Return character offsets of span in text, or None when absent."""
    if not span:
        return None

    start = text.find(span)
    if start < 0:
        return None

    return start, start + len(span)


def _span_result(label: str, text: str, span: tuple[int, int]) -> SpanResult:
    """Build a read-only span pre-annotation for a located substring."""
    start, end = span
    val: SpanValue = {
        "start": start,
        "end": end,
        "text": text[start:end],
        "labels": [label],
    }

    return {
        "from_name": "span",
        "to_name": "text",
        "type": "labels",
        "readonly": True,
        "hidden": False,
        "value": val,
    }


def _decoded(frame: pl.DataFrame) -> pl.DataFrame:
    """Decode predictions to phrase, reason, and drug columns."""
    if frame.schema["prediction"] != pl.String:
        return frame.unnest("prediction")

    schema = {
        "phrase": pl.String,
        "reason": pl.String,
        "drug": pl.String,
    }

    try:
        col = pl.col("prediction").str.json_decode(pl.Struct(schema))
        return frame.with_columns(col).unnest("prediction")
    except pl.exceptions.ComputeError as exc:
        msg = "predictions are not extraction JSON"
        raise ValueError(msg) from exc


def _choices(row: ET.Element, box: _ChoiceBox) -> None:
    """Append a box as its own flex column of the row."""
    column = ET.SubElement(
        row,
        "View",
        className=_box_class,
        style="flex: 1; min-width: 240px",
    )

    title = ET.SubElement(column, "View", className=_title_class)
    ET.SubElement(title, "Header", value=box.header)

    control = ET.SubElement(
        column,
        "Choices",
        name=box.name,
        toName="text",
        choice="multiple",
        showInLine="false",
    )

    for choice in box.choices:
        ET.SubElement(control, "Choice", value=choice.value, hint=choice.hint)


def _boxes() -> tuple[_ChoiceBox, ...]:
    """Return the choice boxes of the span interface in rendering order."""
    discont_choices = (
        _Choice(
            "Treatment success",
            (
                "Treatment stopped because it worked, was completed, "
                "or the patient improved enough."
            ),
        ),
        _Choice(
            "Treatment inefficacy",
            "Treatment stopped because it was seen as ineffective.",
        ),
        _Choice(
            "Contraindication/unsafe co-usage",
            (
                "Treatment stopped because a condition or risk factor "
                "made continuing it unsafe or inappropriate."
            ),
        ),
        _Choice(
            "Adverse reactions",
            (
                "Treatment stopped because of adverse side effects, "
                "allergic reactions, or negative interactions with "
                "other medications."
            ),
        ),
        _Choice("Other", "Treatment stopped for other or unknown reasons."),
    )

    other_choices = (
        _Choice(
            "Adverse reaction without discontinuation",
            (
                "Adverse side effects, allergic reactions, or negative "
                "interactions where treatment continued or was not the "
                "reason it stopped."
            ),
        ),
        _Choice(
            "Dosage change",
            ("Dose was increased or decreased. No justification or cause is needed."),
        ),
        _Choice(
            "Other non-discontinuation",
            (
                "An event that is neither a discontinuation nor any of "
                "the reasons listed here."
            ),
        ),
    )

    meta_choices = (
        _Choice(
            "Incomplete phrase",
            (
                "Phrase too short to decide alone but decidable when "
                "expanded; annotate the expanded phrase. "
                "Use only when obvious."
            ),
        ),
        _Choice(
            "Incorrect phrase",
            (
                "Extracted phrase is noise or a false positive; "
                "any real description is elsewhere. "
                "Use only when obvious."
            ),
        ),
    )

    return (
        _ChoiceBox(
            name="discont",
            header="Drug discontinuation",
            choices=discont_choices,
        ),
        _ChoiceBox(
            name="other",
            header="Not discontinuation",
            choices=other_choices,
        ),
        _ChoiceBox(
            name="meta",
            header="Meta",
            choices=meta_choices,
        ),
    )


def extraction_controls() -> tuple[str, ...]:
    """Return the choice control names of the extract interface."""
    return tuple(box.name for box in _boxes())


def extraction_config() -> str:
    """Build the Label Studio interface for extraction labeling."""
    view = ET.Element("View")
    ET.SubElement(view, "Text", name="text", value="$text")

    style = ET.SubElement(view, "Style")
    style.text = f"""
        .{_title_class} {{
            border: 1px solid transparent;
            border-radius: 3px;
            padding: 2px 8px;
            margin-bottom: 4px;
        }}

        .{_box_class}:has(input:checked) .{_title_class} {{
            border-color: #1f9d55;
        }}
    """

    row = ET.SubElement(
        view,
        "View",
        style="display: flex; flex-wrap: wrap; gap: 16px; align-items: flex-start",
    )

    for box in _boxes():
        _choices(row, box)

    hidden = ET.SubElement(view, "View", style="display: none")
    spans = ET.SubElement(hidden, "Labels", name="span", toName="text")
    ET.SubElement(spans, "Label", value="phrase", background="#ffe08a")
    ET.SubElement(spans, "Label", value="reason", background="#a9c6ff")
    ET.SubElement(spans, "Label", value="drug", background="#a8e6a3")

    ET.indent(view)
    return ET.tostring(view, encoding="unicode")


def extraction_task(
    user: str,
    phrase: str,
    reason: str,
    drug: str,
    run_id: str,
) -> ExtractionTask | None:
    """Build a task with all three spans highlighted, or None without them."""
    phrase_span = _locate_span(user, phrase)
    if phrase_span is None:
        return None

    reason_span = _locate_span(user, reason)
    if reason_span is None:
        return None

    drug_span = _locate_span(user, drug)
    if drug_span is None:
        return None

    results = [
        _span_result("phrase", user, phrase_span),
        _span_result("reason", user, reason_span),
        _span_result("drug", user, drug_span),
    ]

    data: ExtractionData = {
        "text": user,
        "phrase": phrase,
        "reason": reason,
        "drug": drug,
    }

    return {
        "data": data,
        "predictions": [{"result": results, "model_version": run_id}],
    }


def _assert_span_fields(dataset: DatasetName) -> None:
    """
    Raise when an extraction dataset does not match the exported span fields.

    The span label config highlights a phrase, a reason and a drug, so only
    a dataset extracting exactly those three fields can be exported as span
    tasks.
    """
    exported: tuple[str, ...] = ("phrase", "reason", "drug")
    declared = get_extract_fields(dataset)
    if set(declared) != set(exported):
        msg = (
            f"dataset '{dataset}' extracts {', '.join(declared)}, "
            f"the span export covers {', '.join(exported)}"
        )
        raise NotImplementedError(msg)


def extraction_labeling(
    dataset: DatasetName,
    df: pl.DataFrame,
    run: str,
) -> Labeling:
    """Build span tasks from extraction predictions, dropping unusable rows."""
    _assert_span_fields(dataset)

    tasks: list[Task] = []
    if df.height == 0:
        return Labeling(tasks=tasks, config=extraction_config())

    df = _decoded(df)
    before = df.height
    df = df.filter(
        (pl.col("phrase") != "") & (pl.col("reason") != "") & (pl.col("drug") != "")
    )

    if dropped := before - df.height:
        logger.info("dropped %d rows with empty fields", dropped)

    before = df.height
    df = df.unique("phrase", maintain_order=True)
    if dupes := before - df.height:
        logger.warning("dropped %d duplicate phrases", dupes)

    skipped = 0
    rows = df.select("user", "phrase", "reason", "drug")
    for user, phrase, reason, drug in rows.iter_rows():
        task = extraction_task(user, phrase, reason, drug, run)
        if task is None:
            skipped += 1
            continue

        tasks.append(task)

    if skipped:
        logger.warning(
            "dropped %d rows without a locatable phrase, reason or drug", skipped
        )

    return Labeling(tasks=tasks, config=extraction_config())

"""Span tasks for extraction dataset predictions."""

import xml.etree.ElementTree as ET
from typing import TYPE_CHECKING

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


def span_config() -> str:
    """Build the Label Studio interface for highlighting extracted spans."""
    view = ET.Element("View")
    ET.SubElement(view, "Text", name="text", value="$text")

    ET.SubElement(view, "Header", value="Esimese taseme märgendid")
    first_level: tuple[tuple[str, str], ...] = (
        (
            "Adverse reactions (ADR)",
            (
                "Adverse side effects, allergic reactions, or negative "
                "interactions. May co-occur with DD or DC. "
                "Alone does not imply discontinuation."
            ),
        ),
        (
            "Drug discontinuation (DD)",
            (
                "Interruption of at least 14 days; shorter breaks are not DD. "
                "May co-occur with ADR. Subtype is optional."
            ),
        ),
        (
            "Dosage change (DC)",
            (
                "Increase or decrease of dosage. No justification or cause "
                "is needed. May co-occur with ADR."
            ),
        ),
        (
            "Incorrect phrase",
            (
                "Extracted phrase is noise or a false positive; "
                "any real description is elsewhere. Use only when obvious."
            ),
        ),
    )
    choices = ET.SubElement(
        view,
        "Choices",
        name="label",
        toName="text",
        choice="multiple",
        showInLine="true",
    )
    for value, hint in first_level:
        ET.SubElement(choices, "Choice", value=value, hint=hint)

    ET.SubElement(view, "Header", value="Drug discontinuation alamliigid")
    subtypes: tuple[tuple[str, str], ...] = (
        (
            "Treatment success",
            "Discontinuation due to successful completion or sufficient improvement.",
        ),
        (
            "Treatment inefficacy",
            (
                "Discontinuation due to perceived ineffectiveness "
                "or loss of belief in efficacy."
            ),
        ),
        (
            "Contraindication",
            (
                "Discontinuation due to a condition or risk factor making "
                "continued use unsafe or inappropriate."
            ),
        ),
    )
    sub_choices = ET.SubElement(
        view,
        "Choices",
        name="dd_subtype",
        toName="text",
        choice="multiple",
        showInLine="true",
    )
    for value, hint in subtypes:
        ET.SubElement(sub_choices, "Choice", value=value, hint=hint)

    ET.SubElement(view, "Header", value="Metamärgendused fraasi kvaliteedi kohta")
    quality: tuple[tuple[str, str], ...] = (
        (
            "Incomplete phrase",
            (
                "Phrase too short to decide alone but decidable when expanded; "
                "annotate the expanded phrase. Use only when obvious."
            ),
        ),
        (
            "Complex fact",
            (
                "Phrase is relevant but the decision requires the whole text. "
                "Use only when obvious."
            ),
        ),
        (
            "Incorrect phrase",
            (
                "Extracted phrase is noise or a false positive; "
                "any real description is elsewhere. Use only when obvious."
            ),
        ),
    )
    quality_choices = ET.SubElement(
        view,
        "Choices",
        name="quality",
        toName="text",
        choice="multiple",
        showInLine="true",
    )
    for value, hint in quality:
        ET.SubElement(quality_choices, "Choice", value=value, hint=hint)

    hidden = ET.SubElement(view, "View", style="display: none")
    spans = ET.SubElement(hidden, "Labels", name="span", toName="text")
    ET.SubElement(spans, "Label", value="phrase", background="#ffe08a")
    ET.SubElement(spans, "Label", value="drug", background="#a8e6a3")

    ET.indent(view)
    return ET.tostring(view, encoding="unicode")


def span_task(
    user: str,
    phrase: str,
    reason: str,
    drug: str,
    run_id: str,
) -> ExtractionTask | None:
    """Build a task with phrase and drug highlighted, or None without both."""
    phrase_span = _locate_span(user, phrase)
    if phrase_span is None:
        return None

    drug_span = _locate_span(user, drug)
    if drug_span is None:
        return None

    results = [
        _span_result("phrase", user, phrase_span),
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

    The span label config highlights a phrase and a drug, and every task shows
    the reason next to them, so only a dataset extracting exactly those three
    fields can be exported as span tasks.
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
        return Labeling(tasks=tasks, config=span_config())

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
        task = span_task(user, phrase, reason, drug, run)
        if task is None:
            skipped += 1
            continue

        tasks.append(task)

    if skipped:
        logger.warning("dropped %d rows without a locatable phrase or drug", skipped)

    return Labeling(tasks=tasks, config=span_config())

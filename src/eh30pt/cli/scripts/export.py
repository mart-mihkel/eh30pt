"""Export prediction shards as Label Studio tasks with a labeling config."""

import json
import xml.etree.ElementTree as ET
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, TypedDict

import mlflow
import polars as pl

from eh30pt.logging import logger
from eh30pt.settings import MlflowSettings

if TYPE_CHECKING:
    from pathlib import Path


class TaskData(TypedDict):
    """Label Studio task input showing the text with its extracted fields."""

    text: str
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


class TaskPrediction(TypedDict):
    """Label Studio pre-annotation holding the phrase highlight."""

    result: list[SpanResult]
    model_version: str


class ClassificationTask(TypedDict):
    """Label Studio text-classification task for one extracted phrase."""

    data: TaskData
    predictions: list[TaskPrediction]


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


def _extracted(frame: pl.DataFrame) -> pl.DataFrame:
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


def _label_config() -> str:
    """Build the Label Studio interface matching the exported tasks."""
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


def _to_task(
    user: str,
    phrase: str,
    reason: str,
    drug: str,
    run_id: str,
) -> ClassificationTask | None:
    """Build a task with phrase and drug highlighted, or None without a phrase."""
    phrase_span = _locate_span(user, phrase)
    if phrase_span is None:
        return None

    results = [_span_result("phrase", user, phrase_span)]

    drug_span = _locate_span(user, drug)
    if drug_span is not None:
        results.append(_span_result("drug", user, drug_span))

    data: TaskData = {"text": user, "phrase": phrase, "reason": reason, "drug": drug}

    return {
        "data": data,
        "predictions": [{"result": results, "model_version": run_id}],
    }


def main(*, settings: MlflowSettings, output: Path) -> None:
    """Export the run's prediction shards as Label Studio classification tasks."""
    mlflow.set_tracking_uri(settings.tracking_uri)

    run = settings.run_id
    if run is None:
        msg = "no mlflow run id to export"
        raise ValueError(msg)

    logger.info("exporting predictions of run '%s'", run)

    shards: list[pl.DataFrame] = []
    artifacts = mlflow.artifacts.list_artifacts(
        artifact_path=MlflowSettings.predictions_path,
        run_id=run,
    )

    logger.info("found %d prediction shards", len(artifacts))
    with TemporaryDirectory(prefix="eh30pt-export-") as tmp:
        for artifact in artifacts:
            path = mlflow.artifacts.download_artifacts(
                artifact_path=artifact.path,
                run_id=run,
                dst_path=tmp,
            )

            logger.debug("downloaded shard '%s'", artifact.path)
            shards.append(pl.read_parquet(path))

    skipped = 0
    tasks: list[ClassificationTask] = []
    df = pl.concat(shards) if shards else pl.DataFrame()
    logger.info("read %d prediction rows", df.height)
    if df.height > 0:
        df = _extracted(df).filter(
            (pl.col("phrase") != "") & (pl.col("reason") != "") & (pl.col("drug") != "")
        )

        if dropped := df.height - df.height:
            logger.info("dropped %d rows with empty fields", dropped)

        deduped = df.unique("phrase", maintain_order=True)
        if dupes := df.height - deduped.height:
            logger.info("dropped %d duplicate phrases", dupes)

        rows = deduped.select("user", "phrase", "reason", "drug")
        for user, phrase, reason, drug in rows.iter_rows():
            task = _to_task(user, phrase, reason, drug, run)
            if task is None:
                skipped += 1
                continue

            tasks.append(task)

    if skipped:
        logger.warning("skipped %d rows without a locatable phrase", skipped)

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(tasks, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("wrote %d tasks to '%s'", len(tasks), output)

    config_path = output.with_suffix(".xml")
    config_path.write_text(_label_config(), encoding="utf-8")
    logger.info("wrote labeling config to '%s'", config_path)

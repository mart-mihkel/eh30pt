"""Turning Label Studio annotations into training rows, split three ways."""

from typing import TYPE_CHECKING, NamedTuple, TypedDict, cast, get_args

import pandera.polars as pa
import polars as pl

from eh30pt.logging import logger
from eh30pt.types import DiscontClassifierLabel

if TYPE_CHECKING:
    from collections.abc import Sequence

    from pandera.typing.polars import DataFrame

    from eh30pt.datasets.discontinuation.classify import DiscontClassifierRow
    from eh30pt.labeling.studio import StudioTask
    from eh30pt.labeling.types import ChoiceValue


class _DataSplits(NamedTuple):
    """The row lists the labeled rows are carved into."""

    train: list[DiscontClassifierRow]
    eval: list[DiscontClassifierRow]
    test: list[DiscontClassifierRow]


class _ExtractionData(TypedDict):
    text: str
    phrase: str
    reason: str
    drug: str


class ExtractionFrameSchema(pa.DataFrameModel):
    """One row per annotated choice, labeled with classifier reasons."""

    project_id: pl.Int64
    task_id: pl.Int64
    annotation_id: pl.Int64
    result_id: pl.String
    annotator: pl.Int64
    from_name: pl.String
    text: pl.String
    phrase: pl.String
    reason: pl.String
    drug: pl.String
    label: pl.String = pa.Field(isin=get_args(DiscontClassifierLabel.__value__))


def extraction_df(
    tasks: Sequence[StudioTask],
    project: int,
) -> DataFrame[ExtractionFrameSchema]:
    """Build one row per annotated choice, ignoring the meta phrase flags."""
    rows = []
    skipped = 0
    for task in tasks:
        tid = task["id"]
        data = cast("_ExtractionData", task["data"])

        for annt in task["annotations"]:
            results = [res for res in annt["result"] if res["type"] == "choices"]
            if len(results) == 0:
                logger.warning("missing results for task %d, skipping", tid)
                skipped += 1
                continue

            n_res = len(results)
            n_meta = len([res for res in results if res["from_name"] == "meta"])

            res = results[0]
            val = cast("ChoiceValue", res["value"])
            choices = val["choices"]
            choice = choices[0]

            if n_res == n_meta:
                choice = "Other non-discontinuation"
                logger.warning(
                    "only meta annotation for task %d, defaulting to '%s'",
                    tid,
                    choice,
                )
            elif n_res - n_meta > 1:
                logger.warning(
                    "multiple results for task %d, defaulting to first",
                    tid,
                )
            elif len(choices) > 1:
                logger.warning(
                    "multiple choices for task %d: %s, defaulting to first",
                    tid,
                    choices,
                )

            row = {
                "project_id": project,
                "task_id": tid,
                "annotation_id": annt["id"],
                "result_id": res["id"],
                "annotator": annt["completed_by"],
                "from_name": res["from_name"],
                "text": data["text"],
                "phrase": data["phrase"],
                "reason": data["reason"],
                "drug": data["drug"],
                "label": choice,
            }

            rows.append(row)

    if skipped > 0:
        logger.warning(
            "skipped %d missing annotations from project %d",
            skipped,
            project,
        )

    return ExtractionFrameSchema.validate(pl.DataFrame(rows))


def stratify(
    df: DataFrame[ExtractionFrameSchema],
    *,
    on: str,
    ratios: tuple[float, float, float],
    seed: int,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Split rows into train, eval and test, keeping every group proportional."""
    if round(sum(ratios), 6) != 1.0:
        msg = f"split ratios must sum to one, got {ratios}"
        raise ValueError(msg)

    shuffled = df.sample(fraction=1.0, shuffle=True, seed=seed)
    groups = shuffled.partition_by(on, maintain_order=True, as_dict=True)

    train: list[pl.DataFrame] = []
    eval_: list[pl.DataFrame] = []
    test: list[pl.DataFrame] = []
    overflow = 0

    for group in groups.values():
        size = group.height

        if size < len(ratios):
            target = overflow % len(ratios)
            overflow += 1
            empty = group.head(0)
            parts = [empty, empty, empty]
            parts[target] = group
        else:
            n_train = round(size * ratios[0])
            n_eval = round(size * ratios[1])
            parts = [
                group.head(n_train),
                group.slice(n_train, n_eval),
                group.slice(n_train + n_eval),
            ]

        train.append(parts[0])
        eval_.append(parts[1])
        test.append(parts[2])

    splits = (
        pl.concat(train, how="vertical").sample(fraction=1.0, shuffle=True, seed=seed),
        pl.concat(eval_, how="vertical").sample(fraction=1.0, shuffle=True, seed=seed),
        pl.concat(test, how="vertical").sample(fraction=1.0, shuffle=True, seed=seed),
    )

    logger.debug(
        "stratified %d rows into train=%d eval=%d test=%d",
        df.height,
        splits[0].height,
        splits[1].height,
        splits[2].height,
    )

    return splits

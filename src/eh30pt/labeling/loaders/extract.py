"""Turning Label Studio annotations into training rows, split three ways."""

from collections import Counter
from typing import TYPE_CHECKING, NamedTuple, TypedDict

from returns.result import Failure, Success
from sklearn.model_selection import ShuffleSplit, StratifiedShuffleSplit

from eh30pt.labeling.studio import StudioClient, group_choices
from eh30pt.logging import logger

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Sequence

    from eh30pt.labeling.studio import StudioTask
    from eh30pt.settings import LabelStudioSettings


class LabeledRow[LabelT: str](TypedDict):
    """One Label Studio task, carrying its picked annotation as a row."""

    id: int
    text: str
    phrase: str
    drug: str
    label: LabelT


class SplitRows[LabelT: str](NamedTuple):
    """The row lists the voted rows are carved into."""

    train: list[LabeledRow[LabelT]]
    eval: list[LabeledRow[LabelT]]
    test: list[LabeledRow[LabelT]]


type Fold[LabelT: str] = Callable[[str], LabelT | None]
"""
Folds one Label Studio choice onto a dataset's own label.

Returning None drops the choice.
"""


def labeled_row[LabelT: str](
    *,
    task_id: int,
    text: str,
    phrase: str,
    drug: str,
    discont: Sequence[str],
    other: Sequence[str],
    fold: Fold[LabelT],
) -> LabeledRow[LabelT] | None:
    """
    Map one annotation to a training row, or None to drop it.

    The first choice trains: meta flags never disqualify, and several choices
    across or within the boxes fall back to the first one, preferring the
    discontinuation box. Only annotations without any mappable choice are out.
    """
    if not text or not phrase or not drug:
        return None

    choices = discont or other
    if not choices:
        return None

    label = fold(choices[0])
    if label is None:
        return None

    return {
        "id": task_id,
        "text": text,
        "phrase": phrase,
        "drug": drug,
        "label": label,
    }


def labeled_rows[LabelT: str](
    tasks: Sequence[StudioTask],
    fold: Fold[LabelT],
) -> Iterator[LabeledRow[LabelT]]:
    """
    Yield one training row per task from its first usable annotation.

    Tasks mix machine span highlights, which carry no choices, with human
    choice annotations, so the first annotation mapping to a row wins and
    anything before it is skipped. Tasks without one train nothing.
    """
    seen = 0
    dropped = 0
    for task in tasks:
        seen += 1
        data = task["data"]

        pick: LabeledRow[LabelT] | None = None
        for annotation in sorted(task["annotations"], key=lambda item: item["id"]):
            grouped = group_choices(annotation, ("discont", "other"))
            pick = labeled_row(
                task_id=task["id"],
                text=data.get("text", ""),
                phrase=data.get("phrase", ""),
                drug=data.get("drug", ""),
                discont=grouped["discont"],
                other=grouped["other"],
                fold=fold,
            )

            if pick is not None:
                break

        if pick is None:
            dropped += 1
            continue

        yield pick

    logger.debug(
        "kept %d of %d tasks (%d without a usable choice)",
        seen - dropped,
        seen,
        dropped,
    )


def stream_labeled_rows[LabelT: str](
    settings: LabelStudioSettings,
    labeling_projects: Sequence[int],
    fold: Fold[LabelT],
) -> Iterator[LabeledRow[LabelT]]:
    """Stream voted training rows from Label Studio projects, connecting lazily."""
    with StudioClient(settings) as studio:
        tallies: list[tuple[int, int, int]] = []
        for project in labeling_projects:
            match studio.tasks(project):
                case Success(tasks):
                    rows = list(labeled_rows(tasks, fold))
                    tallies.append((project, len(tasks), len(rows)))
                    yield from rows
                case Failure(exc):
                    msg = f"could not read tasks of project {project}: {exc}"
                    raise RuntimeError(msg) from exc

        logger.debug(
            "voted rows per project (tasks -> rows): %s",
            ", ".join(
                f"{project}: {task_count} -> {row_count}"
                for project, task_count, row_count in tallies
            ),
        )


def _shuffle_split(
    labels: list[str],
    size: int,
    seed: int,
    *,
    stage: str,
) -> tuple[list[int], list[int]]:
    """Split positions stratifiably, falling back to a plain shuffle."""
    positions = range(len(labels))
    try:
        splitter = StratifiedShuffleSplit(n_splits=1, test_size=size, random_state=seed)
        train_index, held_index = next(splitter.split(positions, labels))
    except ValueError as exc:
        logger.warning(
            "stratified %s split failed (%s), falling back to a plain shuffle",
            stage,
            exc,
        )

        plain = ShuffleSplit(n_splits=1, test_size=size, random_state=seed)
        train_index, held_index = next(plain.split(positions))
    else:
        logger.debug(
            "stratified %s split held out %d of %d rows with seed %d",
            stage,
            len(held_index),
            len(labels),
            seed,
        )

    return train_index.tolist(), held_index.tolist()


def carve[LabelT: str](
    rows: Sequence[LabeledRow[LabelT]],
    *,
    seed: int,
) -> SplitRows[LabelT]:
    """Carve the voted rows into stratified 70/20/10 train, eval and test splits."""
    ordered = sorted(rows, key=lambda row: row["id"])
    logger.debug("carving %d voted rows with seed %d", len(ordered), seed)

    min_rows_for_three_splits = 3
    if len(ordered) < min_rows_for_three_splits:
        logger.debug("too few rows for three splits, keeping everything in train")
        return SplitRows(train=ordered, eval=[], test=[])

    n_eval = max(1, round(len(ordered) * 0.2))
    n_test = max(1, round(len(ordered) * 0.1))
    counts = Counter(row.get("label") for row in ordered)
    train_rows = [row for row in ordered if counts[row.get("label")] == 1]
    pool = [row for row in ordered if counts[row.get("label")] > 1]
    logger.debug(
        "%d singleton rows stay in train, carving %d pooled rows",
        len(train_rows),
        len(pool),
    )

    if len(pool) < min_rows_for_three_splits:
        logger.debug("pool too small for three splits, keeping everything in train")
        return SplitRows(train=ordered, eval=[], test=[])

    pool_labels = [row.get("label", "") or "" for row in pool]
    train_index, held_index = _shuffle_split(
        pool_labels,
        min(n_eval + n_test, len(pool) - 1),
        seed,
        stage="train/held",
    )

    train_rows.extend(pool[index] for index in train_index)
    held = [pool[index] for index in held_index]

    held_labels = [row.get("label", "") or "" for row in held]
    eval_index, test_index = _shuffle_split(
        held_labels, min(n_test, len(held) - 1), seed, stage="eval/test"
    )

    splits = SplitRows(
        train=train_rows,
        eval=[held[index] for index in eval_index],
        test=[held[index] for index in test_index],
    )

    logger.debug(
        "split rows into train=%d eval=%d test=%d",
        len(splits.train),
        len(splits.eval),
        len(splits.test),
    )

    return splits


def above[LabelT: str](
    splits: SplitRows[LabelT],
    start_after: int | None,
) -> SplitRows[LabelT]:
    """
    Drop the rows at or below `start_after` from every split.

    The filter runs after the carve, so a resumed run sees the same rows in
    the same splits it saw before, only without the consumed ones.
    """
    if start_after is None:
        return splits

    consumed: int = start_after

    def keep(rows: list[LabeledRow[LabelT]]) -> list[LabeledRow[LabelT]]:
        return [row for row in rows if row["id"] > consumed]

    return SplitRows(
        train=keep(splits.train),
        eval=keep(splits.eval),
        test=keep(splits.test),
    )

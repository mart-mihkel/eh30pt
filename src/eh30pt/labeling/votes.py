"""Turning Label Studio annotations into voted training rows, split three ways."""

from collections import Counter
from typing import TYPE_CHECKING, NamedTuple, NotRequired, TypedDict

from returns.result import Failure, Success
from sklearn.model_selection import ShuffleSplit, StratifiedShuffleSplit

from eh30pt.labeling.studio import StudioClient, group_choices
from eh30pt.logging import logger

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Sequence

    from eh30pt.labeling.studio import StudioTask
    from eh30pt.settings import LabelStudioSettings


class LabeledRow[LabelT: str](TypedDict):
    """
    One Label Studio task, carrying the annotation that won its vote.

    Only `id`, `text` and `label` are certain: `phrase` and `drug` arrive with
    the extraction that preceded the labeling, and rows streamed straight from
    the events query predate that extraction entirely.
    """

    id: int
    text: str
    phrase: NotRequired[str]
    drug: NotRequired[str]
    label: NotRequired[LabelT]


class SplitRows[LabelT: str](NamedTuple):
    """The row lists the voted rows are carved into."""

    train: list[LabeledRow[LabelT]]
    eval: list[LabeledRow[LabelT]]
    test: list[LabeledRow[LabelT]]


type Fold[LabelT: str] = Callable[[str], LabelT | None]
"""
Folds one Label Studio choice onto a dataset's own label.

Returning None drops the choice, which is how a dataset ignores labels it
does not know about rather than inventing one for them.
"""


def labeled_row[LabelT: str](
    *,
    task_id: int,
    text: str,
    phrase: str,
    drug: str,
    discont: Sequence[str],
    other: Sequence[str],
    meta: Sequence[str],
    fold: Fold[LabelT],
) -> LabeledRow[LabelT] | None:
    """
    Map one first-per-task annotation to a training row, or None to drop it.

    Only a lone choice trains: anything flagged by meta, asserting in both
    boxes at once, or carrying zero or several labels, is left out.
    """
    if not text or not phrase or not drug:
        return None

    if meta or (discont and other):
        return None

    choices = discont or other
    if len(choices) != 1:
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
    Yield one training row per task from its annotations' majority vote.

    Every mappable annotation votes once; ties and tasks without a mappable
    annotation train nothing. The winning row is the earliest annotation behind
    the winning label.

    Votes are counted after folding, so annotators who disagreed on the label
    but folded it the same way still produce a row.
    """
    seen = 0
    dropped_empty = 0
    dropped_tied = 0
    for task in tasks:
        seen += 1
        data = task["data"]
        rows: list[LabeledRow[LabelT]] = []
        for annotation in sorted(task["annotations"], key=lambda item: item["id"]):
            grouped = group_choices(annotation, ("discont", "other", "meta"))
            row = labeled_row(
                task_id=task["id"],
                text=data.get("text", ""),
                phrase=data.get("phrase", ""),
                drug=data.get("drug", ""),
                discont=grouped["discont"],
                other=grouped["other"],
                meta=grouped["meta"],
                fold=fold,
            )
            if row is not None:
                rows.append(row)

        counts = Counter(row["label"] for row in rows)
        top = counts.most_common(2)
        if not top:
            dropped_empty += 1
            continue

        if len(top) > 1 and top[0][1] == top[1][1]:
            dropped_tied += 1
            continue

        winner = top[0][0]
        yield next(row for row in rows if row["label"] == winner)

    logger.debug(
        "voted %d of %d tasks (%d without a mappable annotation, %d tied)",
        seen - dropped_empty - dropped_tied,
        seen,
        dropped_empty,
        dropped_tied,
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

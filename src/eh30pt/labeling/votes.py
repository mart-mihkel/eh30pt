"""Turning Label Studio annotations into voted training rows, split three ways."""

from collections import Counter
from typing import TYPE_CHECKING, NamedTuple, NotRequired, TypedDict, cast

from datasets import Dataset
from returns.result import Failure, Success

from eh30pt.labeling.studio import StudioClient, group_choices

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
    for task in tasks:
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
        if not top or (len(top) > 1 and top[0][1] == top[1][1]):
            continue

        winner = top[0][0]
        yield next(row for row in rows if row["label"] == winner)


def stream_labeled_rows[LabelT: str](
    settings: LabelStudioSettings,
    labeling_projects: Sequence[int],
    fold: Fold[LabelT],
) -> Iterator[LabeledRow[LabelT]]:
    """Stream voted training rows from Label Studio projects, connecting lazily."""
    with StudioClient(settings) as studio:
        for project in labeling_projects:
            match studio.tasks(project):
                case Success(tasks):
                    yield from labeled_rows(tasks, fold)
                case Failure(exc):
                    msg = f"could not read tasks of project {project}: {exc}"
                    raise RuntimeError(msg) from exc


def carve[LabelT: str](
    rows: Sequence[LabeledRow[LabelT]],
    *,
    seed: int = 0,
) -> SplitRows[LabelT]:
    """
    Carve the voted rows into train, eval and test splits.

    Rows are ordered by id first, so which split a row lands in depends on
    the rows themselves rather than the order Label Studio paged them in.
    Two tenths are held out and cut in half a second time, mirroring how
    oblique carves eval out of what its first cut held back, and rows too
    few to leave train at least one stay in train on their own.
    """
    ordered = sorted(rows, key=lambda row: row["id"])
    tenth = max(1, round(len(ordered) * 0.1))

    if len(ordered) - tenth * 2 < 1:
        return SplitRows(train=ordered, eval=[], test=[])

    dataset = Dataset.from_list(cast("list[dict[str, object]]", ordered))
    rest = dataset.train_test_split(test_size=tenth * 2, seed=seed, keep_in_memory=True)
    halves = rest["test"].train_test_split(
        test_size=tenth, seed=seed, keep_in_memory=True
    )

    return SplitRows(
        train=_rows_of(rest["train"]),
        eval=_rows_of(halves["train"]),
        test=_rows_of(halves["test"]),
    )


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
        train=keep(splits.train), eval=keep(splits.eval), test=keep(splits.test)
    )


def _rows_of[LabelT: str](dataset: Dataset) -> list[LabeledRow[LabelT]]:
    """
    Return the rows of one carved split as typed rows.

    Every row comes from `labeled_row`, which fills all four fields, so the
    arrow columns round trip back into the full typed dict.
    """
    return cast("list[LabeledRow[LabelT]]", dataset.to_list())

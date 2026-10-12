"""Tests for turning Label Studio annotations into voted, split training rows."""

from typing import Final

from eh30pt.labeling.votes import (
    LabeledRow,
    above,
    carve,
    labeled_row,
    labeled_rows,
)

_labels: Final = {"discontinued": "yes", "continued": "no"}


def _fold(choice: str) -> str | None:
    """Fold one choice onto the binary answers, dropping unknown labels."""
    return _labels.get(choice)


def test_labeled_row_maps_a_lone_choice() -> None:
    row = labeled_row(
        task_id=1,
        text="text",
        phrase="phrase",
        drug="drug",
        discont=("discontinued",),
        other=(),
        meta=(),
        fold=_fold,
    )

    assert row == {
        "id": 1,
        "text": "text",
        "phrase": "phrase",
        "drug": "drug",
        "label": "yes",
    }


def test_labeled_row_drops_flagged_or_ambiguous_annotations() -> None:
    for discont, other, meta in (
        ((), (), ()),  # no choice at all
        (("discontinued", "discontinued"), (), ()),  # two choices
        ((), ("discontinued", "continued"), ()),  # two choices
        (("discontinued",), ("continued",), ()),  # asserted in both boxes
        (("discontinued",), (), ("Incomplete phrase",)),  # flagged by meta
        (("unmapped choice",), (), ()),  # the fold rejects it
        ((), (), ()),
    ):
        assert (
            labeled_row(
                task_id=1,
                text="t",
                phrase="p",
                drug="d",
                discont=discont,
                other=other,
                meta=meta,
                fold=_fold,
            )
            is None
        )


def test_labeled_row_drops_rows_missing_extracted_fields() -> None:
    for text, phrase, drug in (
        ("", "p", "d"),
        ("t", "", "d"),
        ("t", "p", ""),
    ):
        assert (
            labeled_row(
                task_id=1,
                text=text,
                phrase=phrase,
                drug=drug,
                discont=("discontinued",),
                other=(),
                meta=(),
                fold=_fold,
            )
            is None
        )


def _ballot(annotation_id: int, control: str, label: str) -> dict[str, object]:
    return {
        "id": annotation_id,
        "result": [
            {
                "from_name": control,
                "to_name": "text",
                "type": "choices",
                "value": {"choices": [label]},
            }
        ],
    }


def _task(task_id: int, ballots: list[dict[str, object]]) -> dict[str, object]:
    return {
        "id": task_id,
        "inner_id": 1,
        "data": {"text": "t", "phrase": "p", "drug": "d", "reason": "r"},
        "annotations": ballots,
    }


def test_labeled_rows_take_the_majority_vote() -> None:
    tasks = [
        _task(
            1,
            [
                _ballot(1, "discont", "unmapped choice"),
                _ballot(2, "discont", "discontinued"),
                _ballot(3, "discont", "continued"),
                _ballot(4, "discont", "discontinued"),
            ],
        ),
    ]

    rows = list(labeled_rows(tasks, _fold))  # ty: ignore[invalid-argument-type]

    assert [(row["id"], row["label"]) for row in rows] == [(1, "yes")]


def test_labeled_rows_drop_tied_tasks() -> None:
    tasks = [
        _task(
            1, [_ballot(1, "discont", "discontinued"), _ballot(2, "other", "continued")]
        )
    ]

    assert list(labeled_rows(tasks, _fold)) == []  # ty: ignore[invalid-argument-type]


def test_labeled_rows_skip_unmappable_annotations() -> None:
    tasks = [_task(1, [{"id": 1, "result": []}, _ballot(2, "other", "continued")])]

    rows = list(labeled_rows(tasks, _fold))  # ty: ignore[invalid-argument-type]

    assert [(row["id"], row["label"]) for row in rows] == [(1, "no")]


def _voted_rows(count: int) -> list[LabeledRow[str]]:
    """Return `count` voted rows cycling through the folded labels."""
    labels = tuple(_labels.values())
    return [
        {
            "id": task_id,
            "text": "text",
            "phrase": "phrase",
            "drug": "drug",
            "label": labels[task_id % len(labels)],
        }
        for task_id in range(count)
    ]


def test_carve_puts_every_row_in_exactly_one_split() -> None:
    rows = _voted_rows(100)
    splits = carve(rows)

    assert [len(splits.train), len(splits.eval), len(splits.test)] == [80, 10, 10]
    assert sorted(row["id"] for row in (*splits.train, *splits.eval, *splits.test)) == [
        row["id"] for row in rows
    ]


def test_carve_shuffles_the_splits_reproducibly() -> None:
    splits = carve(_voted_rows(100))
    reordered = carve(list(reversed(_voted_rows(100))))

    assert [row["id"] for row in splits.test] == [row["id"] for row in reordered.test]
    assert sorted(row["id"] for row in splits.test) != list(range(90, 100))


def test_carve_keeps_rows_that_cannot_fill_three_splits_in_train() -> None:
    for count in (0, 1, 2):
        rows = _voted_rows(count)
        splits = carve(rows)

        assert splits.train == rows
        assert splits.eval == []
        assert splits.test == []


def test_above_drops_consumed_rows_without_moving_them() -> None:
    splits = carve(_voted_rows(100))

    assert above(splits, start_after=None) is splits

    consumed = 40
    resumed = above(splits, start_after=consumed)

    for before, after in zip(splits, resumed, strict=True):
        assert [row["id"] for row in after] == [
            row["id"] for row in before if row["id"] > consumed
        ]

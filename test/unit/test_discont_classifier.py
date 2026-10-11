"""Tests for mapping exported annotations to classifier training rows."""

from eh30pt.datasets.discontinuation.classify import (
    DiscontClassifier,
    DiscontClassifierRow,
    labeled_row,
)


def test_labeled_row_maps_each_box_choice() -> None:
    for discont, other, label in (
        (("Treatment success",), (), "Treatment success"),
        (("Treatment inefficacy",), (), "Treatment inefficacy"),
        (("Contraindication",), (), "Contraindication"),
        (("Adverse reactions",), (), "Adverse reactions"),
        (("Other",), (), "Other"),
        (
            (),
            ("Adverse reaction without discontinuation",),
            "Adverse reaction without discontinuation",
        ),
        (
            (),
            ("Dosage change",),
            "Dosage change",
        ),
        ((), ("Other non-discontinuation",), "Other non-discontinuation"),
    ):
        row = labeled_row(
            task_id=1,
            text="text",
            phrase="phrase",
            drug="drug",
            discont=discont,
            other=other,
            meta=(),
        )

        assert row == {
            "id": 1,
            "text": "text",
            "phrase": "phrase",
            "drug": "drug",
            "label": label,
        }


def test_labeled_row_drops_flagged_or_ambiguous_annotations() -> None:
    assert (
        labeled_row(
            task_id=1,
            text="t",
            phrase="p",
            drug="d",
            discont=(),
            other=(),
            meta=(),
        )
        is None
    )
    assert (
        labeled_row(
            task_id=1,
            text="t",
            phrase="p",
            drug="d",
            discont=("Other", "Other"),
            other=(),
            meta=(),
        )
        is None
    )
    assert (
        labeled_row(
            task_id=1,
            text="t",
            phrase="p",
            drug="d",
            discont=("Nope",),
            other=(),
            meta=(),
        )
        is None
    )
    assert (
        labeled_row(
            task_id=1,
            text="t",
            phrase="p",
            drug="d",
            discont=("Other",),
            other=("Dosage change",),
            meta=(),
        )
        is None
    )
    assert (
        labeled_row(
            task_id=1,
            text="t",
            phrase="p",
            drug="d",
            discont=(),
            other=(
                "Dosage change",
                "Other non-discontinuation",
            ),
            meta=(),
        )
        is None
    )
    assert (
        labeled_row(
            task_id=1,
            text="t",
            phrase="p",
            drug="d",
            discont=("Other",),
            other=(),
            meta=("Incomplete phrase",),
        )
        is None
    )
    assert (
        labeled_row(
            task_id=1,
            text="",
            phrase="p",
            drug="d",
            discont=("Other",),
            other=(),
            meta=(),
        )
        is None
    )
    assert (
        labeled_row(
            task_id=1,
            text="t",
            phrase="",
            drug="d",
            discont=("Other",),
            other=(),
            meta=(),
        )
        is None
    )
    assert (
        labeled_row(
            task_id=1,
            text="t",
            phrase="p",
            drug="",
            discont=("Other",),
            other=(),
            meta=(),
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
    from eh30pt.datasets.discontinuation.classify import labeled_rows

    tasks = [
        _task(
            1,
            [
                _ballot(1, "discont", "Other"),
                _ballot(2, "discont", "Adverse reactions"),
                _ballot(3, "discont", "Adverse reactions"),
            ],
        ),
    ]

    rows = list(labeled_rows(tasks))  # ty: ignore[invalid-argument-type]

    assert [(row["id"], row["label"]) for row in rows] == [(1, "Adverse reactions")]


def test_labeled_rows_drop_tied_tasks() -> None:
    from eh30pt.datasets.discontinuation.classify import labeled_rows

    tasks = [
        _task(
            1,
            [
                _ballot(1, "discont", "Other"),
                _ballot(2, "other", "Dosage change"),
            ],
        ),
    ]

    assert list(labeled_rows(tasks)) == []  # ty: ignore[invalid-argument-type]


def test_labeled_rows_skip_unmappable_annotations() -> None:
    from eh30pt.datasets.discontinuation.classify import labeled_rows

    tasks = [
        {
            "id": 1,
            "inner_id": 1,
            "data": {"text": "t", "phrase": "p", "drug": "d", "reason": "r"},
            "annotations": [
                {"id": 1, "result": []},
                {
                    "id": 2,
                    "result": [
                        {
                            "from_name": "other",
                            "to_name": "text",
                            "type": "choices",
                            "value": {"choices": ["Dosage change"]},
                        }
                    ],
                },
            ],
        },
    ]

    rows = list(labeled_rows(tasks))  # ty: ignore[invalid-argument-type]

    assert [(row["id"], row["label"]) for row in rows] == [(1, "Dosage change")]


def _voted_rows(count: int) -> list[DiscontClassifierRow]:
    """Return `count` voted rows cycling through the classifier labels."""
    labels = tuple(DiscontClassifier.id2label.values())
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
    from eh30pt.datasets.discontinuation.classify import _carve

    rows = _voted_rows(100)
    splits = _carve(rows)

    assert [len(splits.train), len(splits.eval), len(splits.test)] == [80, 10, 10]
    assert sorted(row["id"] for row in (*splits.train, *splits.eval, *splits.test)) == [
        row["id"] for row in rows
    ]


def test_carve_shuffles_the_splits_reproducibly() -> None:
    from eh30pt.datasets.discontinuation.classify import _carve

    splits = _carve(_voted_rows(100))
    reordered = _carve(list(reversed(_voted_rows(100))))

    assert [row["id"] for row in splits.test] == [row["id"] for row in reordered.test]
    assert sorted(row["id"] for row in splits.test) != list(range(90, 100))


def test_carve_keeps_rows_that_cannot_fill_three_splits_in_train() -> None:
    from eh30pt.datasets.discontinuation.classify import _carve

    for count in (0, 1, 2):
        rows = _voted_rows(count)
        splits = _carve(rows)

        assert splits.train == rows
        assert splits.eval == []
        assert splits.test == []


def test_above_drops_consumed_rows_without_moving_them() -> None:
    from eh30pt.datasets.discontinuation.classify import _above, _carve

    splits = _carve(_voted_rows(100))

    assert _above(splits, start_after=None) is splits

    consumed = 40
    resumed = _above(splits, start_after=consumed)

    for before, after in zip(splits, resumed, strict=True):
        assert [row["id"] for row in after] == [
            row["id"] for row in before if row["id"] > consumed
        ]

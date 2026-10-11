"""Tests for mapping exported annotations to classifier training rows."""

from eh30pt.datasets.discontinuation.classify import labeled_row


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

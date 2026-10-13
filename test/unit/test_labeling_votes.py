"""Tests for turning Label Studio annotations into voted, split training rows."""

from typing import Final

from eh30pt.labeling.loaders.extract import (
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
        fold=_fold,
    )

    assert row == {
        "id": 1,
        "text": "text",
        "phrase": "phrase",
        "drug": "drug",
        "label": "yes",
    }


def test_labeled_row_ignores_meta_flags() -> None:
    row = labeled_row(
        task_id=1,
        text="text",
        phrase="phrase",
        drug="drug",
        discont=(),
        other=("continued",),
        fold=_fold,
    )

    assert row is not None
    assert row["label"] == "no"


def test_labeled_row_picks_the_first_choice() -> None:
    for discont, other, label in (
        (("discontinued", "continued"), (), "yes"),  # two choices, first wins
        ((), ("continued", "discontinued"), "no"),  # two choices, first wins
        (("discontinued",), ("continued",), "yes"),  # both boxes, discont first
    ):
        row = labeled_row(
            task_id=1,
            text="text",
            phrase="phrase",
            drug="drug",
            discont=discont,
            other=other,
            fold=_fold,
        )

        assert row is not None
        assert row["label"] == label


def test_labeled_row_drops_annotations_without_a_mappable_choice() -> None:
    for discont, other in (
        ((), ()),  # no choice at all, such as a machine span highlight
        (("unmapped choice",), ()),  # the fold rejects the first choice
    ):
        assert (
            labeled_row(
                task_id=1,
                text="t",
                phrase="p",
                drug="d",
                discont=discont,
                other=other,
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


def _span_ballot(annotation_id: int, label: str) -> dict[str, object]:
    return {
        "id": annotation_id,
        "result": [
            {
                "from_name": "span",
                "to_name": "text",
                "type": "labels",
                "value": {
                    "start": 0,
                    "end": 5,
                    "text": "katke",
                    "labels": [label],
                },
            }
        ],
    }


def test_labeled_rows_take_the_first_usable_annotation() -> None:
    tasks = [
        _task(
            1,
            [
                _ballot(1, "discont", "discontinued"),
                _ballot(2, "other", "continued"),
            ],
        ),
    ]

    rows = list(labeled_rows(tasks, _fold))  # ty: ignore[invalid-argument-type]

    assert [(row["id"], row["label"]) for row in rows] == [(1, "yes")]


def test_labeled_rows_skip_machine_span_annotations() -> None:
    tasks = [
        _task(
            1,
            [
                _span_ballot(1, "phrase"),
                _span_ballot(2, "reason"),
                _span_ballot(3, "drug"),
                _ballot(4, "discont", "discontinued"),
            ],
        ),
    ]

    rows = list(labeled_rows(tasks, _fold))  # ty: ignore[invalid-argument-type]

    assert [(row["id"], row["label"]) for row in rows] == [(1, "yes")]


def test_labeled_rows_skip_unmappable_annotations() -> None:
    tasks = [_task(1, [{"id": 1, "result": []}, _ballot(2, "other", "continued")])]

    rows = list(labeled_rows(tasks, _fold))  # ty: ignore[invalid-argument-type]

    assert [(row["id"], row["label"]) for row in rows] == [(1, "no")]


def test_labeled_rows_keep_meta_flagged_choices() -> None:
    tasks = [
        {
            "id": 1,
            "inner_id": 1,
            "data": {"text": "t", "phrase": "p", "drug": "d"},
            "annotations": [
                {
                    "id": 1,
                    "result": [
                        {
                            "from_name": "discont",
                            "to_name": "text",
                            "type": "choices",
                            "value": {"choices": ["discontinued"]},
                        },
                        {
                            "from_name": "meta",
                            "to_name": "text",
                            "type": "choices",
                            "value": {"choices": ["Incomplete phrase"]},
                        },
                    ],
                }
            ],
        },
    ]

    rows = list(labeled_rows(tasks, _fold))  # ty: ignore[invalid-argument-type]

    assert [(row["id"], row["label"]) for row in rows] == [(1, "yes")]


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
    splits = carve(rows, seed=0)

    assert [len(splits.train), len(splits.eval), len(splits.test)] == [70, 20, 10]
    assert sorted(row["id"] for row in (*splits.train, *splits.eval, *splits.test)) == [
        row["id"] for row in rows
    ]


def test_carve_shuffles_the_splits_reproducibly() -> None:
    splits = carve(_voted_rows(100), seed=0)
    reordered = carve(list(reversed(_voted_rows(100))), seed=0)

    assert [row["id"] for row in splits.test] == [row["id"] for row in reordered.test]
    assert sorted(row["id"] for row in splits.test) != list(range(90, 100))


def test_carve_keeps_rows_that_cannot_fill_three_splits_in_train() -> None:
    for count in (0, 1, 2):
        rows = _voted_rows(count)
        splits = carve(rows, seed=0)

        assert splits.train == rows
        assert splits.eval == []
        assert splits.test == []


def test_carve_stratifies_labels_across_splits() -> None:
    rows: list[LabeledRow[str]] = [
        {
            "id": task_id,
            "text": "text",
            "phrase": "phrase",
            "drug": "drug",
            "label": "yes",
        }
        for task_id in range(90)
    ] + [
        {
            "id": task_id,
            "text": "text",
            "phrase": "phrase",
            "drug": "drug",
            "label": "no",
        }
        for task_id in range(90, 100)
    ]
    splits = carve(rows, seed=0)

    assert {row["label"] for row in splits.eval} == {"yes", "no"}
    assert {row["label"] for row in splits.test} == {"yes", "no"}


def test_carve_gives_every_class_to_eval_and_test() -> None:
    labels = tuple(f"class-{index}" for index in range(8))
    rows = [
        {
            "id": task_id,
            "text": "text",
            "phrase": "phrase",
            "drug": "drug",
            "label": labels[task_id % len(labels)],
        }
        for task_id in range(80)
    ]
    splits = carve(rows, seed=0)  # ty: ignore[invalid-argument-type]

    assert {row["label"] for row in splits.eval} == set(labels)
    assert {row["label"] for row in splits.test} == set(labels)


def test_carve_varies_splits_with_seed() -> None:
    first = carve(_voted_rows(100), seed=0)
    second = carve(_voted_rows(100), seed=1)

    assert [row["id"] for row in first.test] != [row["id"] for row in second.test]


def test_above_drops_consumed_rows_without_moving_them() -> None:
    splits = carve(_voted_rows(100), seed=0)

    assert above(splits, start_after=None) is splits

    consumed = 40
    resumed = above(splits, start_after=consumed)

    for before, after in zip(splits, resumed, strict=True):
        assert [row["id"] for row in after] == [
            row["id"] for row in before if row["id"] > consumed
        ]

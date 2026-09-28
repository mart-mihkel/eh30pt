"""Tests for label-constrained decoding helpers."""

import torch

from eh30pt.modeling.trainer import _allowed_next, _constrain_eval_logits


def test_allowed_next_walks_completions() -> None:
    completions = ((1, 2), (1, 3))
    assert _allowed_next(completions, 0, ()) == [1]
    assert _allowed_next(completions, 0, (1,)) == [2, 3]


def test_allowed_next_falls_back() -> None:
    completions = ((1, 2), (1, 3))
    assert _allowed_next(completions, 0, (1, 2)) == [0]
    assert _allowed_next(completions, 0, (9,)) == [0]
    assert _allowed_next((), 0, ()) == [0]


def test_constrain_forces_likeliest_legal_completion() -> None:
    completions = ((1, 2), (3, 4), (5,))
    labels = torch.tensor(
        [
            [-100, -100, 1, 2],
            [-100, 5, -100, -100],
            [-100, -100, -100, -100],
        ]
    )
    logits = torch.zeros(3, 4, 6)

    logits[0, 1] = torch.tensor([0.0, 0.0, 0.0, 5.0, 0.0, 100.0])
    logits[0, 2] = torch.tensor([0.0, 0.0, 1.0, 0.0, 2.0, 100.0])
    logits[1, 0] = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 7.0])

    masked = _constrain_eval_logits(logits, labels, completions)

    assert torch.equal(masked[0, 0], logits[0, 0])
    assert torch.equal(masked[0, 3], logits[0, 3])
    assert torch.equal(masked[1, 1:], logits[1, 1:])
    assert torch.equal(masked[2], logits[2])
    assert masked[0, 1, 5] == float("-inf")
    forced = tuple(int(torch.argmax(masked[b, t])) for b, t in ((0, 1), (0, 2)))
    assert forced == (3, 4)
    assert int(torch.argmax(masked[1, 0])) == completions[2][0]


def test_constrain_empty_completions_returns_logits() -> None:
    logits = torch.zeros(1, 2, 5)
    labels = torch.tensor([[-100, 0]])
    assert torch.equal(_constrain_eval_logits(logits, labels, ()), logits)


def test_constrain_skips_rows_without_label_span() -> None:
    logits = torch.zeros(1, 3, 4)
    labels = torch.tensor([[-100, -100, -100]])
    completions = ((1, 2),)

    assert torch.equal(_constrain_eval_logits(logits, labels, completions), logits)


def test_constrain_skips_negative_force_position() -> None:
    logits = torch.zeros(1, 2, 4)
    labels = torch.tensor([[0, -100]])
    completions = ((1,),)

    assert torch.equal(_constrain_eval_logits(logits, labels, completions), logits)


def test_constrain_skips_length_mismatched_candidates() -> None:
    logits = torch.zeros(1, 3, 4)
    labels = torch.tensor([[-100, 1, 2]])
    completions = ((1, 2, 3),)

    assert torch.equal(_constrain_eval_logits(logits, labels, completions), logits)

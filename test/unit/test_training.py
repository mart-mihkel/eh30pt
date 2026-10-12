"""Tests for label-constrained decoding helpers."""

from typing import TYPE_CHECKING, NamedTuple

import torch
from transformers.loss.loss_utils import ForCausalLMLoss

from eh30pt.modeling.training.constraints import (
    _allowed_next,
    _constrain_eval_logits,
    _constrained_causal_loss,
    _prefix_allowed_tokens,
    _restrict_to_label_prefixes,
)

if TYPE_CHECKING:
    from torch import Tensor


class _Outputs(NamedTuple):
    """Stand-in for a causal LM output object carrying logits."""

    logits: Tensor


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


def test_constrain_leaves_other_positions_and_rows_untouched() -> None:
    """Only the argmax at label positions may change, never the rest of the row."""
    completions = ((1, 2), (3, 4))
    labels = torch.tensor([[-100, -100, 1, 2]])
    logits = torch.zeros(1, 5, 6)
    logits[0, 1] = torch.tensor([0.0, 5.0, 0.0, 0.0, 0.0, 0.0])
    logits[0, 2] = torch.tensor([0.0, 0.0, 6.0, 0.0, 0.0, 0.0])

    masked = _constrain_eval_logits(logits, labels, completions)

    assert torch.equal(masked[0, 0], logits[0, 0])
    assert torch.equal(masked[0, 3], logits[0, 3])
    assert torch.equal(masked[0, 4], logits[0, 4])

    forced_rows = masked[0, [1, 2], :]
    assert (forced_rows == float("-inf")).sum().item() == forced_rows.numel() - 2
    assert forced_rows.argmax(-1).tolist() == [1, 2]


def test_prefix_allowed_tokens_walks_toward_any_legal_label() -> None:
    completions = ((1, 2), (1, 3), (4,))
    gold = (1, 2)

    # The (4,) label starts the same way but diverges immediately after.
    assert _prefix_allowed_tokens(completions, gold) == (
        frozenset({1, 4}),
        frozenset({2, 3}),
    )


def test_prefix_allowed_tokens_keeps_gold_when_completion_is_unknown() -> None:
    """A label missing from the eval split must still be trainable."""
    assert _prefix_allowed_tokens(((1, 2),), (9, 9)) == (
        frozenset({1, 9}),
        frozenset({9}),
    )


def test_restrict_to_label_prefixes_masks_illegal_next_tokens() -> None:
    completions = ((1, 2), (1, 3))
    labels = torch.tensor([[-100, -100, 1, 2]])
    logits = torch.zeros(1, 4, 5)

    restricted = _restrict_to_label_prefixes(logits, labels, completions)

    # The label span sits at 2 and 3, so those logits are predicted at 1 and 2.
    # Position 1 sees an empty prefix, so only token {1} stays live.
    assert restricted[0, 1].tolist() == [
        float("-inf"),
        0.0,
        float("-inf"),
        float("-inf"),
        float("-inf"),
    ]
    # Position 2 has seen the token 1, so both continuations stay live.
    assert restricted[0, 2].tolist() == [
        float("-inf"),
        float("-inf"),
        0.0,
        0.0,
        float("-inf"),
    ]
    # Positions outside the span the constraint covers are untouched.
    assert torch.equal(restricted[0, 0], logits[0, 0])
    assert torch.equal(restricted[0, 3], logits[0, 3])


def test_constrained_causal_loss_stays_finite_for_unknown_gold() -> None:
    """Masking must never leave the gold token with a -inf logit."""
    labels = torch.tensor([[-100, -100, 7, 7]])

    loss = _constrained_causal_loss(((1, 2),))(
        _Outputs(logits=torch.zeros(1, 4, 8)), labels
    )

    assert torch.isfinite(loss)


def test_constrained_causal_loss_is_smaller_than_the_unrestricted_one() -> None:
    """Restricting the softmax must make the gold label look more likely."""
    completions = ((1, 2), (3, 4))
    labels = torch.tensor([[-100, -100, 1, 2]])
    outputs = _Outputs(logits=torch.zeros(1, 4, 6))

    restricted = _constrained_causal_loss(completions)(outputs, labels)
    unrestricted = ForCausalLMLoss(outputs.logits, labels, 6)

    assert float(restricted) < float(unrestricted)

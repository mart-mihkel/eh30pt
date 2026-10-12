"""Label-set logic shared by the constrained trainers."""

from typing import TYPE_CHECKING, Any, cast

import torch
from transformers.loss.loss_utils import ForCausalLMLoss

from eh30pt.constants import SENTINEL_TOKEN

if TYPE_CHECKING:
    from collections.abc import Callable

    from eh30pt.types import DataSplits


def _allowed_next(
    completions: tuple[tuple[int, ...], ...],
    fallback: int,
    tail: tuple[int, ...],
) -> list[int]:
    """Return legal continuations of `tail`, falling back to `fallback`."""
    out = {
        completion[len(tail)]
        for completion in completions
        if len(completion) > len(tail) and completion[: len(tail)] == tail
    }

    return sorted(out) if out else [fallback]


def _label_completions(
    data: DataSplits,
    *,
    eos_token_id: int | None = None,
) -> tuple[tuple[int, ...], ...]:
    """Collect distinct completion token sequences from the eval stream."""
    seen: list[tuple[int, ...]] = []
    for row in data.eval:
        labels = cast("list[int]", row["labels"])
        completion = tuple(token for token in labels if token != SENTINEL_TOKEN)
        if not completion or completion == (eos_token_id,):
            continue

        if completion not in seen:
            seen.append(completion)

    return tuple(seen)


def _prefix_allowed_tokens(
    completions: tuple[tuple[int, ...], ...],
    gold: tuple[int, ...],
) -> tuple[frozenset[int], ...]:
    """
    Return the tokens each gold step may legally continue to.

    The set at step `i` holds every token that keeps the sequence heading
    toward some legal label, plus the gold token itself. The gold token is
    unioned in because `completions` is built from the eval split: a training
    row whose label is absent from it, or whose prefix happens to match a
    different label, would otherwise be masked away from its own target and
    train against a -inf logit.
    """
    return tuple(
        frozenset(_allowed_next(completions, token, gold[:i])) | {token}
        for i, token in enumerate(gold)
    )


def _restrict_to_label_prefixes(
    logits: torch.Tensor,
    labels: torch.Tensor,
    completions: tuple[tuple[int, ...], ...],
) -> torch.Tensor:
    """
    Mask each label position's logits down to its legal next tokens.

    Positions outside the label span are left alone; the loss already ignores
    them through the sentinel value.
    """
    restricted = logits.clone()
    for b in range(restricted.shape[0]):
        pos = torch.where(labels[b] != SENTINEL_TOKEN)[0]
        if len(pos) == 0:
            continue

        force_at = pos - 1
        if bool((force_at < 0).any()):
            continue

        gold = tuple(labels[b][pos].tolist())
        for step, allowed in enumerate(_prefix_allowed_tokens(completions, gold)):
            row = force_at[step]
            allowed_ids = torch.tensor(
                sorted(allowed),
                dtype=torch.long,
                device=restricted.device,
            )
            restricted[b, row] = float("-inf")
            restricted[b, row, allowed_ids] = logits[b, row, allowed_ids]

    return restricted


def _constrained_causal_loss(
    completions: tuple[tuple[int, ...], ...],
) -> Callable[..., torch.Tensor]:
    """
    Build the `compute_loss_func` that trains under the same constraint as eval.

    The softmax is renormalized over the tokens each gold step may legally
    continue to, so the model is never asked to spend capacity separating a
    label from tokens that can never be generated.
    """

    def loss(
        outputs: Any,  # noqa: ANN401
        labels: torch.Tensor,
        *,
        num_items_in_batch: torch.Tensor | None = None,
        **kwargs: Any,  # noqa: ANN401
    ) -> torch.Tensor:
        logits = outputs["logits"] if isinstance(outputs, dict) else outputs.logits
        restricted = _restrict_to_label_prefixes(
            logits.float(),
            labels,
            completions,
        )

        return ForCausalLMLoss(
            restricted,
            labels,
            logits.size(-1),
            num_items_in_batch=num_items_in_batch,
            **kwargs,
        )

    return loss


def _constrain_eval_logits(
    logits: torch.Tensor,
    labels: torch.Tensor,
    completions: tuple[tuple[int, ...], ...],
) -> torch.Tensor:
    """Force each row's argmax to its most likely legal completion."""
    if not completions:
        return logits

    masked = logits.clone()
    for b in range(masked.shape[0]):
        pos = torch.where(labels[b] != SENTINEL_TOKEN)[0]
        if len(pos) == 0:
            continue

        force_at = pos - 1
        if bool((force_at < 0).any()):
            continue

        cands = [c for c in completions if len(c) == len(pos)]
        if not cands:
            continue

        cand_ids = torch.tensor(cands, dtype=torch.long, device=masked.device)

        scores = masked[b, force_at].gather(1, cand_ids.t()).sum(0)
        winner = cands[int(torch.argmax(scores))]
        forced = torch.tensor(winner, dtype=torch.long, device=masked.device)

        masked[b, force_at, :] = float("-inf")
        masked[b, force_at, forced] = 0.0

    return masked

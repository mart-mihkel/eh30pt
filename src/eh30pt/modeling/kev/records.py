"""Labelled decision records built from dataset splits."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from eh30pt.logging import logger
from eh30pt.systemone.stream import format_state

if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import Any

    from eh30pt.datasets.mixin import DatasetMixin
    from eh30pt.types import FewShot


@dataclass(frozen=True)
class KevRecord:
    """One state, one question, and its gold option index."""

    state: str
    instructions: str
    options: tuple[str, ...]
    label: int

    def __post_init__(self) -> None:
        """Reject labels outside the option list."""
        if not self.options:
            msg = "a decision record needs at least one option"
            raise ValueError(msg)

        if not 0 <= self.label < len(self.options):
            msg = f"label {self.label} is outside {len(self.options)} options"
            raise ValueError(msg)


def records_from_split(
    split: DatasetMixin[Any],
    *,
    instructions: str,
    labels: Sequence[str],
    examples: Sequence[FewShot] = (),
) -> list[KevRecord]:
    """Map a dataset split to decision records using its existing prompts."""
    label_index = {label: index for index, label in enumerate(labels)}
    records: list[KevRecord] = []
    skipped = 0
    for sample in split.samples():
        if sample.label not in label_index:
            skipped += 1
            continue

        records.append(
            KevRecord(
                state=format_state(examples, sample.user),
                instructions=instructions,
                options=tuple(labels),
                label=label_index[sample.label],
            ),
        )

    if skipped:
        logger.debug("skipped %d samples with labels outside the taxonomy", skipped)

    return records

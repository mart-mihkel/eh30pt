"""System One predictor for dataset samples."""

from dataclasses import dataclass
from typing import TYPE_CHECKING

from returns.result import Failure, Success

from eh30pt.logging import logger
from eh30pt.types import FewShot, Predictor

if TYPE_CHECKING:
    from collections.abc import Sequence

    from eh30pt.systemone.client import SystemOneClient


def format_state(examples: Sequence[FewShot], user: str) -> str:
    """Build the state a decision request sends for one sample."""
    rendered = "\n\n".join(f"{example.user}\n{example.label}" for example in examples)
    return f"{rendered}\n\n{user}" if rendered else user


@dataclass(frozen=True)
class SystemOnePredictor(Predictor[FewShot, str | None]):
    """Bind a decision client to a classification question and examples."""

    client: SystemOneClient
    instructions: str
    labels: tuple[str, ...]
    examples: Sequence[FewShot]

    async def predict(self, sample: FewShot) -> str | None:
        """Return the selected label, or None for a failed request."""
        result = await self.client.choose(
            state=format_state(self.examples, sample.user),
            instructions=self.instructions,
            labels=self.labels,
        )
        match result:
            case Success(choice):
                return choice
            case Failure(exc) | exc:
                logger.error("System One prediction failed: %s", exc)
                return None

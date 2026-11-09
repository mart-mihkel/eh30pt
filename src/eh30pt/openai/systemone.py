"""Classification through the llama.cpp System One endpoint."""

from typing import TYPE_CHECKING

import httpx
from pydantic import BaseModel

from eh30pt.logging import logger
from eh30pt.types import FewShot, Predictor

if TYPE_CHECKING:
    from collections.abc import Sequence

    from eh30pt.settings import ServerSettings


class ChoiceAnswer(BaseModel):
    """Choice returned by the decision model."""

    type: str
    choice: str


class DecisionResponse(BaseModel):
    """Decision model response containing named answers."""

    answers: dict[str, ChoiceAnswer]


class SystemOnePredictor(Predictor[FewShot, str | None]):
    """Predict a dataset label with a System One choice question."""

    def __init__(
        self,
        settings: ServerSettings,
        instructions: str,
        labels: tuple[str, ...],
        examples: Sequence[FewShot],
    ) -> None:
        """Store the endpoint and classification question."""
        self.url = f"{settings.base_url.rstrip('/')}/v1/systemone"
        self.api_key = settings.api_key
        self.timeout = settings.timeout
        self.instructions = instructions
        self.labels = labels
        self.examples = examples

    async def predict(self, sample: FewShot) -> str | None:
        """Return the selected label, or None for an invalid response."""
        examples = "\n\n".join(
            f"{example.user}\n{example.label}" for example in self.examples
        )

        state = f"{examples}\n\n{sample.user}" if examples else sample.user
        payload = {
            "state": state,
            "questions": {
                "label": {
                    "type": "choice",
                    "instructions": self.instructions,
                    "criteria": dict.fromkeys(self.labels),
                }
            },
        }

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    self.url,
                    json=payload,
                    headers={"Authorization": f"Bearer {self.api_key}"},
                )

                response.raise_for_status()
                answer = DecisionResponse.model_validate_json(response.content).answers[
                    "label"
                ]

                if answer.type != "choice" or answer.choice not in self.labels:
                    logger.error("System One returned an invalid choice")
                    return None

                return answer.choice
        except (httpx.HTTPError, ValueError, KeyError) as exc:
            logger.error("System One prediction failed: %s", exc)
            return None

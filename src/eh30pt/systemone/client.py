"""Choice decisions against a System One API."""

from typing import TYPE_CHECKING, Literal, TypedDict

import httpx
from pydantic import BaseModel

from eh30pt.types import asafe

if TYPE_CHECKING:
    from eh30pt.settings import ServerSettings


class ChoiceQuestion(TypedDict):
    """One choice question with named criteria."""

    type: Literal["choice"]
    instructions: str
    criteria: dict[str, None]


class DecisionRequest(TypedDict):
    """System One decision request."""

    state: str
    questions: dict[str, ChoiceQuestion]


class ChoiceAnswer(BaseModel):
    """Choice returned by the decision model."""

    type: Literal["choice"]
    choice: str


class ModelInfo(BaseModel):
    """Model identifier returned by the server."""

    id: str


class ModelList(BaseModel):
    """Models available at the server."""

    data: list[ModelInfo]


class DecisionResponse(BaseModel):
    """Decision model response containing named answers."""

    answers: dict[str, ChoiceAnswer]


class SystemOneClient:
    """Stateful client for System One choice decisions."""

    def __init__(self, settings: ServerSettings) -> None:
        """Configure a reusable HTTP connection pool."""
        self._client = httpx.AsyncClient(
            base_url=settings.base_url,
            headers={"Authorization": f"Bearer {settings.api_key}"},
            timeout=settings.timeout,
        )
        self._model: str | None = None

    @asafe
    async def model(self) -> str:
        """Return the id of the model served at the endpoint."""
        if self._model is not None:
            return self._model

        response = await self._client.get("/v1/models")
        response.raise_for_status()
        models = ModelList.model_validate_json(response.content).data
        if not models:
            msg = f"server at '{self._client.base_url}' serves no models"
            raise ValueError(msg)

        self._model = models[0].id.split("/")[-1]
        return self._model

    @asafe
    async def choose(
        self,
        *,
        state: str,
        instructions: str,
        labels: tuple[str, ...],
    ) -> str:
        """Return the selected label or a failure for an invalid response."""
        payload: DecisionRequest = {
            "state": state,
            "questions": {
                "label": {
                    "type": "choice",
                    "instructions": instructions,
                    "criteria": dict.fromkeys(labels),
                },
            },
        }
        response = await self._client.post("/v1/systemone", json=payload)
        response.raise_for_status()
        answer = DecisionResponse.model_validate_json(response.content).answers["label"]
        if answer.choice not in labels:
            msg = "System One returned an invalid choice"
            raise ValueError(msg)

        return answer.choice

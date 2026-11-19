"""Choice decisions against a System One API."""

from typing import TYPE_CHECKING

import httpx
from pydantic import BaseModel
from typesafe_sdk import Choice, SystemOneResponse

from eh30pt.types import asafe

if TYPE_CHECKING:
    from eh30pt.settings import ServerSettings


class DecisionRequest(BaseModel):
    """System One choice request."""

    state: str
    questions: dict[str, Choice]
    model: str


class ModelInfo(BaseModel):
    """Model identifier returned by the server."""

    id: str


class ModelList(BaseModel):
    """Models available at the server."""

    data: list[ModelInfo]


class SystemOneClient:
    """Stateful client for System One choice decisions."""

    def __init__(self, settings: ServerSettings) -> None:
        """Configure a reusable HTTP connection pool."""
        self._model: str | None = settings.model
        self._client = httpx.AsyncClient(
            base_url=settings.base_url,
            headers={"Authorization": f"Bearer {settings.api_key}"},
            timeout=settings.timeout,
        )

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

    @staticmethod
    def build_payload(
        *,
        state: str,
        instructions: str,
        labels: tuple[str, ...],
        model: str,
    ) -> DecisionRequest:
        """Build a choice request for a model served by the endpoint."""
        question = Choice(
            instructions=instructions,
            criteria=dict.fromkeys(labels),
        )

        return DecisionRequest(
            state=state,
            questions={"label": question},
            model=model,
        )

    @asafe
    async def choose(
        self,
        *,
        state: str,
        instructions: str,
        labels: tuple[str, ...],
    ) -> str:
        """Return the selected label or a failure for an invalid response."""
        model = (await self.model()).unwrap()
        payload = self.build_payload(
            state=state,
            instructions=instructions,
            labels=labels,
            model=model,
        )

        response = await self._client.post(
            "/v1/systemone",
            content=payload.model_dump_json(exclude_none=True),
            headers={"content-type": "application/json"},
        )

        response.raise_for_status()
        response = SystemOneResponse.model_validate_json(response.content)

        answer = response.choices["label"]
        if answer.choice not in labels:
            msg = "System One returned an invalid choice"
            raise ValueError(msg)

        return answer.choice

"""Chat completion against an OpenAI API."""

import json
from time import perf_counter
from typing import TYPE_CHECKING

from openai import AsyncOpenAI, omit

from eh30pt.logging import logger
from eh30pt.types import asafe

if TYPE_CHECKING:
    from collections.abc import Sequence

    from openai.types.chat import ChatCompletion, ChatCompletionMessageParam

    from eh30pt.settings import ServerSettings
    from eh30pt.types import FewShot


class OpenaiClient:
    """Stateful chat completion client for an OpenAI API server."""

    def __init__(
        self,
        settings: ServerSettings,
        seed: int | None = None,
    ) -> None:
        """Store the endpoint and decoding parameters, deferring model resolution."""
        self._client = AsyncOpenAI(
            base_url=settings.base_url,
            api_key=settings.api_key,
            timeout=settings.timeout,
        )

        self._model: str | None = None
        self._fmt = settings.format or omit
        self._seed = seed

    @asafe
    async def model(self) -> str:
        """Return the id of the model served at the endpoint."""
        if self._model is not None:
            return self._model

        logger.debug("listing models at '%s'", self._client.base_url)
        page = await self._client.models.list()
        if not page.data:
            msg = f"server at '{self._client.base_url}' serves no models"
            raise ValueError(msg)

        model = page.data[0].id.split("/")[-1]
        logger.debug("using model '%s'", model)

        self._model = model
        return model

    @staticmethod
    def build_conversation(
        user: str | None = None,
        system: str | None = None,
        examples: Sequence[FewShot] = (),
    ) -> list[ChatCompletionMessageParam]:
        """Build chat turns from system prompt, few-shot examples and user prompt."""
        conversation: list[ChatCompletionMessageParam] = []

        if system is not None:
            conversation.append({"role": "system", "content": system})

        for example in examples:
            conversation.append({"role": "user", "content": example.user})
            conversation.append({"role": "assistant", "content": example.label})

        if user is not None:
            conversation.append({"role": "user", "content": user})

        return conversation

    @staticmethod
    def _extract_content(completion: ChatCompletion) -> str:
        """Return the assistant content of the first choice in completion."""
        choices = completion.choices
        if not choices:
            msg = "server returned no choices"
            raise ValueError(msg)

        content = choices[0].message.content
        if content is None:
            msg = "server returned no content"
            raise ValueError(msg)

        return content

    @asafe
    async def complete(self, conversation: list[ChatCompletionMessageParam]) -> str:
        """Complete ready-made messages on the server."""
        logger.debug(
            "querying completion (messages=%d) (structured=%s)",
            len(conversation),
            self._fmt is not None,
        )

        model = await self.model()

        start = perf_counter()
        completion = await self._client.chat.completions.create(
            model=model.unwrap(),
            messages=conversation,
            response_format=self._fmt,
            seed=self._seed,
        )

        elapsed = perf_counter() - start
        logger.debug("received '%s' in %.0f ms", completion.id, 1000 * elapsed)

        content = self._extract_content(completion)
        if self._fmt is not omit:
            return json.loads(content)

        return content

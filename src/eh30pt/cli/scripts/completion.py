"""Query a llama.cpp server through its OpenAI-compatible chat endpoint."""

import json
from typing import TYPE_CHECKING

from openai import AsyncOpenAI, omit

from eh30pt.logging import logger

if TYPE_CHECKING:
    from openai.types.chat import ChatCompletion, ChatCompletionMessageParam
    from openai.types.chat.completion_create_params import ResponseFormatJSONSchema


def build_conversation(
    user_prompt: str,
    system_prompt: str | None,
) -> list[ChatCompletionMessageParam]:
    """Build the chat turns for user prompt, prepending system prompt when given."""
    conversation: list[ChatCompletionMessageParam] = []

    if system_prompt is not None:
        conversation.append({"role": "system", "content": system_prompt})

    conversation.append({"role": "user", "content": user_prompt})
    return conversation


def extract_content(completion: ChatCompletion) -> str:
    """Return the assistant content of the first choice in completion."""
    if not completion.choices:
        msg = "llama server returned no choices"
        raise ValueError(msg)

    message = completion.choices[0].message

    content = message.content
    if content is None:
        msg = "llama server returned no content"
        raise ValueError(msg)

    reasoning = getattr(message, "reasoning_content", None)
    if reasoning is not None:
        logger.debug("reasoning trace: '%s'", reasoning)
    else:
        logger.debug("model performed no reasoning")

    return content


def build_response_format(labels: tuple[str, ...]) -> ResponseFormatJSONSchema:
    """Build the response format constraining generation to labels."""
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "label",
            "schema": {
                "type": "object",
                "properties": {
                    "label": {"type": "string", "enum": list(labels)},
                },
                "required": ["label"],
            },
        },
    }


def extract_label(completion: ChatCompletion, labels: tuple[str, ...]) -> str:
    """Return the validated label of the first choice in completion."""
    content = extract_content(completion)

    try:
        decoded = json.loads(content)
    except json.JSONDecodeError as exc:
        msg = f"llama server returned invalid json: {content}"
        raise ValueError(msg) from exc

    if not isinstance(decoded, dict) or "label" not in decoded:
        msg = f"llama server returned no label: {content}"
        raise ValueError(msg)

    label = decoded["label"]
    if not isinstance(label, str) or label not in labels:
        msg = f"llama server returned unknown label: {label}"
        raise ValueError(msg)

    logger.debug("extracted label: %r", label)
    return label


def _client(base_url: str) -> AsyncOpenAI:
    return AsyncOpenAI(base_url=base_url, api_key="no-auth", timeout=300)


async def server_model(base_url: str) -> str:
    """Return the id of the model."""
    logger.debug("listing models on server")
    async with _client(base_url) as client:
        models = await client.models.list()

    if not models.data:
        msg = f"llama server at '{base_url}' serves no models"
        raise ValueError(msg)

    model = models.data[0].id
    logger.debug("using model '%s'", model.split("/")[-1])
    return model


async def complete(
    base_url: str,
    model: str,
    conversation: list[ChatCompletionMessageParam],
    *,
    labels: tuple[str, ...] | None = None,
    seed: int | None = None,
) -> str:
    """Complete ready-made messages on the server."""
    logger.debug("completing message")
    response_format = build_response_format(labels) if labels is not None else omit

    async with _client(base_url) as client:
        completion = await client.chat.completions.create(
            model=model,
            messages=conversation,
            response_format=response_format,
            seed=seed if seed is not None else omit,
        )

    logger.debug("received completion with %d choices", len(completion.choices))
    if labels is not None:
        return extract_label(completion, labels)

    return extract_content(completion)


async def main(
    user_prompt: str,
    system_prompt: str,
    base_url: str,
    *,
    labels: tuple[str, ...] | None = None,
    seed: int | None = None,
) -> None:
    """Complete prompt on the llama server."""
    logger.debug("completing prompt (%d chars)", len(user_prompt))

    model = await server_model(base_url)
    conversation = build_conversation(user_prompt, system_prompt)
    content = await complete(base_url, model, conversation, labels=labels, seed=seed)
    logger.info("completion: %s", content)

"""Query a llama.cpp server through its OpenAI-compatible chat endpoint."""

import json
from typing import Final, NotRequired, TypedDict, cast

import httpx

from eh30pt.logging import logger

_TIMEOUT: Final = 300.0
"""Seconds to wait for a completion, generation can take a while."""


class ChatMessage(TypedDict):
    """Single turn of a chat completion request."""

    role: str
    content: str


class ChatCompletionRequest(TypedDict):
    """Body of an OpenAI-style chat completion request."""

    messages: list[ChatMessage]
    max_tokens: int
    response_format: NotRequired[ResponseFormat]


class _LabelProperty(TypedDict):
    """Label enum of a classification response schema."""

    type: str
    enum: list[str]


class _LabelProperties(TypedDict):
    """Properties of a classification response schema."""

    label: _LabelProperty


class _LabelJsonSchema(TypedDict):
    """JSON schema of a classification response."""

    type: str
    properties: _LabelProperties
    required: list[str]


class _JsonSchemaSpec(TypedDict):
    """Named JSON schema of a structured response format."""

    name: str
    schema: _LabelJsonSchema


class ResponseFormat(TypedDict):
    """JSON-schema response format constraining generation to labels."""

    type: str
    json_schema: _JsonSchemaSpec


class _ResponseMessage(TypedDict):
    """Assistant message of a chat completion response."""

    role: str
    content: str


class _ResponseChoice(TypedDict):
    """Single completion choice of a chat completion response."""

    message: _ResponseMessage


class _ChatCompletion(TypedDict):
    """Subset of an OpenAI-style chat completion response."""

    choices: list[_ResponseChoice]


def build_messages(prompt: str, system: str | None) -> list[ChatMessage]:
    """
    Build the chat turns for `prompt`, prepending `system` when given.

    Parameters
    ----------
    prompt : str
        User prompt to send to the server.
    system : str | None
        System prompt steering the completion, omitted when None.

    Returns
    -------
    list[ChatMessage]
        Chat turns starting with the user prompt.

    """
    messages: list[ChatMessage] = []

    if system is not None:
        messages.append({"role": "system", "content": system})

    messages.append({"role": "user", "content": prompt})

    return messages


def extract_content(payload: _ChatCompletion) -> str:
    """
    Return the assistant content of the first choice in `payload`.

    Parameters
    ----------
    payload : _ChatCompletion
        Parsed chat completion response from the server.

    Returns
    -------
    str
        Assistant content of the first choice.

    Raises
    ------
    ValueError
        If the response contains no choices.

    """
    if not payload["choices"]:
        msg = "llama server returned no choices"
        raise ValueError(msg)

    return payload["choices"][0]["message"]["content"]


def build_response_format(labels: tuple[str, ...]) -> ResponseFormat:
    """
    Build the response format constraining generation to `labels`.

    Parameters
    ----------
    labels : tuple[str, ...]
        Allowed label values for the classification.

    Returns
    -------
    ResponseFormat
        JSON-schema response format with a label enum.

    """
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


def classification_prompt(labels: tuple[str, ...]) -> str:
    """
    Build the default system prompt for classifying into `labels`.

    Parameters
    ----------
    labels : tuple[str, ...]
        Allowed label values for the classification.

    Returns
    -------
    str
        System prompt describing the JSON label response.

    """
    joined = ", ".join(labels)

    return (
        "Classify the text. "
        'Respond with a JSON object like {"label": "<TAG>"} '
        f"where <TAG> is one of: {joined}."
    )


def extract_label(payload: _ChatCompletion, labels: tuple[str, ...]) -> str:
    """
    Return the validated label of the first choice in `payload`.

    Parameters
    ----------
    payload : _ChatCompletion
        Parsed chat completion response from the server.
    labels : tuple[str, ...]
        Allowed label values for the classification.

    Returns
    -------
    str
        Label of the first choice.

    Raises
    ------
    ValueError
        If the content is not JSON with a known `label` value.

    """
    content = extract_content(payload)

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

    return label


def main(
    prompt: str,
    *,
    base_url: str,
    system: str | None,
    max_tokens: int,
    labels: tuple[str, ...] | None = None,
) -> str:
    """
    Complete `prompt` on the llama server at `base_url`.

    Parameters
    ----------
    prompt : str
        User prompt to send to the server.
    base_url : str
        Base URL of the llama server.
    system : str | None
        System prompt steering the completion, a classification prompt
        over `labels` when None and `labels` are given.
    max_tokens : int
        Maximum number of tokens to generate.
    labels : tuple[str, ...] | None
        Allowed label values, constraining generation and validating
        the returned label when given.

    Returns
    -------
    str
        Label of the first choice when `labels` are given, otherwise
        the assistant content of the first choice.

    """
    if labels is not None and system is None:
        system = classification_prompt(labels)

    url = base_url.rstrip("/") + "/v1/chat/completions"
    body: ChatCompletionRequest = {
        "messages": build_messages(prompt, system),
        "max_tokens": max_tokens,
    }

    if labels is not None:
        body["response_format"] = build_response_format(labels)

    logger.info("querying '%s'", url)
    response = httpx.post(url, json=body, timeout=_TIMEOUT)
    response.raise_for_status()

    payload = cast("_ChatCompletion", response.json())

    if labels is not None:
        return extract_label(payload, labels)

    return extract_content(payload)

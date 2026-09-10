"""Query a server and print the completion."""

from typing import TYPE_CHECKING

from returns.result import Failure, Success

from eh30pt.logging import logger
from eh30pt.openai.complete import CompletionClient

if TYPE_CHECKING:
    from eh30pt.settings import ServerSettings


async def main(
    *,
    user_prompt: str,
    system_prompt: str,
    server: ServerSettings,
    seed: int | None = None,
) -> None:
    """Complete prompt on the server."""
    logger.debug("completing prompt (%d chars)", len(user_prompt))
    client = CompletionClient(server, seed)
    conversation = client.build_conversation(user_prompt, system_prompt)
    match await client.complete(conversation):
        case Success(content):
            logger.info("completion: %s", content)
        case Failure(exc):
            logger.error("completion failed: %s", exc)

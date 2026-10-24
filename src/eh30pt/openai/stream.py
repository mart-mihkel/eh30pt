"""Bounded-concurrency streaming over massive sample iterators."""

import asyncio
from dataclasses import dataclass
from itertools import islice
from typing import TYPE_CHECKING

from returns.result import Failure, Success

from eh30pt.logging import logger
from eh30pt.settings import MlflowSettings
from eh30pt.tracking.shards import flush_full_shards
from eh30pt.types import FewShot, GenerateRecord, Predictor

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Iterator, Sequence
    from pathlib import Path

    from openai.types.chat import ChatCompletionMessageParam
    from tqdm import tqdm

    from eh30pt.openai.complete import CompletionClient
    from eh30pt.types import ArtifactLogger


@dataclass(frozen=True)
class CompletionPredictor(Predictor[FewShot, str | None]):
    """Bind a completion client to its base conversation for streaming."""

    client: CompletionClient
    conversation: list[ChatCompletionMessageParam]

    async def predict(self, sample: FewShot) -> str | None:
        """Complete one streamed sample, returning None when invalid."""
        extended: list[ChatCompletionMessageParam] = [
            *self.conversation,
            {"role": "user", "content": sample.user},
        ]

        match await self.client.complete(extended):
            case Success(content):
                return content
            case Failure(exc) | exc:
                logger.error("completion failed: %s", exc)
                return None


async def map_concurrent[ItemT, ResultT](
    items: Sequence[ItemT],
    predictor: Predictor[ItemT, ResultT],
    concurrency: int,
) -> list[ResultT]:
    """Run predictor over items with bounded concurrency, preserving input order."""
    if concurrency < 1:
        msg = f"concurrency must be at least 1, got {concurrency}"
        raise ValueError(msg)

    semaphore = asyncio.Semaphore(concurrency)

    async def _run(item: ItemT) -> ResultT:
        async with semaphore:
            return await predictor.predict(item)

    return list(await asyncio.gather(*(_run(item) for item in items)))


async def batched_concurrent[ItemT, ResultT](
    stream: Iterator[ItemT],
    predictor: Predictor[ItemT, ResultT],
    concurrency: int,
) -> AsyncIterator[tuple[list[ItemT], list[ResultT]]]:
    """Yield ordered result batches for `stream` with bounded concurrency."""
    if concurrency < 1:
        msg = f"concurrency must be at least 1, got {concurrency}"
        raise ValueError(msg)

    while True:
        window = list(islice(stream, concurrency))
        if not window:
            return

        results = await map_concurrent(window, predictor, concurrency=concurrency)
        yield window, results


def maybe_log_invalid(invalid: int, n_seen: int) -> None:
    """Warn when invalid responses were marked as invalid."""
    if invalid:
        logger.warning("%d/%d invalid responses", invalid, n_seen)


def log_summary(
    bar: tqdm,
    verb: str,
    n_seen: int,
    invalid: int,
    n_shards: int | None = None,
) -> None:
    """Log a one-line throughput summary for a closed progress bar."""
    elapsed = bar.format_dict.get("elapsed") or 0.0
    rate = n_seen / elapsed if elapsed > 0 else 0.0
    shards = f" across {n_shards} shards" if n_shards is not None else ""
    logger.info(
        "%s %d samples (%d invalid) in %.1fs (%.1f/s)%s",
        verb,
        n_seen,
        invalid,
        elapsed,
        rate,
        shards,
    )


def append_prediction_window(
    batch: list[GenerateRecord],
    window: list[FewShot],
    results: list[str | None],
) -> int:
    """Append one ordered window to batch, returning its invalid count."""
    invalid = 0
    for sample, prediction in zip(window, results, strict=True):
        if sample.id is None:
            msg = "streamed samples must carry row ids"
            raise ValueError(msg)

        if prediction is None:
            invalid += 1

        record = GenerateRecord(
            id=sample.id,
            user=sample.user,
            prediction=prediction if prediction is not None else "<invalid>",
        )
        batch.append(record)

    return invalid


async def stream_predictions(
    samples: Iterator[FewShot],
    predictor: Predictor[FewShot, str | None],
    *,
    concurrency: int,
    shard_size: int,
    directory: Path | str,
    ctx: ArtifactLogger,
    batch: list[GenerateRecord],
    bar: tqdm,
    start_shard: int = 0,
) -> tuple[int, int, int]:
    """Stream samples concurrently into batch, returning (n_seen, invalid, n_shards)."""
    n_seen = 0
    invalid = 0
    n_shards = start_shard
    async for window, results in batched_concurrent(
        samples,
        predictor,
        concurrency=concurrency,
    ):
        invalid += append_prediction_window(batch, window, results)
        n_seen += len(window)
        n_shards = flush_full_shards(
            ctx,
            directory,
            MlflowSettings.predictions_path,
            n_shards,
            batch,
            shard_size=shard_size,
        )

        bar.update(len(window))

    return n_seen, invalid, n_shards

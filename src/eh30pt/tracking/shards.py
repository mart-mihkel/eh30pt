"""Sharded prediction artifact persistence."""

from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl

from eh30pt.logging import logger

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from typing import Any

    from eh30pt.types import ArtifactLogger


def write_shard[RecordT: Mapping[str, object] | tuple[Any, ...]](
    directory: Path | str,
    prefix: str,
    index: int,
    batch: Sequence[RecordT],
) -> Path:
    """Write one batch as a numbered Parquet shard."""
    logger.debug("writing temporary shard to '%s'", directory)
    shard = Path(directory) / f"{prefix}-{index:05d}.parquet"
    pl.DataFrame(list(batch)).write_parquet(shard)
    return shard


def log_shard[RecordT: Mapping[str, object] | tuple[Any, ...]](
    ctx: ArtifactLogger,
    directory: Path | str,
    prefix: str,
    index: int,
    batch: Sequence[RecordT],
) -> int:
    """Write batch as a shard, log it, and remove the local copy."""
    shard = write_shard(directory, prefix, index, batch)

    try:
        logger.debug("logging shard as mlflow artifact at '%s'", prefix)
        ctx.log_artifact(shard, artifact_path=prefix)
    finally:
        logger.debug("unlinking temporary shard at '%s'", directory)
        shard.unlink(missing_ok=True)

    return index + 1


def flush_full_shards[RecordT: Mapping[str, object] | tuple[Any, ...]](
    ctx: ArtifactLogger,
    directory: Path | str,
    prefix: str,
    index: int,
    batch: list[RecordT],
    *,
    shard_size: int,
) -> int:
    """Log every full shard in batch, keeping the remainder buffered."""
    while len(batch) >= shard_size:
        index = log_shard(ctx, directory, prefix, index, batch[:shard_size])
        del batch[:shard_size]

    return index

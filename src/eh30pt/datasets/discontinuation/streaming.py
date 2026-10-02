"""Shared Est-Health streaming for discontinuation datasets."""

from typing import TYPE_CHECKING

from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from eh30pt.esthealth.queries import (
    select_dg_events,
    select_observed_events,
    select_trigger_events,
)
from eh30pt.logging import logger

if TYPE_CHECKING:
    from collections.abc import Iterator

    from eh30pt.esthealth.queries import EventTextRow
    from eh30pt.types import DataSource


def _select_source(
    source: DataSource,
    session: Session,
    *,
    page_size: int,
    start_after: int | None,
) -> Iterator[EventTextRow]:
    """Yield (id, text) rows from the dg, trigger filter, or random sample."""
    if source == "drug-group":
        return iter(select_dg_events(session, start_after=start_after))

    if source == "observed":
        return select_observed_events(
            session,
            page_size=page_size,
            start_after=start_after,
        )

    if source == "trigger":
        return select_trigger_events(
            session,
            page_size=page_size,
            start_after=start_after,
        )

    msg = f"invalid discontinuation data source '{source}'"
    raise ValueError(msg)


def _stream_rows(
    url: str,
    source: DataSource,
    *,
    page_size: int,
    start_after: int | None,
    message: str,
) -> Iterator[dict[str, int | str]]:
    """Stream (id, text) rows with id above start_after, connecting lazily."""
    logger.debug("%s", message)
    engine = create_engine(url)

    try:
        with Session(engine) as session:
            skipped = 0
            for row in _select_source(
                source,
                session,
                page_size=page_size,
                start_after=start_after,
            ):
                if row.text is None:
                    skipped += 1
                    continue

                yield {"id": row.id, "text": row.text}

            if skipped:
                logger.debug("skipping %d textless events", skipped)
    finally:
        engine.dispose()

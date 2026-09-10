"""Est-Health queries."""

from typing import TYPE_CHECKING, NamedTuple, cast

from sqlalchemy import func, select

from eh30pt.esthealth.models import CdaDgEvent, CdaObservedEvent, CdaTriggerEvent
from eh30pt.logging import logger

if TYPE_CHECKING:
    from collections.abc import Iterator

    from sqlalchemy.orm import Session


class EventTextRow(NamedTuple):
    """Event id paired with its extracted document text."""

    id: int
    text: str | None


def _assert_page_size(page_size: int) -> None:
    """Raise ValueError when the page size is not positive."""
    if page_size < 1:
        msg = f"page_size must be positive, got {page_size}"
        raise ValueError(msg)


def select_observed_events(
    session: Session,
    *,
    page_size: int = 10000,
    limit: int | None = None,
    start_after: int | None = None,
) -> Iterator[EventTextRow]:
    """Yield random sample (id, text) rows with id above start_after."""
    _assert_page_size(page_size)

    last_id: int | None = start_after
    yielded = 0

    while True:
        if limit is not None and yielded >= limit:
            return

        batch_size = page_size if limit is None else min(page_size, limit - yielded)
        logger.debug("querying next page (%d events)", batch_size)

        stmt = (
            select(CdaObservedEvent.id, CdaObservedEvent.text)
            .order_by(CdaObservedEvent.id)
            .limit(batch_size)
        )

        if last_id is not None:
            stmt = stmt.where(CdaObservedEvent.id > last_id)

        page = session.execute(stmt).all()

        if not page:
            return

        for row in page:
            row_id = cast("int", row[0])
            row_text = cast("str | None", row[1])
            last_id = row_id
            yielded += 1

            yield EventTextRow(id=row_id, text=row_text)

        if len(page) < batch_size:
            return


def select_dg_events(
    session: Session,
    *,
    per_type: int = 80_000,
    start_after: int | None = None,
) -> list[EventTextRow]:
    """Return up to per_type (id, text) rows for each event type."""
    if per_type < 1:
        msg = f"per_type must be positive, got {per_type}"
        raise ValueError(msg)

    logger.debug("querying dg events (%d per type)", per_type)
    ranked = (
        func.row_number()
        .over(
            partition_by=CdaDgEvent.event_type,
            order_by=CdaDgEvent.id,
        )
        .label("rn")
    )

    inner = select(CdaDgEvent.id, CdaDgEvent.text, ranked)
    if start_after is not None:
        inner = inner.where(CdaDgEvent.id > start_after)

    sub = inner.subquery()
    stmt = select(sub.c.id, sub.c.text).where(sub.c.rn <= per_type).order_by(sub.c.id)
    page = session.execute(stmt).all()

    return [
        EventTextRow(id=cast("int", row[0]), text=cast("str | None", row[1]))
        for row in page
    ]


def select_trigger_events(
    session: Session,
    *,
    page_size: int = 10000,
    limit: int | None = None,
    start_after: int | None = None,
) -> Iterator[EventTextRow]:
    """Yield high-recall candidate (id, text) rows with id above start_after."""
    _assert_page_size(page_size)

    last_id: int | None = start_after
    yielded = 0

    while True:
        if limit is not None and yielded >= limit:
            return

        batch_size = page_size if limit is None else min(page_size, limit - yielded)
        logger.debug("querying next page (%d events)", batch_size)

        stmt = (
            select(CdaTriggerEvent.id, CdaTriggerEvent.text)
            .order_by(CdaTriggerEvent.id)
            .limit(batch_size)
        )

        if last_id is not None:
            stmt = stmt.where(CdaTriggerEvent.id > last_id)

        page = session.execute(stmt).all()

        if not page:
            return

        for row in page:
            row_id = cast("int", row[0])
            row_text = cast("str | None", row[1])
            last_id = row_id
            yielded += 1

            yield EventTextRow(id=row_id, text=row_text)

        if len(page) < batch_size:
            return

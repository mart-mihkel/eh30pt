"""Bounded Est-Health queries."""

from typing import TYPE_CHECKING, NamedTuple, cast

from sqlalchemy import select

from eh30pt.esthealth.models import CdaDgEvent, CdaObservedEvent, CdaTriggerEvent
from eh30pt.logging import logger

if TYPE_CHECKING:
    from sqlalchemy.orm import Session

    from eh30pt.types import DgEventType


class EventTextRow(NamedTuple):
    """Event id paired with its extracted document text."""

    id: int
    text: str | None


def _assert_limit(limit: int) -> None:
    if limit < 1:
        msg = f"limit must be positive, got {limit}"
        raise ValueError(msg)


def select_observed_events(
    session: Session,
    limit: int = 100_000,
) -> list[EventTextRow]:
    """Return a bounded set of observed events ordered by id."""
    _assert_limit(limit)
    logger.debug("querying up to %d observed events", limit)
    stmt = (
        select(CdaObservedEvent.id, CdaObservedEvent.text)
        .order_by(CdaObservedEvent.id)
        .limit(limit)
    )

    return [
        EventTextRow(cast("int", row[0]), cast("str | None", row[1]))
        for row in session.execute(stmt)
    ]


def select_dg_events(
    session: Session,
    event_type: DgEventType,
    limit: int = 100_000,
) -> list[EventTextRow]:
    """Return a bounded set of events of the specified type ordered by id."""
    _assert_limit(limit)
    logger.debug("querying up to %d %s events", limit, event_type)
    stmt = (
        select(CdaDgEvent.id, CdaDgEvent.text)
        .where(CdaDgEvent.event_type == event_type)
        .order_by(CdaDgEvent.id)
        .limit(limit)
    )

    return [
        EventTextRow(cast("int", row[0]), cast("str | None", row[1]))
        for row in session.execute(stmt)
    ]


def select_trigger_events(session: Session, limit: int = 100_000) -> list[EventTextRow]:
    """Return a bounded set of trigger events ordered by id."""
    _assert_limit(limit)
    logger.debug("querying up to %d trigger events", limit)
    stmt = (
        select(CdaTriggerEvent.id, CdaTriggerEvent.text)
        .order_by(CdaTriggerEvent.id)
        .limit(limit)
    )

    return [
        EventTextRow(cast("int", row[0]), cast("str | None", row[1]))
        for row in session.execute(stmt)
    ]

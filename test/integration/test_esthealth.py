"""Est-Health query integration tests against a seeded postgres."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

import pytest
from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import Session

from eh30pt.esthealth.models import CdaEvent
from eh30pt.esthealth.queries import (
    select_dg_events,
    select_observed_events,
    select_trigger_events,
)
from eh30pt.settings import PostgresSettings

if TYPE_CHECKING:
    from collections.abc import Iterator

    from eh30pt.types import DgEventType


@pytest.fixture
def session(database: None) -> Iterator[Session]:
    """Yield a session on the seeded postgres."""
    _ = database
    engine = create_engine(PostgresSettings().connection_url)

    try:
        with Session(engine) as session:
            yield session
    finally:
        engine.dispose()


def test_null_data_event_text_is_none(session: Session) -> None:
    """NULL data rows seeded in the db deserialize to None."""
    events = session.scalars(select(CdaEvent).where(CdaEvent.data.is_(None))).all()

    assert events

    for event in events:
        assert event.text is None


def test_seeded_event_text_deserializes(session: Session) -> None:
    """Non-NULL data rows seeded in the db deserialize via json_to_text."""
    events = session.scalars(
        select(CdaEvent)
        .where(CdaEvent.data.is_not(None))
        .order_by(CdaEvent.id)
        .limit(50)
    ).all()

    assert events

    for event in events:
        assert event.data is not None
        assert event.text is not None
        assert event.text.text == event.data["text"]


def test_discontinuation_seed_matches_query_pattern(session: Session) -> None:
    """Discontinuation rows seeded in the db match the default text pattern."""
    total = session.scalar(select(func.count()).select_from(CdaEvent))

    expected_total = 50000
    assert total == expected_total

    seeded = session.scalar(
        select(func.count()).select_from(CdaEvent).where(CdaEvent.table == "anamnesis")
    )

    expected_rows = 24000
    assert seeded == expected_rows

    epi_ids = session.scalars(select(CdaEvent.epi_id)).all()

    assert epi_ids
    assert all(
        epi_id is not None and re.fullmatch(r"[0-9]+", epi_id) is not None
        for epi_id in epi_ids
    )


def test_event_queries_are_bounded_and_ordered(session: Session) -> None:
    for query in (select_trigger_events, select_observed_events):
        events = query(session, limit=100)
        assert events
        assert [event.id for event in events] == sorted(event.id for event in events)
        assert query(session, limit=3) == events[:3]


def test_select_rejects_non_positive_limit(session: Session) -> None:
    import pytest

    for query in (select_trigger_events, select_observed_events):
        with pytest.raises(ValueError, match="limit must be positive"):
            query(session, limit=0)

    with pytest.raises(ValueError, match="limit must be positive"):
        select_dg_events(session, event_type="switch", limit=0)


def test_select_dg_events_returns_rows(session: Session) -> None:
    from eh30pt.esthealth.models import CdaDgEvent

    event_types: tuple[DgEventType, ...] = (
        "discontinuation",
        "switch",
        "usage_gap",
    )

    for event_type in event_types:
        rows = select_dg_events(session, event_type=event_type)

        assert rows
        assert [row.id for row in rows] == sorted(row.id for row in rows)
        assert select_dg_events(session, event_type=event_type, limit=1) == rows[:1]

        stored = session.scalars(
            select(CdaDgEvent.event_type).where(
                CdaDgEvent.id.in_([row.id for row in rows])
            )
        ).all()

        assert stored
        assert all(item == event_type for item in stored)

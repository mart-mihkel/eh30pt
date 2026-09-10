"""Est-Health queries."""

from typing import TYPE_CHECKING

from eh30pt.esthealth.models import Event

if TYPE_CHECKING:
    from collections.abc import AsyncIterator


async def select_events(
    *,
    page_size: int = 1000,
    using: str | None = None,
) -> AsyncIterator[Event]:
    """Yield events in primary-key order, fetching one page at a time."""
    if page_size < 1:
        msg = f"page_size must be positive, got {page_size}"
        raise ValueError(msg)

    offset = 0
    while True:
        page = (
            await Event.objects.order_by("id")
            .limit(page_size)
            .offset(offset)
            .filter(hidden=False)
            .all(using=using)
        )

        if not page:
            return

        for event in page:
            yield event

        offset += page_size

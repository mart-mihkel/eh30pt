"""Est-Health ORM models."""

from datetime import datetime  # noqa: TC003  # resolved at runtime by oxyde

from oxyde import Field, Model


class Event(Model):
    """Fact extraction event."""

    id: int = Field(db_pk=True)
    data: dict | None
    hidden: bool | None = Field(default=False)
    epi_id: str | None
    epi_type: str | None
    effective_time: datetime | None
    texts_id: int | None
    schema_: str | None = Field(db_column="schema")
    table: str | None
    field: str | None
    table_id: str | None
    header: str | None
    header_offset: int | None
    event_offset: int | None
    event_header_date: str | None
    doctor_code: str | None
    specialty: str | None
    specialty_code: str | None

    class Meta:
        """Model metadata."""

        is_table = True
        table_name = "events"
        schema = "cda_fact_extraction_202603"

"""Est-Health ORM models."""

import json
from datetime import date, datetime  # noqa: TC003
from typing import TYPE_CHECKING

from estnltk_core.converters.json_importer import json_to_text
from sqlalchemy import (
    JSON,
    BigInteger,
    Boolean,
    Date,
    DateTime,
    Index,
    Integer,
    Text,
    text,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

if TYPE_CHECKING:
    from estnltk import Text as EstnltkText


class Base(DeclarativeBase):
    """Declarative base for Est-Health tables."""


class CdaTriggerWord(Base):
    """Trigger word for selecting candidate event texts."""

    __tablename__ = "cda_trigger_words"
    __table_args__ = ({"schema": "user_mart_mihkel"},)

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    word: Mapped[str] = mapped_column(Text, nullable=False)
    subtype: Mapped[str] = mapped_column(Text, nullable=False)


class CdaEvent(Base):
    """Fact extraction event."""

    __tablename__ = "events"
    __table_args__ = (
        Index("ix_events_epi_id", "epi_id"),
        Index("ix_events_epi_type", "epi_type"),
        Index(
            "ix_events_layers_gin",
            text("to_tsvector('simple', data ->> 'layers')"),
            postgresql_using="gin",
        ),
        Index(
            "ix_events_relation_layers_gin",
            text("to_tsvector('simple', data ->> 'relation_layers')"),
            postgresql_using="gin",
        ),
        {"schema": "cda_fact_extraction_202603"},
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    data: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    """EstNLTK Text object."""

    hidden: Mapped[bool | None] = mapped_column(Boolean, default=False, nullable=True)
    epi_id: Mapped[str | None] = mapped_column(Text, nullable=True)
    epi_type: Mapped[str | None] = mapped_column(Text, nullable=True)
    effective_time: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    texts_id: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    schema_: Mapped[str | None] = mapped_column("schema", Text, nullable=True)
    table: Mapped[str | None] = mapped_column("table", Text, nullable=True)
    field: Mapped[str | None] = mapped_column(Text, nullable=True)
    table_id: Mapped[str | None] = mapped_column(Text, nullable=True)
    header: Mapped[str | None] = mapped_column(Text, nullable=True)
    header_offset: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    event_offset: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    event_header_date: Mapped[str | None] = mapped_column(Text, nullable=True)
    doctor_code: Mapped[str | None] = mapped_column(Text, nullable=True)
    specialty: Mapped[str | None] = mapped_column(Text, nullable=True)
    specialty_code: Mapped[str | None] = mapped_column(Text, nullable=True)

    @property
    def text(self) -> EstnltkText | None:
        """Deserialize jsonb data."""
        if self.data is None:
            return None

        return json_to_text(json.dumps(self.data))


class CdaObservedEvent(Base):
    """Random subsampled codrug events for discontinuation detection."""

    __tablename__ = "cda_observed_anamnesis_events_202603"
    __table_args__ = ({"schema": "user_mart_mihkel"},)

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    epi_id: Mapped[str | None] = mapped_column(Text, nullable=True)
    table: Mapped[str | None] = mapped_column("table", Text, nullable=True)
    field: Mapped[str | None] = mapped_column(Text, nullable=True)
    text: Mapped[str | None] = mapped_column(Text, nullable=True)
    word: Mapped[str | None] = mapped_column(Text, nullable=True)


class CdaTriggerEvent(Base):
    """High-recall discontinuation candidate with matched trigger words."""

    __tablename__ = "cda_observed_anamnesis_tw_events_202603"
    __table_args__ = ({"schema": "user_mart_mihkel"},)

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    epi_id: Mapped[str | None] = mapped_column(Text, nullable=True)
    table: Mapped[str | None] = mapped_column("table", Text, nullable=True)
    field: Mapped[str | None] = mapped_column(Text, nullable=True)
    text: Mapped[str | None] = mapped_column(Text, nullable=True)
    word: Mapped[str | None] = mapped_column(Text, nullable=True)


class CdaDgEvent(Base):
    """Distinct anamnesis event with its drug-related event type."""

    __tablename__ = "dg_cda_observed_anamnesis_events_202603"
    __table_args__ = ({"schema": "user_mart_mihkel"},)

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    epi_id: Mapped[str | None] = mapped_column(Text, nullable=True)
    table: Mapped[str | None] = mapped_column("table", Text, nullable=True)
    field: Mapped[str | None] = mapped_column(Text, nullable=True)
    text: Mapped[str | None] = mapped_column(Text, nullable=True)
    event_type: Mapped[str | None] = mapped_column(Text, nullable=True)


class CdaObservedDocument(Base):
    """DDI follow-up document observed for an episode."""

    __tablename__ = "ddi194_observed_cda_documents"
    __table_args__ = (
        Index("ix_ddi194_observed_cda_documents_document_offset", "document_offset"),
        Index("ix_ddi194_observed_cda_documents_epi_id", "epi_id"),
        Index(
            "ix_ddi194_observed_cda_documents_index_event_episode",
            "index_event",
            "episode",
        ),
        Index(
            "ix_ddi194_observed_cda_documents_index_drug_codrug",
            "index_drug",
            "codrug",
        ),
        Index("ix_ddi194_observed_cda_documents_patient_id", "patient_id"),
        Index("ix_ddi194_observed_cda_documents_start_date", "start_date"),
        {"schema": "user_sven"},
    )

    epi_id: Mapped[str] = mapped_column(Text, primary_key=True)
    epi_type: Mapped[str] = mapped_column(Text, primary_key=True)
    document_offset: Mapped[int] = mapped_column(Integer, primary_key=True)
    patient_id: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    index_drug: Mapped[str | None] = mapped_column(Text, nullable=True)
    codrug: Mapped[str | None] = mapped_column(Text, nullable=True)
    episode: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    index_event: Mapped[date | None] = mapped_column(Date, nullable=True)
    start_date: Mapped[date | None] = mapped_column(Date, nullable=True)
    time_source: Mapped[str | None] = mapped_column(Text, nullable=True)

"""Drug discontinuation reason extraction dataset."""

import json
from functools import partial
from textwrap import dedent
from typing import TYPE_CHECKING, ClassVar, TypedDict, cast

from datasets.iterable_dataset import IterableDataset
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from torch.utils.data import IterableDataset as TorchIterableDataset

from eh30pt.datasets.mixin import DatasetMixin
from eh30pt.esthealth.queries import (
    EventTextRow,
    select_dg_events,
    select_observed_events,
    select_trigger_events,
)
from eh30pt.logging import logger
from eh30pt.settings import PostgresSettings
from eh30pt.types import DataSplits, ExtractShot, FewShot

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

    from openai.types.shared_params import ResponseFormatJSONSchema
    from transformers import BatchEncoding, PreTrainedTokenizerFast

    from eh30pt.types import Architecture, DatasetTask, DataSource


class DiscontExtractRow(TypedDict):
    """One candidate discontinuation text for extraction."""

    id: int
    text: str


class DiscontExtract(DatasetMixin, IterableDataset, TorchIterableDataset):
    """Drug discontinuation reason extraction."""

    task: ClassVar[DatasetTask] = "extraction"

    system_prompt = dedent(
        """
        You are a clinical language model specialized in Estonian medical documentation. You are given clinical summary charts written by doctors about their patients who stopped taking their medication. Your task is to extract the reasons why the patients stopped taking the medications.
        Return your output as a JSON object with the keys: "phrase", "reason", "drug".

        Extraction rules:

        1. "phrase": Must include the full sentence(s) that should contain the fact that the patient stopped taking the medication and the reason, if they exist. Include multiple sentences if needed to preserve semantic clarity.
        2. "reason": Extract only the reason why the patient stopped taking medication (e.g., "vererõhk normaliseerus", "tekkis köha", "ei pidanud vajalikuks").
        3. "drug": Extract the name of the drug or its type/class only if explicitly mentioned (e.g., "Nebilet", "Amlodipin", "vererõhuravim", “diabeediravim”, “statiin”).

        Output format:

        Return only one JSON object per matched case.
        Do not paraphrase or clean the extracted texts; return the exact text as-is from the input.
        If no valid case is found in the input for one or more of the classes, leave the value as an empty string.
        """  # noqa: E501
    ).strip()

    extract_shots: ClassVar[list[ExtractShot]] = [
        ExtractShot(
            user="Patsient katkestas Nebileti tarvitamise, vererõhk normaliseerus.",
            extractions={
                "phrase": "Patsient katkestas Nebileti tarvitamise, vererõhk normaliseerus.",  # noqa: E501
                "reason": "vererõhk normaliseerus",
                "drug": "Nebilet",
            },
        ),
        ExtractShot(
            user="tekkis köha, enalapriil ära jäeti",
            extractions={
                "phrase": "tekkis köha, enalapriil ära jäeti",
                "reason": "tekkis köha",
                "drug": "enalapriil",
            },
        ),
        ExtractShot(
            user="kõrvaltoimed, jäeti ravim ära",
            extractions={
                "phrase": "Kõrvaltoimed, jäeti ravim ära",
                "reason": "kõrvaltoimed",
                "drug": "",
            },
        ),
        ExtractShot(
            user="Arst jättis statiini ära",
            extractions={
                "phrase": "Arst jättis statiini ära",
                "reason": "",
                "drug": "statiin",
            },
        ),
        ExtractShot(
            user="kaebusi ei ole, TVL (töö võimetus leht) lõpetada",
            extractions={"phrase": "", "reason": "", "drug": ""},
        ),
    ]

    few_shots: ClassVar[list[FewShot]] = [
        FewShot(
            user=shot.user,
            label=json.dumps(shot.extractions, ensure_ascii=False, sort_keys=True),
        )
        for shot in extract_shots
    ]

    def __init__(
        self,
        rows: Callable[[], Iterator[DiscontExtractRow]],
        tokenizer: PreTrainedTokenizerFast,
        arch: Architecture,
        n_shot: int = 0,
        n_virtual: int = 0,
    ) -> None:
        """Store the row factory with its tokenization settings."""
        self._rows = rows
        self._tokenizer = tokenizer
        self._arch = arch
        self._n_shot = n_shot
        self._n_virtual = n_virtual
        self._warned = False

    def __iter__(self) -> Iterator[dict[str, list[int] | bool | int]]:
        """Yield tokenized rows for the split."""
        for row in self._rows():
            encoded = self.tokenize(row, self._tokenizer, self._n_shot, self._n_virtual)
            yield cast("dict[str, list[int] | bool | int]", dict(encoded))

    @classmethod
    def response_format(cls) -> ResponseFormatJSONSchema:
        """Build the response format constraining generation to the extract fields."""
        return {
            "type": "json_schema",
            "json_schema": {
                "name": "extraction",
                "schema": {
                    "type": "object",
                    "required": ["phrase", "reason", "drug"],
                    "properties": {
                        "reason": {"type": "string", "maxLength": 1024},
                        "phrase": {"type": "string", "maxLength": 512},
                        "drug": {"type": "string", "maxLength": 256},
                    },
                },
            },
        }

    def samples(self) -> Iterator[FewShot]:
        """Yield prompt and label pairs for the split."""
        if not self._warned:
            logger.warning("discontinuation extract dataset has no labels")
            self._warned = True

        for row in self._rows():
            yield FewShot(user=row["text"], label="", id=row["id"])

    def tokenize(
        self,
        row: DiscontExtractRow,
        tokenizer: PreTrainedTokenizerFast,
        n_shot: int = 0,
        n_virtual: int = 0,
    ) -> BatchEncoding[list[int]]:
        """Tokenize one row with few-shot context."""
        if not self._warned:
            logger.warning("discontinuation extract dataset has no labels")
            self._warned = True

        if n_shot > len(self.few_shots):
            msg = "requested more few shot examples than defined"
            raise ValueError(msg)

        user_prompt = row["text"]
        few_shots = self.few_shots[:n_shot]
        label = ""

        return self._encode_example(
            arch=self._arch,
            tokenizer=tokenizer,
            user_prompt=user_prompt,
            label=label,
            few_shots=few_shots,
            n_virtual=n_virtual,
        )


def _select(source: DataSource, session: Session) -> Iterator[EventTextRow]:
    """Yield (id, text) rows from a dg event type, trigger filter, or sample."""
    if source == "discont":
        return iter(select_dg_events(session, event_type="discontinuation"))

    if source == "switch":
        return iter(select_dg_events(session, event_type="switch"))

    if source == "usage-gap":
        return iter(select_dg_events(session, event_type="usage_gap"))

    if source == "observed":
        return select_observed_events(session)

    if source == "trigger":
        return select_trigger_events(session)

    msg = f"invalid discontinuation data source '{source}'"
    raise ValueError(msg)


def _no_rows() -> Iterator[DiscontExtractRow]:
    """Yield no rows for the empty train and eval splits."""
    return iter(())


def _stream_rows(url: str, source: DataSource) -> Iterator[DiscontExtractRow]:
    """Stream (id, text) rows with id above start_after, connecting lazily."""
    engine = create_engine(url)
    with Session(engine) as session:
        skipped = 0
        for row in _select(source, session):
            if row.text is None:
                skipped += 1
                continue

            yield {"id": row.id, "text": row.text}

        if skipped:
            logger.debug("skipped %d textless events", skipped)


def load_discont_extract(
    tokenizer: PreTrainedTokenizerFast,
    arch: Architecture,
    n_shot: int = 0,
    n_virtual: int = 0,
    *,
    source: DataSource,
) -> DataSplits:
    """Load extract candidates as splits with only test populated."""
    if arch != "decoder":
        msg = f"discont-extract is an extraction dataset, '{arch}' is not supported"
        raise ValueError(msg)

    logger.debug("extract rows stream from Est-Health on first iteration")
    url = PostgresSettings().connection_url
    test_rows = partial(_stream_rows, url, source=source)

    return DataSplits(
        train=DiscontExtract(_no_rows, tokenizer, arch, n_shot, n_virtual),
        eval=DiscontExtract(_no_rows, tokenizer, arch, n_shot, n_virtual),
        test=DiscontExtract(test_rows, tokenizer, arch, n_shot, n_virtual),
    )

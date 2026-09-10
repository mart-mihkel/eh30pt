"""Discontinuation reason and decision datasets loading, prompting, and tokenization."""

import json
from functools import partial
from textwrap import dedent
from typing import TYPE_CHECKING, ClassVar, Literal, TypedDict, cast

from datasets.iterable_dataset import IterableDataset
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from torch.utils.data import IterableDataset as TorchIterableDataset

from eh30pt.datasets.mixin import DatasetMixin
from eh30pt.esthealth.queries import (
    select_dg_events,
    select_observed_events,
    select_trigger_events,
)
from eh30pt.logging import logger
from eh30pt.settings import PostgresSettings
from eh30pt.types import DataSplits, ExtractShot, FewShot

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Mapping

    from openai.types.shared_params import ResponseFormatJSONSchema
    from transformers import BatchEncoding, PreTrainedTokenizerFast

    from eh30pt.esthealth.queries import EventTextRow
    from eh30pt.types import Architecture, DatasetTask, DataSource


type DiscontClassifierLabel = Literal[
    "Adverse reactions",
    "Treatment success",
    "Treatment inefficacy",
    "Contraindication",
    "Non-medical reasons",
    "Other",
    "Indeterminate",
]


class DiscontClassifierRow(TypedDict):
    """One labeled discontinuation reason."""

    id: int
    text: str


class DiscontClassifier(DatasetMixin, IterableDataset, TorchIterableDataset):
    """Drug discontinuation reason classification."""

    id2label: ClassVar[Mapping[int, DiscontClassifierLabel]] = {
        0: "Adverse reactions",
        1: "Treatment success",
        2: "Treatment inefficacy",
        3: "Contraindication",
        4: "Non-medical reasons",
        5: "Other",
        6: "Indeterminate",
    }

    label2id: ClassVar[Mapping[DiscontClassifierLabel, int]] = {
        "Adverse reactions": 0,
        "Treatment success": 1,
        "Treatment inefficacy": 2,
        "Contraindication": 3,
        "Non-medical reasons": 4,
        "Other": 5,
        "Indeterminate": 6,
    }

    system_prompt = dedent(
        """
        You are a helpful assistant that reads Estonian electronic health records written by doctors about drug discontinuation for their patients on medication. For each input, classify the reason why the patient stopped taking the medication.

        Your input is the extracted reason for stopping the drug.

        Here are the categories you must use:

        1. Adverse reactions: Discontinuation due to adverse side effects, allergic reactions, or negative interactions with other medications.
        2. Treatment success: Discontinuation due to successful treatment completion or sufficient improvement in health.
        3. Treatment inefficacy: Perceived ineffectiveness of the treatment or loss of belief in its efficacy.
        4. Contraindication: Discontinuation due to the emergence or discovery of a medical condition or risk factor that makes the continued use of the treatment unsafe or inappropriate (e.g. "vastunäidustatud").
        5. Non-medical reasons: Discontinuation due to factor unrelated to the patient's health or the treatment's medical effects, such as financial constraints, access issues, personal choice, cultural beliefs, or social circumstances.
        6. Other: Other medical reasons not explicitly covered by the other categories.
        7. Indeterminate: Unclear or unspecified reasons for discontinuation.

        Only choose one category per input. If the reason is ambiguous or not clearly stated, select "Indeterminate". Respond only with the category name.
        """  # noqa: E501
    ).strip()

    few_shots: ClassVar[list[FewShot]] = [
        FewShot(user="tekkis tugev iiveldus ja peapööritus", label="Adverse reactions"),
        FewShot(user="vererõhk püsivalt normis", label="Treatment success"),
        FewShot(user="valu ei leevenenud, ebaefektiivne", label="Treatment inefficacy"),
        FewShot(user="raseduse tõttu vastunäidustatud", label="Contraindication"),
        FewShot(user="jättis rohu ostmata, raha pole", label="Non-medical reasons"),
        FewShot(user="ravim ära jäetud", label="Indeterminate"),
        FewShot(user="patsient viidi üle insuliinravile", label="Other"),
    ]

    def __init__(
        self,
        rows: Callable[[], Iterator[DiscontClassifierRow]],
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

    def samples(self) -> Iterator[FewShot]:
        """Yield prompt and label pairs for the split."""
        if not self._warned:
            logger.warning("discontinuation classifier dataset has no labels")
            self._warned = True

        for row in self._rows():
            yield FewShot(user=row["text"], label="", id=row["id"])

    def tokenize(
        self,
        row: DiscontClassifierRow,
        tokenizer: PreTrainedTokenizerFast,
        n_shot: int = 0,
        n_virtual: int = 0,
    ) -> BatchEncoding[list[int]]:
        """Tokenize one row with few-shot context."""
        if not self._warned:
            logger.warning("discontinuation classifier dataset has no labels")
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


def _select_source(
    source: DataSource,
    session: Session,
    *,
    page_size: int,
    start_after: int | None,
) -> Iterator[EventTextRow]:
    """Yield (id, text) rows from the dg, trigger filter, or random sample."""
    if source == "dg":
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


def _no_cls_rows() -> Iterator[DiscontClassifierRow]:
    """Yield no rows for the empty train and eval splits."""
    return iter(())


def _stream_classifier_rows(
    url: str,
    source: DataSource,
    *,
    page_size: int,
    start_after: int | None = None,
) -> Iterator[DiscontClassifierRow]:
    """Stream discontinuation rows with id above start_after, connecting lazily."""
    logger.debug("streaming discontinuation events")
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


def load_discont_classifier(
    tokenizer: PreTrainedTokenizerFast,
    arch: Architecture,
    n_shot: int = 0,
    n_virtual: int = 0,
    *,
    page_size: int = 10000,
    source: DataSource = "trigger",
    start_after: int | None = None,
) -> DataSplits:
    """Load discontinuation candidates as splits with only test populated."""
    logger.debug("discontinuation rows stream from Est-Health on first iteration")
    url = PostgresSettings().connection_url
    test_rows = partial(
        _stream_classifier_rows,
        url,
        page_size=page_size,
        source=source,
        start_after=start_after,
    )

    return DataSplits(
        train=DiscontClassifier(_no_cls_rows, tokenizer, arch, n_shot, n_virtual),
        eval=DiscontClassifier(_no_cls_rows, tokenizer, arch, n_shot, n_virtual),
        test=DiscontClassifier(test_rows, tokenizer, arch, n_shot, n_virtual),
    )


type DeciderLabel = Literal[
    "Doctor",
    "Patient",
    "Unspecified",
]


class DeciderRow(TypedDict):
    """One candidate discontinuation text."""

    id: int
    text: str


class DiscontDecider(DatasetMixin, IterableDataset, TorchIterableDataset):
    """Drug discontinuation decision attribution."""

    id2label: ClassVar[Mapping[int, DeciderLabel]] = {
        0: "Doctor",
        1: "Patient",
        2: "Unspecified",
    }

    label2id: ClassVar[Mapping[DeciderLabel, int]] = {
        "Doctor": 0,
        "Patient": 1,
        "Unspecified": 2,
    }

    system_prompt = dedent(
        """
        You are a helpful assistant that reads Estonian electronic health records written by doctors about drug discontinuation for their patients on medication. For each input, determine who made the decision to stop the use of a medication: the patient or the doctor.
        Your input is the extracted phrase about stopping the drug and the full anamnesis for context.

        Here are the categories you must use:

        1. Doctor: The doctor made the decision that the patient should stop the medications.
        2. Patient: The patient stopped taking the medications without a clear signal from the doctor.
        3. Unspecified: Based on the text, it is unclear who made the decision.

        Only choose one category per input. Respond only with the category name.
        """  # noqa: E501
    ).strip()

    few_shots: ClassVar[list[FewShot]] = [
        FewShot(user="katkestas, iiveldus ja peavalu", label="Patient"),
        FewShot(user="katkestas losartaani, operatsioonieelse pausi", label="Doctor"),
        FewShot(user="kolesterool olnud normis", label="Unspecified"),
        FewShot(user="patsient jättis rohu võtmata", label="Patient"),
        FewShot(user="peatatud kõrgenenud maksaproovide tõttu", label="Doctor"),
    ]

    def __init__(
        self,
        rows: Callable[[], Iterator[DeciderRow]],
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

    def samples(self) -> Iterator[FewShot]:
        """Yield prompt and label pairs for the split."""
        if not self._warned:
            logger.warning("discontinuation decider dataset has no labels")
            self._warned = True

        for row in self._rows():
            yield FewShot(user=row["text"], label="", id=row["id"])

    def tokenize(
        self,
        row: DeciderRow,
        tokenizer: PreTrainedTokenizerFast,
        n_shot: int = 0,
        n_virtual: int = 0,
    ) -> BatchEncoding[list[int]]:
        """Tokenize one row with few-shot context."""
        if not self._warned:
            logger.warning("discontinuation decider dataset has no labels")
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


def _no_decider_rows() -> Iterator[DeciderRow]:
    """Yield no rows for the empty train and eval splits."""
    return iter(())


def _stream_decider_rows(
    url: str,
    source: DataSource,
    *,
    page_size: int,
    start_after: int | None = None,
) -> Iterator[DeciderRow]:
    """Stream decider rows with id above start_after, connecting lazily."""
    logger.debug("streaming high-recall discontinuation events")
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


def load_discont_decider(
    tokenizer: PreTrainedTokenizerFast,
    arch: Architecture,
    n_shot: int = 0,
    n_virtual: int = 0,
    *,
    page_size: int = 10000,
    source: DataSource = "trigger",
    start_after: int | None = None,
) -> DataSplits:
    """Load decider candidates as splits with only test populated."""
    logger.debug("decider rows stream from Est-Health on first iteration")
    url = PostgresSettings().connection_url
    test_rows = partial(
        _stream_decider_rows,
        url,
        page_size=page_size,
        source=source,
        start_after=start_after,
    )

    return DataSplits(
        train=DiscontDecider(_no_decider_rows, tokenizer, arch, n_shot, n_virtual),
        eval=DiscontDecider(_no_decider_rows, tokenizer, arch, n_shot, n_virtual),
        test=DiscontDecider(test_rows, tokenizer, arch, n_shot, n_virtual),
    )


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
            user="kaebusi ei ole",
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


def _no_extract_rows() -> Iterator[DiscontExtractRow]:
    """Yield no rows for the empty train and eval splits."""
    return iter(())


def _stream_extract_rows(
    url: str,
    page_size: int,
    source: DataSource,
    *,
    start_after: int | None = None,
) -> Iterator[DiscontExtractRow]:
    """Stream extract rows with id above start_after, connecting lazily."""
    logger.debug("streaming high-recall discontinuation events")
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


def load_discont_extract(
    tokenizer: PreTrainedTokenizerFast,
    arch: Architecture,
    n_shot: int = 0,
    n_virtual: int = 0,
    *,
    page_size: int = 1000,
    source: DataSource = "trigger",
    start_after: int | None = None,
) -> DataSplits:
    """Load extract candidates as splits with only test populated."""
    if arch != "decoder":
        msg = f"discont-extract is an extraction dataset, '{arch}' is not supported"
        raise ValueError(msg)

    logger.debug("extract rows stream from Est-Health on first iteration")
    url = PostgresSettings().connection_url
    test_rows = partial(
        _stream_extract_rows,
        url,
        page_size=page_size,
        source=source,
        start_after=start_after,
    )

    return DataSplits(
        train=DiscontExtract(_no_extract_rows, tokenizer, arch, n_shot, n_virtual),
        eval=DiscontExtract(_no_extract_rows, tokenizer, arch, n_shot, n_virtual),
        test=DiscontExtract(test_rows, tokenizer, arch, n_shot, n_virtual),
    )

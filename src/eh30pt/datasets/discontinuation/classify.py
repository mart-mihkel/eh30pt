"""Drug discontinuation reason classification dataset."""

from collections import Counter
from textwrap import dedent
from typing import (
    TYPE_CHECKING,
    ClassVar,
    TypedDict,
    cast,
)

import polars as pl
from datasets.iterable_dataset import IterableDataset
from torch.utils.data import IterableDataset as TorchIterableDataset

from eh30pt.datasets.mixin import DatasetMixin
from eh30pt.labeling.loaders.extract import ExtractionFrameSchema, stratify
from eh30pt.logging import format_counts, logger
from eh30pt.types import DataSplits, DiscontClassifierLabel, FewShot

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping, Sequence
    from pathlib import Path

    from transformers import BatchEncoding, PreTrainedTokenizerFast

    from eh30pt.types import Architecture


class DiscontClassifierRow(TypedDict):
    """One discontinuation event carrying one of the ten classifier labels."""

    id: int
    text: str
    phrase: str
    drug: str
    label: str


class DiscontClassifier(DatasetMixin, IterableDataset, TorchIterableDataset):
    """Drug discontinuation reason classification."""

    id2label: ClassVar[Mapping[int, DiscontClassifierLabel]] = {
        0: "Adverse reactions",
        1: "Treatment success",
        2: "Treatment inefficacy",
        3: "Contraindication/unsafe co-usage",
        4: "Other",
        5: "Adverse reaction without discontinuation",
        6: "Dosage change",
        7: "Other non-discontinuation",
        8: "Incomplete phrase",
        9: "Incorrect phrase",
    }

    label2id: ClassVar[Mapping[str, int]] = {
        "Adverse reactions": 0,
        "Treatment success": 1,
        "Treatment inefficacy": 2,
        "Contraindication/unsafe co-usage": 3,
        "Other": 4,
        "Adverse reaction without discontinuation": 5,
        "Dosage change": 6,
        "Other non-discontinuation": 7,
        "Incomplete phrase": 8,
        "Incorrect phrase": 9,
    }

    system_prompt = dedent(
        """
        You are a helpful assistant that reads Estonian electronic health records written by doctors about drug discontinuation for their patients on medication. For each input, classify the reason why the patient stopped taking the medication.

        Your input is the clinical text together with the extracted phrase describing the event and the drug in question.

        Here are the categories you must use:

        1. Adverse reactions: Discontinuation due to adverse side effects, allergic reactions, or negative interactions with other medications.
        2. Treatment success: Discontinuation due to successful treatment completion or sufficient improvement in health.
        3. Treatment inefficacy: Perceived ineffectiveness of the treatment or loss of belief in its efficacy.
        4. Contraindication/unsafe co-usage: Discontinuation due to the emergence or discovery of a medical condition or risk factor that makes the continued use of the treatment unsafe or inappropriate (e.g. "vastunäidustatud").
        5. Other: Other medical reasons not explicitly covered by the other categories.
        6. Adverse reaction without discontinuation: Adverse side effects, allergic reactions, or negative interactions where treatment continued.
        7. Dosage change: The drug dose was increased or decreased rather than stopped.
        8. Other non-discontinuation: An event that is neither a discontinuation nor any of the reasons above.
        9. Incomplete phrase: The extracted phrase is too short to decide the event alone; flag it for expansion instead of classifying.
        10. Incorrect phrase: The extracted phrase is noise or a false positive; the real description, if any, is elsewhere.

        Only choose one category per input. If the reason is ambiguous or fits none of the categories above, select "Other". Respond only with the category name.
        """  # noqa: E501
    ).strip()

    few_shots: ClassVar[list[FewShot]] = [
        FewShot(user="ei talu", label="Adverse reactions"),
        FewShot(user="vererõhk püsivalt normis", label="Treatment success"),
        FewShot(user="valu ei leevenenud, ebaefektiivne", label="Treatment inefficacy"),
        FewShot(
            user="soovib rasestuda, katkestab ravi",
            label="Contraindication/unsafe co-usage",
        ),
        FewShot(user="patsient viidi üle insuliinravile", label="Other"),
        FewShot(
            user="iiveldus tekkis, aga ravi jätkub",
            label="Adverse reaction without discontinuation",
        ),
        FewShot(
            user="vähendasime statiini doosi",
            label="Dosage change",
        ),
        FewShot(
            user="patsient jätkab ravi muutuseta",
            label="Other non-discontinuation",
        ),
        FewShot(
            user="katkestas",
            label="Incomplete phrase",
        ),
        FewShot(
            user="ilm on ilus",
            label="Incorrect phrase",
        ),
    ]

    @staticmethod
    def _format_user_prompt(text: str, phrase: str, drug: str) -> str:
        """Format clinical text with its phrase and drug as a prompt."""
        return dedent(f"""
            Text: {text}
            Phrase: {phrase}
            Drug: {drug}
            Category:
        """).strip()

    def __init__(
        self,
        rows: Sequence[DiscontClassifierRow],
        tokenizer: PreTrainedTokenizerFast,
        arch: Architecture,
        n_shot: int = 0,
        n_virtual: int = 0,
    ) -> None:
        """Store the split rows with their tokenization settings."""
        self._rows = rows
        self._tokenizer = tokenizer
        self._arch = arch
        self._n_shot = n_shot
        self._n_virtual = n_virtual

    def __iter__(self) -> Iterator[dict[str, list[int] | bool | int]]:
        """Yield tokenized rows for the split."""
        for row in self._rows:
            encoded = self.tokenize(row, self._tokenizer, self._n_shot, self._n_virtual)
            yield cast("dict[str, list[int] | bool | int]", dict(encoded))

    def samples(self) -> Iterator[FewShot]:
        """Yield prompt and label pairs for the split."""
        for row in self._rows:
            yield FewShot(
                user=self._format_user(row),
                label=row.get("label", ""),
                id=row["id"],
            )

    def tokenize(
        self,
        row: DiscontClassifierRow,
        tokenizer: PreTrainedTokenizerFast,
        n_shot: int = 0,
        n_virtual: int = 0,
    ) -> BatchEncoding[list[int]]:
        """Tokenize one row with few-shot context."""
        if n_shot > len(self.few_shots):
            msg = "requested more few shot examples than defined"
            raise ValueError(msg)

        few_shots = self.few_shots[:n_shot]

        return self._encode_example(
            arch=self._arch,
            tokenizer=tokenizer,
            user_prompt=self._format_user(row),
            label=row.get("label", ""),
            few_shots=few_shots,
            n_virtual=n_virtual,
        )

    @classmethod
    def _format_user(cls, row: DiscontClassifierRow) -> str:
        """
        Format a row, falling back to bare text without extracted fields.

        Event rows streamed for evaluation predate phrase and drug extraction,
        so only labeled rows carry them.
        """
        phrase = row.get("phrase", "")
        drug = row.get("drug", "")
        if not phrase or not drug:
            return row["text"]

        return cls._format_user_prompt(row["text"], phrase, drug)


def _rows(frame: pl.DataFrame) -> list[DiscontClassifierRow]:
    """Return each row of a split frame as a typed classifier row."""
    return [
        {
            "id": row["task_id"],
            "text": row["text"],
            "phrase": row["phrase"],
            "drug": row["drug"],
            "label": row["label"],
        }
        for row in frame.to_dicts()
    ]


def load_discont_classifier(
    tokenizer: PreTrainedTokenizerFast,
    arch: Architecture,
    n_shot: int = 0,
    n_virtual: int = 0,
    *,
    data_path: Path,
    seed: int,
) -> DataSplits:
    """Load the exported Label Studio rows as train, eval and test splits."""
    df = pl.read_parquet(data_path)
    df = ExtractionFrameSchema.validate(df)
    train, eval_, test = stratify(df, on="label", ratios=(0.7, 0.2, 0.1), seed=seed)
    logger.debug(
        "classifier splits train=(%s) eval=(%s) test=(%s)",
        format_counts(Counter(train["label"])),
        format_counts(Counter(eval_["label"])),
        format_counts(Counter(test["label"])),
    )

    return DataSplits(
        train=DiscontClassifier(_rows(train), tokenizer, arch, n_shot, n_virtual),
        eval=DiscontClassifier(_rows(eval_), tokenizer, arch, n_shot, n_virtual),
        test=DiscontClassifier(_rows(test), tokenizer, arch, n_shot, n_virtual),
    )

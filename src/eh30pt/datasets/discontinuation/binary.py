"""Binary drug discontinuation classification dataset."""

from collections import Counter
from textwrap import dedent
from typing import (
    TYPE_CHECKING,
    ClassVar,
    Literal,
    TypedDict,
    cast,
)

import polars as pl
from datasets.iterable_dataset import IterableDataset
from torch.utils.data import IterableDataset as TorchIterableDataset

from eh30pt.datasets.mixin import DatasetMixin
from eh30pt.labeling.loaders.extract import ExtractionFrameSchema, stratify
from eh30pt.logging import format_counts, logger
from eh30pt.types import DataSplits, FewShot

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping, Sequence
    from pathlib import Path

    from transformers import BatchEncoding, PreTrainedTokenizerFast

    from eh30pt.types import Architecture


type DiscontBinaryLabel = Literal["yes", "no"]


class DiscontBinaryRow(TypedDict):
    """One discontinuation event answered yes or no."""

    id: int
    text: str
    phrase: str
    drug: str
    label: str


class DiscontBinary(DatasetMixin, IterableDataset, TorchIterableDataset):
    """Whether an event discontinued the medication, whatever the reason."""

    id2label: ClassVar[Mapping[int, DiscontBinaryLabel]] = {
        0: "yes",
        1: "no",
    }

    label2id: ClassVar[Mapping[str, int]] = {
        "yes": 0,
        "no": 1,
    }

    reason2label: ClassVar[Mapping[str, DiscontBinaryLabel]] = {
        "Adverse reactions": "yes",
        "Treatment success": "yes",
        "Treatment inefficacy": "yes",
        "Contraindication/unsafe co-usage": "yes",
        "Other": "yes",
        "Adverse reaction without discontinuation": "no",
        "Dosage change": "no",
        "Other non-discontinuation": "no",
        "Incomplete phrase": "no",
        "Incorrect phrase": "no",
    }

    system_prompt = dedent(
        """
        You are a helpful assistant that reads Estonian electronic health records written by doctors about drug discontinuation for their patients on medication. For each input, decide whether the event discontinued the medication or not.
        Your input is the clinical text together with the extracted phrase describing the event and the drug in question.

        Answer "yes" when the patient stopped taking the medication, whatever the reason: adverse reactions, completed treatment, perceived ineffectiveness, a contraindication, or another medical or non-medical reason.
        Answer "no" otherwise.

        Respond only with "yes" or "no".
        """  # noqa: E501
    ).strip()

    few_shots: ClassVar[list[FewShot]] = [
        FewShot(user="ei talu", label="yes"),
        FewShot(user="vererõhk püsivalt normis", label="yes"),
        FewShot(user="valu ei leevenenud, ebaefektiivne", label="yes"),
        FewShot(user="soovib rasestuda, katkestab ravi", label="yes"),
        FewShot(user="patsient viidi üle insuliinravile", label="yes"),
        FewShot(user="iiveldus tekkis, aga ravi jätkub", label="no"),
        FewShot(user="vähendasime statiini doosi", label="no"),
        FewShot(user="patsient jätkab ravi muutuseta", label="no"),
        FewShot(user="katkestas", label="no"),
        FewShot(user="ilm on ilus", label="no"),
    ]

    @staticmethod
    def _format_user_prompt(text: str, phrase: str, drug: str) -> str:
        """Format clinical text with its phrase and drug as a prompt."""
        return dedent(f"""
            Text: {text}
            Phrase: {phrase}
            Drug: {drug}
            Discontinuation:
        """).strip()

    def __init__(
        self,
        rows: Sequence[DiscontBinaryRow],
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
        row: DiscontBinaryRow,
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
    def _format_user(cls, row: DiscontBinaryRow) -> str:
        """Format a row, falling back to bare text without extracted fields."""
        phrase = row.get("phrase", "")
        drug = row.get("drug", "")
        if not phrase or not drug:
            return row["text"]

        return cls._format_user_prompt(row["text"], phrase, drug)


def _map(df: pl.DataFrame) -> pl.DataFrame:
    """Fold each reason label down to the binary answer it implies."""
    return df.with_columns(
        pl.col("label").replace_strict(DiscontBinary.reason2label, default=None),
    ).drop_nulls("label")


def _rows(frame: pl.DataFrame) -> list[DiscontBinaryRow]:
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


def load_discont_binary(
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
    train, eval_, test = (
        _map(frame)
        for frame in stratify(df, on="label", ratios=(0.7, 0.2, 0.1), seed=seed)
    )

    logger.debug(
        "binary splits train=(%s) eval=(%s) test=(%s)",
        format_counts(Counter(train["label"])),
        format_counts(Counter(eval_["label"])),
        format_counts(Counter(test["label"])),
    )

    return DataSplits(
        train=DiscontBinary(_rows(train), tokenizer, arch, n_shot, n_virtual),
        eval=DiscontBinary(_rows(eval_), tokenizer, arch, n_shot, n_virtual),
        test=DiscontBinary(_rows(test), tokenizer, arch, n_shot, n_virtual),
    )

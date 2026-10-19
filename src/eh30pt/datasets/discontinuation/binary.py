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


type DiscontBinaryLabel = Literal["jah", "ei"]


class DiscontBinaryRow(TypedDict):
    """One discontinuation event answered jah or ei."""

    id: int
    text: str
    phrase: str
    drug: str
    label: str


class DiscontBinary(DatasetMixin, IterableDataset, TorchIterableDataset):
    """Whether an event discontinued the medication, whatever the reason."""

    id2label: ClassVar[Mapping[int, str]] = {0: "jah", 1: "ei"}
    label2id: ClassVar[Mapping[str, int]] = {"jah": 0, "ei": 1}
    reason2label: ClassVar[Mapping[str, DiscontBinaryLabel]] = {
        "Adverse reactions": "jah",
        "Treatment success": "jah",
        "Treatment inefficacy": "jah",
        "Contraindication/unsafe co-usage": "jah",
        "Other": "jah",
        "Adverse reaction without discontinuation": "ei",
        "Dosage change": "ei",
        "Other non-discontinuation": "ei",
    }

    system_prompt = dedent(
        """
        Tuvasta ravimi tarvitamise lõpetamine anamneesi ja fraasi põhjal.

        - Anamnees: arsti sissekanne patsiendi terviseseisundi ja ravi kohta.
        - Fraas: tekstist eraldatud fraas, mis kirjeldab ravimiga seotud sündmust.
        - Ravim: ravim, mille kasutamise lõpetamist tuleb hinnata.

        Kas fraasi põhjal saab õelda, et patsient on lõpetanud ravimi tarvitamise? (jah/ei)
        """  # noqa: E501
    ).strip()

    encoder_system_prompt = "Otsusta, kas patsient lõpetas ravimi kasutamise."

    @staticmethod
    def _format_user_prompt(text: str, phrase: str, drug: str) -> str:
        """Format clinical text with its phrase and drug as a prompt."""
        return dedent(f"""
            Anamnees: {text}
            Fraas: {phrase}
            Ravim: {drug}
        """).strip()

    few_shots: ClassVar[list[FewShot]] = [
        FewShot(
            user=_format_user_prompt(
                "Patsient lõpetas enalapriili kasutamise, kuna ei talunud ravimit, tekkis kuiv köha ja lööve.",  # noqa: E501
                "lõpetas enalapriili kasutamise, kuna ei talunud ravimit",
                "enalapriil",
            ),
            label="jah",
        ),
        FewShot(
            user=_format_user_prompt(
                "Vererõhk on püsivalt normis, lõpetas nebivolooli kasutamise",
                "Vererõhk on püsivalt normis, lõpetas nebivolooli kasutamise",
                "nebivolool",
            ),
            label="jah",
        ),
        FewShot(
            user=_format_user_prompt(
                "Valu ei leevenenud, ibuprofeen osutus ebaefektiivseks, ravi lõpetati",
                "valu ei leevenenud, ibuprofeen osutus ebaefektiivseks",
                "ibuprofeen",
            ),
            label="jah",
        ),
        FewShot(
            user=_format_user_prompt(
                "Patsient soovib rasestuda, katkestab statiiniravi",
                "soovib rasestuda, katkestab statiiniravi",
                "statiin",
            ),
            label="jah",
        ),
        FewShot(
            user=_format_user_prompt(
                "Patsient viidi üle insuliinravile, metformiin lõpetati.",
                "Patsient viidi üle insuliinravile, metformiin lõpetati.",
                "metformiin",
            ),
            label="jah",
        ),
        FewShot(
            user=_format_user_prompt(
                "Metformiini kasutamisel tekkis iiveldus, aga ravi jätkub samas doosis",
                "tekkis iiveldus, aga ravi jätkub",
                "metformiin",
            ),
            label="ei",
        ),
        FewShot(
            user=_format_user_prompt(
                "Kõrvaltoimete tõttu vähendasime statiini doosi",
                "vähendasime statiini doosi",
                "statiin",
            ),
            label="ei",
        ),
        FewShot(
            user=_format_user_prompt(
                "Patsient jätkab enalapriilravi muutuseta, kaebusi ei ole.",
                "patsient jätkab enalapriilravi muutuseta",
                "enalapriil",
            ),
            label="ei",
        ),
    ]

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
        for frame in stratify(df, on="label", ratios=(0.8, 0.1, 0.1), seed=seed)
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

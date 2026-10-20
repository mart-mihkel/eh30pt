"""Drug discontinuation reason classification dataset."""

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
from eh30pt.types import DataSplits, DiscontClassifierLabel, FewShot

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping, Sequence
    from pathlib import Path

    from transformers import BatchEncoding, PreTrainedTokenizerFast

    from eh30pt.types import Architecture


type DiscontClassifierLabelEst = Literal[
    "Kõrvaltoimed",
    "Ravi edukas",
    "Ravi ebaefektiivne",
    "Vastunäidustus või ohtlik kooskasutamine",
    "Muu põhjus",
    "Kõrvaltoimed ilma lõpetamiseta",
    "Ravi ebaefektiivne ilma lõpetamiseta",
    "Annuse muutmine",
    "Muu sündmus ilma ravi lõpetamiseta",
]


class DiscontClassifierRow(TypedDict):
    """One discontinuation event carrying one of the classifier labels."""

    id: int
    text: str
    phrase: str
    drug: str
    label: str


class DiscontClassifier(DatasetMixin, IterableDataset, TorchIterableDataset):
    """Drug discontinuation reason classification."""

    id2label: ClassVar[Mapping[int, str]] = {
        0: "Kõrvaltoimed",
        1: "Ravi edukas",
        2: "Ravi ebaefektiivne",
        3: "Vastunäidustus või ohtlik kooskasutamine",
        4: "Muu põhjus",
        5: "Kõrvaltoimed ilma lõpetamiseta",
        6: "Ravi ebaefektiivne ilma lõpetamiseta",
        7: "Annuse muutmine",
        8: "Muu sündmus ilma ravi lõpetamiseta",
    }

    label2id: ClassVar[Mapping[str, int]] = {
        "Kõrvaltoimed": 0,
        "Ravi edukas": 1,
        "Ravi ebaefektiivne": 2,
        "Vastunäidustus või ohtlik kooskasutamine": 3,
        "Muu põhjus": 4,
        "Kõrvaltoimed ilma lõpetamiseta": 5,
        "Ravi ebaefektiivne ilma lõpetamiseta": 6,
        "Annuse muutmine": 7,
        "Muu sündmus ilma ravi lõpetamiseta": 8,
    }

    label2margend: ClassVar[
        Mapping[DiscontClassifierLabel, DiscontClassifierLabelEst]
    ] = {
        "Adverse reactions": "Kõrvaltoimed",
        "Treatment success": "Ravi edukas",
        "Treatment inefficacy": "Ravi ebaefektiivne",
        "Contraindication/unsafe co-usage": "Vastunäidustus või ohtlik kooskasutamine",
        "Other": "Muu põhjus",
        "Adverse reaction without discontinuation": "Kõrvaltoimed ilma lõpetamiseta",
        "Treatment inefficacy without discontinuation": (
            "Ravi ebaefektiivne ilma lõpetamiseta"
        ),
        "Dosage change": "Annuse muutmine",
        "Other non-discontinuation": "Muu sündmus ilma ravi lõpetamiseta",
    }

    system_prompt = dedent(
        """
        Tuvasta ravimi tarvitamise lõpetamise põhjus anamneesi ja fraasi põhjal.

        - Anamnees: arsti sissekanne patsiendi terviseseisundi ja ravi kohta.
        - Fraas: tekstist eraldatud fraas, mis kirjeldab ravimiga seotud sündmust.
        - Ravim: ravim, mille kasutamise lõpetamist tuleb hinnata.

        Vali iga sisendi jaoks täpselt üks järgmistest kategooriatest.

        1. **Kõrvaltoimed**: Ravimi kasutamine lõpetati kõrvaltoimete, allergilise reaktsiooni või teiste ravimitega tekkinud kahjuliku koostoime tõttu.
        2. **Ravi edukas**: Ravimi kasutamine lõpetati, sest ravieesmärk saavutati, ravikuur sai edukalt läbi või patsiendi terviseseisund paranes piisavalt.
        3. **Ravi ebaefektiivne**: Ravimi kasutamine lõpetati, sest ravi ei olnud piisavalt tõhus, selle toime oli kadunud või patsient ei uskunud enam ravi tõhususse.
        4. **Vastunäidustus või ohtlik kooskasutamine**: Ravimi kasutamine lõpetati, sest ilmnes või avastati meditsiiniline seisund, vastunäidustus või riskitegur, mille tõttu ravimi kasutamise jätkamine oli ohtlik või sobimatu. Siia kuuluvad ka olukorrad, kus ravimi kasutamine lõpetati ohtliku koostoime või samaaegse kasutamise riski tõttu.
        5. **Muu põhjus**: Ravimi kasutamine lõpetati muul meditsiinilisel või mittemeditsiinilisel põhjusel, mida eespool nimetatud kategooriad ei hõlma. Siia kuuluvad näiteks patsiendi isiklik otsus, ravimi kättesaadavuse probleemid või muud täpsustamata lõpetamise põhjused. Kasuta seda kategooriat ka juhul, kui lõpetamise põhjus on ebaselge, kuid ravimi kasutamise lõpetamine on kinnitatud.
        6. **Kõrvaltoimed ilma lõpetamiseta**: Esines kõrvaltoime, allergiline reaktsioon või kahjulik koostoime, kuid tekstist selgub, et ravimi kasutamist jätkati ega lõpetatud selle sündmuse tõttu.
        7. **Annuse muutmine**: Ravimi annust suurendati või vähendati, kuid ravimi kasutamist ei lõpetatud. Kasuta seda kategooriat ka juhul, kui annuse muutmine on sündmuse peamine kirjeldatud tegevus ning ravi jätkub muudetud annusega.
        8. **Ravi ebaefektiivne ilma lõpetamiseta**: Ravimi kasutamist jätkati, kuigi ravi ei olnud piisavalt tõhus, selle toime oli kadunud või patsient ei uskunud enam ravi tõhususse.
        9. **Muu sündmus ilma ravi lõpetamiseta**: Ravimi kasutamist ei lõpetatud ning sündmus ei kuulu kategooriasse **Kõrvaltoimed ilma lõpetamiseta**, **Ravi ebaefektiivne ilma lõpetamiseta** ega **Annuse muutmine**. Siia kuuluvad näiteks ravi jätkamine muutusteta, ravimi kasutamise lõpetamise kavatsus või soovitus ilma kinnitatud lõpetamiseta, ajutine ravi katkestamine, kui lõplikku lõpetamist ei ole kinnitatud, ning olukorrad, kus puudub piisav tõendus ravimi kasutamise lõpetamise kohta.

        Väljasta ainult üks kategooria nimi, ära lisa selgitusi, põhjendusi ega muud teksti.
        """  # noqa: E501
    ).strip()

    encoder_system_prompt = (
        "Klassifitseeri, miks patsient ravimi kasutamise lõpetas või miks ravi jätkus."
    )

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
            label="Kõrvaltoimed",
        ),
        FewShot(
            user=_format_user_prompt(
                "Vererõhk on püsivalt normis, lõpetas nebivolooli kasutamise",
                "Vererõhk on püsivalt normis, lõpetas nebivolooli kasutamise",
                "nebivolool",
            ),
            label="Ravi edukas",
        ),
        FewShot(
            user=_format_user_prompt(
                "Valu ei leevenenud, ibuprofeen osutus ebaefektiivseks, ravi lõpetati",
                "valu ei leevenenud, ibuprofeen osutus ebaefektiivseks",
                "ibuprofeen",
            ),
            label="Ravi ebaefektiivne",
        ),
        FewShot(
            user=_format_user_prompt(
                "Patsient soovib rasestuda, katkestab statiiniravi",
                "soovib rasestuda, katkestab statiiniravi",
                "statiin",
            ),
            label="Vastunäidustus või ohtlik kooskasutamine",
        ),
        FewShot(
            user=_format_user_prompt(
                "Patsient viidi üle insuliinravile, metformiin lõpetati.",
                "Patsient viidi üle insuliinravile, metformiin lõpetati.",
                "metformiin",
            ),
            label="Muu põhjus",
        ),
        FewShot(
            user=_format_user_prompt(
                "Metformiini kasutamisel tekkis iiveldus, aga ravi jätkub samas doosis",
                "tekkis iiveldus, aga ravi jätkub",
                "metformiin",
            ),
            label="Kõrvaltoimed ilma lõpetamiseta",
        ),
        FewShot(
            user=_format_user_prompt(
                "Patsient jätkab ibuprofeeni kasutamist, kuigi valu ei leevenenud.",
                "kuigi valu ei leevenenud",
                "ibuprofeen",
            ),
            label="Ravi ebaefektiivne ilma lõpetamiseta",
        ),
        FewShot(
            user=_format_user_prompt(
                "Kõrvaltoimete tõttu vähendasime statiini doosi",
                "vähendasime statiini doosi",
                "statiin",
            ),
            label="Annuse muutmine",
        ),
        FewShot(
            user=_format_user_prompt(
                "Patsient jätkab enalapriilravi muutuseta, kaebusi ei ole.",
                "patsient jätkab enalapriilravi muutuseta",
                "enalapriil",
            ),
            label="Muu sündmus ilma ravi lõpetamiseta",
        ),
    ]

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
    """Map each exported English label to its Estonian model output."""
    return [
        {
            "id": row["task_id"],
            "text": row["text"],
            "phrase": row["phrase"],
            "drug": row["drug"],
            "label": DiscontClassifier.label2margend[
                cast("DiscontClassifierLabel", row["label"])
            ],
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
    train, eval_, test = stratify(df, on="label", ratios=(0.8, 0.1, 0.1), seed=seed)
    logger.info(
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

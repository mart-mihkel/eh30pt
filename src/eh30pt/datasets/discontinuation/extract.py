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
        Sa oled Eesti meditsiinidokumentidele spetsialiseerunud kliinilise teksti analüüsi mudel. Sinu ülesanne on eraldada arsti koostatud eestikeelsetest kliinilistest kokkuvõtetest teave selle kohta, miks patsient lõpetas konkreetse ravimi kasutamise.

        Sisendiks on kliiniline tekst, mis võib sisaldada teavet patsiendi raviskeemi, ravimi kasutamise lõpetamise ja selle põhjuste kohta.

        Tagasta üks JSON-objekt järgmiste võtmetega: `"phrase"`, `"reason"` ja `"drug"`.

        ### Väljade määramise reeglid

        **1. `phrase`**

        Eralda sisendtekstist täpne lause või laused, mis kinnitavad, et patsient lõpetas ravimi kasutamise, ning võimaluse korral sisaldavad ka lõpetamise põhjust.

        - Säilita algteksti sõnastus muutmata kujul.
        - Kaasa mitu lauset, kui see on vajalik ravimi kasutamise lõpetamise ja selle põhjuse vahelise seose mõistmiseks.
        - Eelista võimalikult lühikest tekstilõiku, mis sisaldab kogu vajalikku teavet.
        - Ära lisa oma selgitusi ega ühenda eraldiseisvaid lauseid viisil, mis muudab algteksti.
        - Ära käsitle üksnes ravimi kasutamise lõpetamise kavatsust, soovitust või plaani tõendina, et patsient lõpetas ravimi kasutamise.
        - Kui tekst ei kinnita, et ravimi kasutamine tegelikult lõpetati, tagasta tühjad väärtused kõigi kolme välja jaoks.

        **2. `reason`**

        Eralda täpne tekstiosa, mis väljendab ravimi kasutamise lõpetamise põhjust.

        - Säilita algteksti sõnastus muutmata kujul.
        - Väljavõte peab kirjeldama põhjust, miks patsient ravimi kasutamise lõpetas, mitte üksnes lõpetamise fakti.
        - Võimalikud põhjused on näiteks vererõhu normaliseerumine, kõrvaltoime teke, ravi ebaefektiivsus, vastunäidustuse ilmnemine või patsiendi otsus, et ravim ei ole enam vajalik.
        - Ära järelda põhjust diagnoosi, ravimi omaduste või üldiste meditsiiniliste teadmiste põhjal, kui seda pole tekstis väljendatud.
        - Ära kasuta ravimi kasutamise lõpetamise fakti ennast põhjusena.
        - Kui ravimi kasutamise lõpetamine on kinnitatud, kuid põhjust ei ole tekstis nimetatud, tagasta tühi string (`""`).

        **3. `drug`**

        Eralda tekstist ravimi nimi või ravimirühm, mille kasutamise patsient lõpetas.

        - Ravimi nimi või ravimirühm peab olema sisendis sõnaselgelt nimetatud.
        - Võimalikud väärtused on näiteks "Nebilet", "Amlodipin", "vererõhuravim", "diabeediravim" või "statiin".
        - Säilita ravimi nimetus algtekstiga samal kujul. Ära teisenda kaubamärki toimeaine nimeks ega vastupidi.
        - Ära järelda ravimi nime või ravimirühma diagnoosi, näidustuse ega muu kaudse teabe põhjal.
        - Kui tekstist ei selgu, millise ravimi kasutamise patsient lõpetas, tagasta tühi string (`""`).

        ### Üldised reeglid

        - Kasuta ainult sisendis esitatud teavet.
        - Erista tegelikult toimunud ravimi kasutamise lõpetamist ravi jätkamisest, annuse muutmisest, ajutisest katkestamisest ja tulevikku suunatud lõpetamisplaanist.
        - Patsiendi enda otsus ja arsti otsus võivad mõlemad olla lõpetamise põhjuseks, kui tekst kinnitab, et ravimi kasutamine lõpetati.
        - Kui tekstis mainitakse mitut ravimit, seo lõpetamise põhjus õige ravimiga. Ära omista ühele ravimile teise ravimi lõpetamise põhjust.
        - Ära parafraseeri, tõlgi, paranda ega normaliseeri eraldatud tekstilõike. Väljad `phrase`, `reason` ja `drug` peavad sisaldama sisendist täpselt kopeeritud teksti.
        - Kui kehtivat ravimi kasutamise lõpetamise juhtumit ei leidu, tagasta kõigi väljade väärtuseks tühi string.

        ### Väljundvorming

        Tagasta ainult üks korrektne JSON-objekt järgmiste võtmetega:

        - `"phrase"` — ravimi kasutamise lõpetamist ja võimaluse korral selle põhjust tõendav tekstilõik.
        - `"reason"` — lõpetamise põhjust väljendav tekstilõik.
        - `"drug"` — lõpetatud ravimi nimi või ravimirühm.

        Kui mõne välja jaoks puudub piisav sõnaselge teave, kasuta selle väärtusena tühja stringi (`""`).

        Ära lisa JSON-objektile selgitusi, kommentaare, Markdowni vormindust ega muud teksti.
        """  # noqa: E501
    ).strip()

    extract_shots: ClassVar[list[ExtractShot]] = [
        ExtractShot(
            user="katkestas Nebileti tarvitamise, vererõhk normaliseerus",
            extractions={
                "phrase": "katkestas Nebileti tarvitamise, vererõhk normaliseerus",
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
            user="diclopramist südameplõksumine, jätab ära",
            extractions={
                "phrase": "diclopramist südameplõksumine, jätab ära",
                "reason": "diclopramist südameplõksumine",
                "drug": "diclopramis",
            },
        ),
        ExtractShot(
            user="liigeste pool kaebusteta. pole võtnud 3 kuud MTX",
            extractions={
                "phrase": "pole võtnud 3 kuud MTX",
                "reason": "liigeste poolt kaebusteta",
                "drug": "MTX",
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
                        "phrase": {"type": "string", "maxLength": 8 * 1024},
                        "reason": {"type": "string", "maxLength": 8 * 1024},
                        "drug": {"type": "string", "maxLength": 8 * 1024},
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

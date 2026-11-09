"""OBL (oblique phrase removal) dataset loading, prompting, and tokenization."""

from textwrap import dedent
from typing import TYPE_CHECKING, ClassVar, Literal, TypedDict, cast

from datasets.arrow_dataset import Dataset

from eh30pt.datasets.mixin import DatasetMixin
from eh30pt.datasets.multinerd import DataSplits
from eh30pt.logging import logger
from eh30pt.types import FewShot

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping

    from transformers import BatchEncoding, PreTrainedTokenizerFast

    from eh30pt.types import Architecture, TokenizedSample


type OblLabel = Literal["seotud", "vaba", "ebaloomulik", "liigne koma", "kaheldav"]


class OblRow(TypedDict):
    """One raw OBL annotation row."""

    id: int
    fpath: str
    sentence: str
    remove_start: int
    remove_end: int
    removed: str
    label: OblLabel
    short_sent: str
    cons_score: float
    ual: float
    la: float


class Obl(DatasetMixin[OblRow]):
    """Estonian oblique-phrase removal."""

    id2label: ClassVar[Mapping[int, str]] = {
        0: "seotud",
        1: "ebaloomulik",
        2: "liigne koma",
        3: "vaba",
        4: "kaheldav",
    }

    label2id: ClassVar[Mapping[str, int]] = {
        "seotud": 0,
        "ebaloomulik": 1,
        "liigne koma": 2,
        "vaba": 3,
        "kaheldav": 4,
    }

    system_prompt = dedent("""
        Sa oled liigse fraasi märgendamise mudel.
        Liigita eemaldatud fraas ühte kategooriasse:

        - liigne koma: ebavajalik või vale koma
        - ebaloomulik: grammatiliselt korrektne, kuid kohmakas või mittekeeleomane
        - kaheldav: ebaselge või piiripealne juhtum
        - seotud: vajalik grammatilise struktuuri jaoks, eemaldamine rikub grammatika
        - vaba: eemaldatav ilma tähendust või grammatikat muutmata

        Väljasta ainult kategooria.
    """).strip()

    encoder_system_prompt = "Liigita lausest eemaldatud fraas ühte kategooriasse."

    label_entoet: ClassVar[Mapping[str, OblLabel]] = {
        "bound": "seotud",
        "free": "vaba",
        "unnatural": "ebaloomulik",
        "redundant comma": "liigne koma",
        "dubious": "kaheldav",
    }

    data_permalink: ClassVar[str] = (
        "https://raw.githubusercontent.com/estnltk/estnltk-model-data/"
        "c6f373a49d29a5a4b5e92326e5309b173bfcd3d1"
        "/phrase_removal/benchmarks/obl_phrases/obl_max/data/obl_all_hand_annotated.csv"
    )

    @staticmethod
    def _format_user_prompt(sentence: str, removed: str) -> str:
        """Format a sentence and removed phrase as a classification prompt."""
        return dedent(f"""
            Lause: {sentence}
            Fraas: {removed}
            Kategooria:
        """).strip()

    examples: ClassVar[list[FewShot]] = [
        FewShot(
            user=_format_user_prompt(
                "Ligi kuus kuud kestnud võitluses elu eest "
                "jäid arstid seekord kaotajaks .",
                "Ligi kuus kuud",
            ),
            label="seotud",
        ),
        FewShot(
            user=_format_user_prompt(
                "Patsiendi ravijärjekorda registreerimine peab "
                "olema tagatud kõigil tööpäevadel kella 9.00- 17.00 .",
                "kõigil tööpäevadel",
            ),
            label="vaba",
        ),
        FewShot(
            user=_format_user_prompt(
                "Lordide Kojas võttis meid vastu lord Janner - "
                "väike sõbralik aadlimees , kes juhatas otsekui dirigent "
                "elegantselt kätega žestikuleerides delegatsiooni "
                "hiiglaslikus parlamendihoones sinna ja tänna .",
                "Lordide Kojas",
            ),
            label="ebaloomulik",
        ),
        FewShot(
            user=_format_user_prompt(
                "Rootsi-Vene sõja ajal ( 1656-1661 ) , kui "
                "ülikoolis katkes õppetöö , olid ülikooli trükikoja "
                "seadmed ja raamatud sissemüürituna peidus Maarja kirikus "
                "altari läheduses .",
                "Rootsi-Vene sõja ajal ( 1656-1661 )",
            ),
            label="liigne koma",
        ),
        FewShot(
            user=_format_user_prompt(
                "CELEBRATE edasiarendusena leiti , et on vaja luua "
                "laiemalt kasutatav õpiobjektide vahendamiskeskkond .",
                "CELEBRATE edasiarendusena",
            ),
            label="kaheldav",
        ),
        FewShot(
            user=_format_user_prompt(
                "Keegi mingeid takistusi ei teinud , pigem õhutasid "
                "kõik tagant ning lasksid tüdrukul endal otsustada ja "
                "saadud kogemustest õppida .",
                "tüdrukul endal",
            ),
            label="seotud",
        ),
        FewShot(
            user=_format_user_prompt(
                'Lennujaama meeneletist ostsin brošüüri " Xinjiang '
                'in brief " ( " Lühidalt Xinjiangist " ) ja lugesin , et '
                "Ürümchi lennuväli on suuruselt Hiinas viies ning see on "
                "reservlennuväljaks Euroopast Pekingisse lendavatele "
                "hiidlennukitele .",
                "Euroopast",
            ),
            label="vaba",
        ),
        FewShot(
            user=_format_user_prompt(
                "Samal ajal pakkus Walesi prints Eesti presidendile "
                "oma tehases toodetud maheküpsiseid , vestlusteemadeks "
                "mahepõllundus ja päevapoliitika .",
                "Samal ajal",
            ),
            label="ebaloomulik",
        ),
        FewShot(
            user=_format_user_prompt(
                "Miinimumpalgal tundub olema normatiivne roll "
                "palgataseme kohandumisele - lisaks sellele , et see võib "
                "takistada tööandjatel tööle võtmast vähemproduktiivset "
                "tööjõudu , suurendab ta ka reservatsioonipalga määra .",
                "lisaks sellele , et see võib takistada tööandjatel "
                "tööle võtmast vähemproduktiivset tööjõudu",
            ),
            label="liigne koma",
        ),
        FewShot(
            user=_format_user_prompt(
                "Rootslased kinnitavad aga , et traadiga telefonile "
                "konkurentsi nad Starmani kaudu ei paku .",
                "Starmani kaudu",
            ),
            label="kaheldav",
        ),
    ]

    def __init__(
        self,
        data: Dataset,
        tokenizer: PreTrainedTokenizerFast,
        arch: Architecture,
        n_shot: int = 0,
        n_virtual: int = 0,
        *,
        sequence_classification: bool = False,
    ) -> None:
        """Store one split with its tokenization settings."""
        self._data = data
        self._tokenizer = tokenizer
        self._arch = arch
        self._n_shot = n_shot
        self._n_virtual = n_virtual
        self._sequence_classification = sequence_classification

    def __len__(self) -> int:
        """Return the number of rows in the split."""
        return len(self._data)

    def __getitem__(self, index: int) -> TokenizedSample:
        """Tokenize the indexed row."""
        row = cast("OblRow", self._data[index])
        encoded = self.tokenize(row, self._tokenizer, self._n_shot, self._n_virtual)
        return cast("TokenizedSample", dict(encoded))

    def samples(self) -> Iterator[FewShot]:
        """Yield decoder prompt and label pairs for the split."""
        for idx, row in enumerate(self._data):
            yield FewShot(
                user=self._format_user_prompt(row["sentence"], row["removed"]),
                label=row["label"],
                id=idx,
            )

    def total(self) -> int:
        """Return the number of rows in the split."""
        return len(self._data)

    def tokenize(
        self,
        row: OblRow,
        tokenizer: PreTrainedTokenizerFast,
        n_shot: int = 0,
        n_virtual: int = 0,
    ) -> BatchEncoding[list[int]]:
        """Tokenize one row with few-shot context."""
        if n_shot > len(self.examples):
            msg = "requested more few shot examples than defined"
            raise ValueError(msg)

        if self._arch == "encoder":
            sep = self._tokenizer.sep_token
            user_prompt = f"{row['removed']}{sep}{row['sentence']}"
        else:
            user_prompt = self._format_user_prompt(row["sentence"], row["removed"])

        few_shots = self.examples[:n_shot]
        label = row["label"]

        return self._encode_example(
            arch=self._arch,
            tokenizer=tokenizer,
            user_prompt=user_prompt,
            label=label,
            few_shots=few_shots,
            n_virtual=n_virtual,
            sequence_classification=self._sequence_classification,
        )


def load_obl(
    tokenizer: PreTrainedTokenizerFast,
    arch: Architecture,
    n_shot: int = 0,
    n_virtual: int = 0,
    *,
    sequence_classification: bool = False,
) -> DataSplits:
    """Load OBL splits as indexable datasets."""
    logger.debug("loading obl csv from github permalink")
    raw = Dataset.from_csv(Obl.data_permalink, sep=";")

    logger.debug("renaming 'type' column to 'label'")
    raw = raw.rename_column("type", "label")

    logger.debug("translating labels to estonian")
    raw = raw.map(lambda row: {"label": Obl.label_entoet[row["label"]]})

    test_rest = raw.train_test_split(test_size=1000, seed=0)
    train_eval = test_rest["train"].train_test_split(test_size=128, seed=0)

    train: Dataset = train_eval["train"]
    eval_: Dataset = train_eval["test"]
    test: Dataset = test_rest["test"]

    return DataSplits(
        Obl(
            train,
            tokenizer,
            arch,
            n_shot,
            n_virtual,
            sequence_classification=sequence_classification,
        ),
        Obl(
            eval_,
            tokenizer,
            arch,
            n_shot,
            n_virtual,
            sequence_classification=sequence_classification,
        ),
        Obl(
            test,
            tokenizer,
            arch,
            n_shot,
            n_virtual,
            sequence_classification=sequence_classification,
        ),
    )

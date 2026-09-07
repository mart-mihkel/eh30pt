"""OBL (oblique phrase removal) dataset loading, prompting, and tokenization."""

from textwrap import dedent
from typing import TYPE_CHECKING, ClassVar, Literal, TypedDict, cast

from datasets.arrow_dataset import Dataset
from datasets.dataset_dict import DatasetDict

from eh30pt.constants import SENTINEL_TOKEN
from eh30pt.datasets.truncation import get_max_length
from eh30pt.logging import logger
from eh30pt.types import DatasetProtocol

if TYPE_CHECKING:
    from collections.abc import Mapping

    from datasets.splits import Split
    from transformers import BatchEncoding, PreTrainedTokenizerFast

    from eh30pt.types import Architecture


type _Label = Literal["seotud", "vaba", "ebaloomulik", "liigne koma", "kaheldav"]


class _Example(TypedDict):
    """A single raw OBL example."""

    id: int
    fpath: str
    sentence: str
    remove_start: int
    remove_end: int
    removed: str
    label: _Label
    short_sent: str
    cons_score: float
    ual: float
    la: float


class Oblique(DatasetProtocol):
    """
    Estonian oblique-phrase removal classification with architecture prompts.

    Parameters
    ----------
    tokenizer:
        Tokenizer used for prompting and tokenization.
    arch:
        Model architecture, selecting the prompt format.

    """

    id2label: ClassVar[Mapping[int, _Label]] = {
        0: "seotud",
        1: "ebaloomulik",
        2: "liigne koma",
        3: "vaba",
        4: "kaheldav",
    }

    label2id: ClassVar[Mapping[_Label, int]] = {
        "seotud": 0,
        "ebaloomulik": 1,
        "liigne koma": 2,
        "vaba": 3,
        "kaheldav": 4,
    }

    _en_to_et: ClassVar[Mapping[str, _Label]] = {
        "bound": "seotud",
        "free": "vaba",
        "unnatural": "ebaloomulik",
        "redundant comma": "liigne koma",
        "dubious": "kaheldav",
    }

    _permalink: ClassVar[str] = (
        "https://raw.githubusercontent.com/estnltk/estnltk-model-data/"
        "c6f373a49d29a5a4b5e92326e5309b173bfcd3d1"
        "/phrase_removal/benchmarks/obl_phrases/obl_max/data/obl_all_hand_annotated.csv"
    )

    def __init__(self, tokenizer: PreTrainedTokenizerFast, arch: Architecture) -> None:
        """Store the tokenizer and architecture."""
        self._tokenizer = tokenizer
        self._arch = arch
        self._data: DatasetDict | None = None
        self._raw_data: DatasetDict | None = None

    @property
    def system_prompt(self) -> str:
        """Return the system prompt for the configured architecture."""
        if self._arch == "encoder":
            return (
                "Mis on seos fragmendi ja lause vahel,\n"
                "kas see on seotud, vaba, kaheldav, ebaloomulik "
                f"või liigne koma?{self._tokenizer.sep_token}"
            )

        if self._arch == "decoder":
            return dedent(
                """
                Sa oled liigse fraasi märgendamise mudel.
                Liigita eemaldatud fraas ühte kategooriasse:

                - liigne koma: ebavajalik või vale koma
                - ebaloomulik: grammatiliselt korrektne, kuid kohmakas või mittekeeleomane
                - kaheldav: ebaselge või piiripealne juhtum
                - seotud: vajalik grammatilise struktuuri jaoks, eemaldamine rikub grammatika
                - vaba: eemaldatav ilma tähendust või grammatikat muutmata

                Väljasta ainult kategooria.
            """  # noqa: E501
            ).strip()

        return dedent(
            """
            liigse fraasi märgendamine: liigita eemaldatud tekst ühte viiest kategooriast.

            liigne koma: ebavajalik või vale koma
            ebaloomulik: grammatiliselt valikuline, kuid kohmakas või mittekeeleomane
            kaheldav: ebaselge või piiripealne juhtum
            seotud: vajalik grammatilise struktuuri jaoks; eemaldamine rikub grammatika
            vaba: eemaldatav ilma tähendust või grammatikat muutmata

            väljasta ainult kategooria
        """  # noqa: E501
        ).strip()

    @property
    def data(self) -> DatasetDict:
        """Return the loaded dataset splits."""
        if self._data is None:
            msg = "dataset not loaded, call load() first"
            raise RuntimeError(msg)

        return self._data

    @property
    def raw_data(self) -> DatasetDict:
        """Return the untokenized dataset splits."""
        if self._raw_data is None:
            msg = "dataset not loaded, call load() first"
            raise RuntimeError(msg)

        return self._raw_data

    def load(
        self,
        *,
        n_shot: int = 0,
        n_virtual: int = 0,
        split: Split | None = None,
    ) -> None:
        """
        Load, tokenize, and prompt-format the OBL dataset into instance state.

        OBL is always loaded in full from a fixed CSV, so `split` has no
        effect; it exists only so this method matches the shared protocol.

        Parameters
        ----------
        n_shot:
            Number of few-shot examples prepended to each prompt.
        n_virtual:
            Number of virtual tokens reserved in the truncation budget.
        split:
            Ignored, the full dataset is always loaded.

        """
        del split

        self._raw_data = self._load_raw()
        examples = self._sample_examples(n_shot)
        self._data = self._tokenize(examples, n_virtual=n_virtual)

    def _load_raw(self) -> DatasetDict:
        logger.debug("loading obl csv from github permalink")
        raw = Dataset.from_csv(self._permalink, sep=";")

        logger.debug("rename 'type' to 'label'")
        raw = raw.rename_column("type", "label")

        logger.debug("translate labels")
        raw = raw.map(self._translate_en_to_et)

        s1 = raw.train_test_split(test_size=1000, seed=0)
        s2 = s1["train"].train_test_split(test_size=128, seed=0)

        return DatasetDict(
            {
                "train": s2["train"],
                "validation": s2["test"],
                "test": s1["test"],
            }
        )

    def _tokenize(self, examples: list[str], n_virtual: int = 0) -> DatasetDict:
        logger.debug("tokenizing oblique")
        data = self.raw_data.map(
            self._tokenize_example,
            fn_kwargs={"examples": examples, "n_virtual": n_virtual},
            remove_columns=[
                "id",
                "fpath",
                "sentence",
                "remove_start",
                "remove_end",
                "removed",
                "label",
                "short_sent",
                "cons_score",
                "ual",
                "la",
            ],
        )

        for subsplit in data:
            logger.debug("tokenized %d %s samples", len(data[subsplit]), subsplit)

        return data

    def _build_prompt(self, example: _Example, examples: list[str]) -> str:
        if self._arch == "encoder":
            prompt = (
                f"{example['removed']}{self._tokenizer.sep_token}{example['sentence']}"
            )
        elif self._arch == "decoder":
            prompt = dedent(f"""
                Lause: {example["sentence"]}
                Fraas: {example["removed"]}
                Kategooria:
            """).strip()
        else:
            prompt = dedent(f"""
                lause: {example["sentence"]}
                fraas: {example["removed"]}
                kategooria:
            """).strip()

        if examples:
            prompt_examples = "\n".join(examples)
            prompt = f"{prompt_examples}\n{prompt}"

        return prompt

    def _encode_prompt(
        self,
        sys: str,
        prompt: str,
        max_length: int | None,
    ) -> BatchEncoding[list[int]]:
        """Encode the system prompt together with the sample prompt."""
        if self._tokenizer.chat_template is None:
            prompt_enc = self._tokenizer(
                f"{sys}\n{prompt}",
                truncation=True,
                max_length=max_length,
            )
        else:
            conv = [
                {"role": "system", "content": sys},
                {"role": "user", "content": prompt},
            ]

            prompt_enc = self._tokenizer.apply_chat_template(
                conv,
                truncation=True,
                max_length=max_length,
                return_dict=True,
                add_generation_prompt=self._arch != "encoder",
            )

        return cast("BatchEncoding[list[int]]", prompt_enc)

    def _encode_answer(
        self,
        sys: str,
        prompt: str,
        label: str,
        max_length: int | None,
    ) -> BatchEncoding[list[int]]:
        """Encode the prompt together with the gold label."""
        if self._tokenizer.chat_template is None:
            answer_enc = self._tokenizer(
                f"{sys}\n{prompt} {label}",
                truncation=True,
                max_length=max_length,
            )
        else:
            conv = [
                {"role": "system", "content": sys},
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": label},
            ]

            answer_enc = self._tokenizer.apply_chat_template(
                conv,
                truncation=True,
                max_length=max_length,
                return_dict=True,
            )

        return cast("BatchEncoding[list[int]]", answer_enc)

    def _tokenize_example(
        self,
        example: _Example,
        examples: list[str],
        n_virtual: int = 0,
    ) -> BatchEncoding:
        max_length = get_max_length(self._tokenizer, n_virtual)
        prompt = self._build_prompt(example, examples)
        sys = self.system_prompt
        label = example["label"]

        prompt_enc: BatchEncoding = self._encode_prompt(sys, prompt, max_length)
        prompt_len = len(prompt_enc["input_ids"])
        truncated = max_length is not None and prompt_len >= max_length

        if self._arch == "encoder":
            prompt_enc["labels"] = self.label2id[label]
            prompt_enc["truncated"] = truncated
            return prompt_enc

        answer_enc: BatchEncoding = self._encode_answer(sys, prompt, label, max_length)
        labels_enc = answer_enc["input_ids"].copy()
        truncated = truncated or (
            max_length is not None and len(labels_enc) >= max_length
        )

        if self._arch == "decoder":
            labels_enc[:prompt_len] = [SENTINEL_TOKEN] * prompt_len
            answer_enc["labels"] = labels_enc
            answer_enc["truncated"] = truncated
            return answer_enc

        idx = prompt_len - int(labels_enc[-1] == self._tokenizer.eos_token_id)
        prompt_enc["labels"] = labels_enc[idx:]
        prompt_enc["truncated"] = truncated
        return prompt_enc

    def _sample_examples(self, n_shot: int) -> list[str]:
        train = self.raw_data["train"]
        max_shots = len(train)
        if n_shot > max_shots:
            msg = f"requested more than {max_shots} examples"
            raise ValueError(msg)
        if n_shot == 0:
            return []

        sampled = train.select(range(n_shot))
        return [self._format_example(example) for example in sampled]

    @staticmethod
    def _format_example(example: _Example) -> str:
        return (
            f"Lause: {example['sentence']}\n"
            f"Fraas: {example['removed']}\n"
            f"Kategooria: {example['label']}\n"
        )

    @staticmethod
    def _translate_en_to_et(example: _Example) -> _Example:
        en_lbl = example["label"]
        example["label"] = Oblique._en_to_et[en_lbl]
        return example

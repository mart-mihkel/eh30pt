"""MultiNERD dataset loading, prompting, and tokenization."""

from textwrap import dedent
from typing import TYPE_CHECKING, ClassVar, Literal, TypedDict, cast

from datasets.load import load_dataset
from datasets.utils.info_utils import VerificationMode

from eh30pt.constants import SENTINEL_TOKEN
from eh30pt.datasets.truncation import get_max_length
from eh30pt.logging import logger
from eh30pt.types import DatasetProtocol

if TYPE_CHECKING:
    from collections.abc import Mapping

    from datasets.dataset_dict import DatasetDict
    from datasets.splits import Split
    from transformers import BatchEncoding, PreTrainedTokenizerFast

    from eh30pt.types import Architecture

type _Lang = Literal[
    "zh",
    "nl",
    "en",
    "fr",
    "de",
    "it",
    "pl",
    "pt",
    "ru",
    "es",
]

type _Tag = Literal[
    "PER",
    "ORG",
    "LOC",
    "ANIM",
    "BIO",
    "CEL",
    "DIS",
    "EVE",
    "FOOD",
    "INST",
    "MEDIA",
    "MYTH",
    "PLANT",
    "TIME",
    "VEHI",
]


class _Batch(TypedDict):
    tokens: list[list[str]]
    ner_tags: list[list[int]]
    lang: list[_Lang]


class _TokBatch(TypedDict):
    input_ids: list[list[int]]
    attention_mask: list[list[int]]
    labels: list[int | list[int]]
    truncated: list[bool]


class Multinerd(DatasetProtocol):
    """
    MultiNERD NER tag classification dataset.

    Parameters
    ----------
    tokenizer:
        Tokenizer used for prompting and tokenization.
    arch:
        Model architecture, selecting the prompt format.

    """

    id2label: ClassVar[Mapping[int, _Tag]] = {
        0: "PER",
        1: "ORG",
        2: "LOC",
        3: "ANIM",
        4: "BIO",
        5: "CEL",
        6: "DIS",
        7: "EVE",
        8: "FOOD",
        9: "INST",
        10: "MEDIA",
        11: "MYTH",
        12: "PLANT",
        13: "TIME",
        14: "VEHI",
    }

    label2id: ClassVar[Mapping[_Tag, int]] = {
        "PER": 0,
        "ORG": 1,
        "LOC": 2,
        "ANIM": 3,
        "BIO": 4,
        "CEL": 5,
        "DIS": 6,
        "EVE": 7,
        "FOOD": 8,
        "INST": 9,
        "MEDIA": 10,
        "MYTH": 11,
        "PLANT": 12,
        "TIME": 13,
        "VEHI": 14,
    }

    _id2label_bio: ClassVar[Mapping[int, str]] = {
        0: "O",
        1: "B-PER",
        2: "I-PER",
        3: "B-ORG",
        4: "I-ORG",
        5: "B-LOC",
        6: "I-LOC",
        7: "B-ANIM",
        8: "I-ANIM",
        9: "B-BIO",
        10: "I-BIO",
        11: "B-CEL",
        12: "I-CEL",
        13: "B-DIS",
        14: "I-DIS",
        15: "B-EVE",
        16: "I-EVE",
        17: "B-FOOD",
        18: "I-FOOD",
        19: "B-INST",
        20: "I-INST",
        21: "B-MEDIA",
        22: "I-MEDIA",
        23: "B-MYTH",
        24: "I-MYTH",
        25: "B-PLANT",
        26: "I-PLANT",
        27: "B-TIME",
        28: "I-TIME",
        29: "B-VEHI",
        30: "I-VEHI",
    }

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
                "What is the NER tag of the entity in the sentence?"
                f"{self._tokenizer.sep_token}"
            )

        return dedent(f"""
            Identify the NER tag of the entity in the sentence.
            Possible tags are: {", ".join(self.id2label.values())}.

            Output only the tag.
        """).strip()

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
        filter_en: bool = True,
    ) -> None:
        """
        Load a modified subset of the MultiNERD dataset into instance state.

        The BIO tagging task is converted to a regular NER tagging task by
        joining tokens with B- and I- prefixes into a single span. O tags are
        dropped entirely.

        Each token is split into a separate sample containing the entire
        context sentence and the target token. The task is to classify the tag
        of the token in the entire sequence.

        Parameters
        ----------
        n_shot:
            Number of few-shot examples prepended to each prompt.
        n_virtual:
            Number of virtual tokens reserved in the truncation budget.
        split:
            Dataset split to load, or None for all splits.
        filter_en:
            Only use the english subset.

        """
        self._raw_data = self._load_raw(split=split, filter_en=filter_en)
        self._data = self._tokenize(
            self._sample_examples(n_shot),
            n_virtual=n_virtual,
        )

    def _load_raw(self, split: Split | None, *, filter_en: bool) -> DatasetDict:
        logger.debug("loading raw multinerd dataset")
        data = load_dataset(
            "Babelscape/multinerd",
            split=split,
            verification_mode=VerificationMode.NO_CHECKS,
        )

        data = cast("DatasetDict", data)
        if filter_en:
            logger.warning("using english only subset")
            data = data.filter(self._filter_english, batched=True)

        return data

    def _tokenize(
        self,
        examples: list[str],
        n_virtual: int = 0,
        num_proc: int = 4,
    ) -> DatasetDict:
        logger.debug("tokenizing multinerd")
        data = self.raw_data.map(
            self._tokenize_batch,
            batched=True,
            num_proc=num_proc,
            remove_columns=["tokens", "ner_tags", "lang"],
            fn_kwargs={"examples": examples, "n_virtual": n_virtual},
        )

        for subsplit in data:
            logger.debug("tokenized %d %s samples", len(data[subsplit]), subsplit)

        return data

    def _build_prompt(self, sentence: str, entity: str, examples: list[str]) -> str:
        if self._arch == "encoder":
            prompt = f"{sentence}{self._tokenizer.sep_token}{entity}"
        else:
            prompt = dedent(f"""
                Sentence: {sentence}
                Entity: {entity}
                Tag:
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
        tag_id: int,
        max_length: int | None,
    ) -> BatchEncoding[list[int]]:
        if self._tokenizer.chat_template is None:
            answer_enc = self._tokenizer(
                f"{sys}\n{prompt} {self.id2label[tag_id]}",
                truncation=True,
                max_length=max_length,
            )
        else:
            conv = [
                {"role": "system", "content": sys},
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": self.id2label[tag_id]},
            ]

            answer_enc = self._tokenizer.apply_chat_template(
                conv,
                truncation=True,
                max_length=max_length,
                return_dict=True,
            )

        return cast("BatchEncoding[list[int]]", answer_enc)

    def _tokenize_batch(
        self,
        batch: _Batch,
        examples: list[str],
        n_virtual: int = 0,
    ) -> _TokBatch:
        all_ids: list[list[int]] = []
        all_attn: list[list[int]] = []
        all_labels: list[int | list[int]] = []
        all_truncated: list[bool] = []
        max_length = get_max_length(self._tokenizer, n_virtual)

        sys = self.system_prompt
        token_tags = zip(batch["tokens"], batch["ner_tags"], strict=True)
        for tokens, raw_tag_ids in token_tags:
            sentence = " ".join(tokens)
            entities, tag_ids = self._join_spans(tokens, raw_tag_ids)

            for entity, tag_id in zip(entities, tag_ids, strict=True):
                if tag_id == -1:
                    continue

                prompt = self._build_prompt(sentence, entity, examples)
                prompt_enc = self._encode_prompt(sys, prompt, max_length)
                prompt_len = len(prompt_enc["input_ids"])
                truncated = max_length is not None and prompt_len >= max_length

                if self._arch == "encoder":
                    all_ids.append(prompt_enc["input_ids"])
                    all_attn.append(prompt_enc["attention_mask"])
                    all_labels.append(tag_id)
                    all_truncated.append(truncated)
                    continue

                answer_enc = self._encode_answer(sys, prompt, tag_id, max_length)
                labels_enc = answer_enc["input_ids"].copy()
                truncated = truncated or (
                    max_length is not None and len(labels_enc) >= max_length
                )

                if self._arch == "decoder":
                    all_ids.append(answer_enc["input_ids"])
                    all_attn.append(answer_enc["attention_mask"])
                    labels_enc[:prompt_len] = [SENTINEL_TOKEN] * prompt_len
                    all_labels.append(labels_enc)
                    all_truncated.append(truncated)
                    continue

                if self._arch == "encoder-decoder":
                    all_ids.append(prompt_enc["input_ids"])
                    all_attn.append(prompt_enc["attention_mask"])
                    idx = prompt_len - int(
                        labels_enc[-1] == self._tokenizer.eos_token_id
                    )
                    all_labels.append(labels_enc[idx:])
                    all_truncated.append(truncated)
                    continue

        return _TokBatch(
            input_ids=all_ids,
            attention_mask=all_attn,
            labels=all_labels,
            truncated=all_truncated,
        )

    @staticmethod
    def _join_spans(
        tokens: list[str],
        tag_ids: list[int],
    ) -> tuple[list[str], list[int]]:
        out_ids = []
        out_tokens = []
        for token, tag_id in zip(tokens, tag_ids, strict=True):
            tag = Multinerd._id2label_bio[tag_id]

            if tag.startswith("B-"):
                tag = cast("_Tag", tag[2:])
                out_ids.append(Multinerd.label2id[tag])
                out_tokens.append(token)
            elif tag.startswith("I-"):
                out_tokens[-1] = f"{out_tokens[-1]} {token}"
            elif tag == "O":
                out_ids.append(-1)
                out_tokens.append(token)
            else:
                tag = cast("_Tag", tag)
                out_ids.append(Multinerd.label2id[tag])
                out_tokens.append(token)

        return out_tokens, out_ids

    @staticmethod
    def _filter_english(batch: _Batch) -> list[bool]:
        return [lang == "en" for lang in batch["lang"]]

    @staticmethod
    def _format_example(sentence: str, entity: str, tag: str) -> str:
        return f"Sentence: {sentence}\nEntity: {entity}\nTag: {tag}\n"

    def _sample_examples(self, n_shot: int) -> list[str]:
        if n_shot == 0:
            return []

        all_examples = []
        for example in self.raw_data["train"]:
            tokens = cast("list[str]", example["tokens"])
            raw_tag_ids = cast("list[int]", example["ner_tags"])
            sentence = " ".join(tokens)
            entities, tag_ids = Multinerd._join_spans(tokens, raw_tag_ids)

            for entity, tag_id in zip(entities, tag_ids, strict=True):
                if tag_id == -1:
                    continue

                examples = self._format_example(
                    sentence,
                    entity,
                    Multinerd.id2label[tag_id],
                )

                all_examples.append(examples)
                break

            if len(all_examples) == n_shot:
                return all_examples

        msg = f"requested more than {len(all_examples)} examples"
        raise ValueError(msg)

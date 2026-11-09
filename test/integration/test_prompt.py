"""Prompting and few-shot formatting integration tests with real tokenizers."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Final, cast

import polars as pl
import pytest
from datasets import Dataset
from sqlalchemy import create_engine as _create_engine
from sqlalchemy.orm import Session as _Session

from eh30pt.constants import SENTINEL_TOKEN
from eh30pt.datasets.discont.classify import (
    DiscontClassifier,
    DiscontClassifierRow,
    load_discont_classifier,
)
from eh30pt.datasets.discont.extract import (
    DiscontExtract,
    DiscontExtractRow,
    load_discont_extract,
)
from eh30pt.datasets.discont.extract import DiscontExtract as _Extract
from eh30pt.datasets.multinerd import Multinerd, MultinerdRow
from eh30pt.datasets.oblique import Obl, OblRow
from eh30pt.datasets.registry import get_labels as _get_labels
from eh30pt.esthealth.queries import select_trigger_events as _select_trigger
from eh30pt.labeling.export.extract import ExtractionFrameSchema
from eh30pt.settings import PostgresSettings as _PostgresSettings
from eh30pt.types import Architecture, FewShot

if TYPE_CHECKING:
    from pathlib import Path

    from transformers import PreTrainedTokenizerFast

    from eh30pt.datasets.multinerd import MultinerdTag

_n_fixture_rows: Final = 2


@pytest.fixture
def oblique_sample() -> OblRow:
    """Return the first real oblique row from the upstream CSV."""
    from datasets import Dataset as _Dataset

    from eh30pt.datasets.oblique import Obl as _Obl

    raw = _Dataset.from_csv(_Obl.data_permalink, sep=";")
    raw = raw.rename_column("type", "label")
    raw = raw.map(lambda row: {"label": _Obl.label_entoet[row["label"]]})
    row = raw[0]
    return {
        "id": row["id"],
        "fpath": row["fpath"],
        "sentence": row["sentence"],
        "remove_start": row["remove_start"],
        "remove_end": row["remove_end"],
        "removed": row["removed"],
        "label": row["label"],
        "short_sent": row["short_sent"],
        "cons_score": row["cons_score"],
        "ual": row["ual"],
        "la": row["la"],
    }


@pytest.fixture
def oblique_data() -> Dataset:
    """Return two real oblique rows from the upstream CSV."""
    from datasets import Dataset as _Dataset

    from eh30pt.datasets.oblique import Obl as _Obl

    raw = _Dataset.from_csv(_Obl.data_permalink, sep=";")
    raw = raw.rename_column("type", "label")
    raw = raw.map(lambda row: {"label": _Obl.label_entoet[row["label"]]})
    return raw.select(range(2))


@pytest.fixture
def nerd_data() -> Dataset:
    """Return two real multinerd rows from Hugging Face."""
    from datasets.load import load_dataset as _load_dataset
    from datasets.utils.info_utils import VerificationMode as _VerificationMode

    raw = _load_dataset(
        "Babelscape/multinerd",
        split="train[:2]",
        verification_mode=_VerificationMode.NO_CHECKS,
    )
    return Dataset.from_list(list(raw))


@pytest.fixture
def shots() -> list[FewShot]:
    """Return two real oblique few-shot examples from the registry."""
    from eh30pt.datasets.registry import get_few_shots

    return get_few_shots("oblique", 2)


@pytest.fixture
def classifier_rows(database: None) -> list[DiscontClassifierRow]:
    """Return classifier rows built from seeded DB text and registry labels."""
    _ = database
    labels = _get_labels("discont-classify/extract-labels")
    engine = _create_engine(_PostgresSettings().connection_url)
    try:
        with _Session(engine) as session:
            events = _select_trigger(session, limit=20)
    finally:
        engine.dispose()
    texts = [event.text for event in events if event.text][:_n_fixture_rows]
    assert len(texts) == _n_fixture_rows
    valid = [
        fields
        for shot in _Extract.examples
        if (fields := cast("dict[str, str]", json.loads(shot.label)))["phrase"]
        and fields["drug"]
    ][:_n_fixture_rows]
    assert len(valid) == _n_fixture_rows
    return [
        {
            "id": events[idx].id,
            "text": texts[idx],
            "phrase": valid[idx]["phrase"],
            "drug": valid[idx]["drug"],
            "label": labels[idx],
        }
        for idx in range(_n_fixture_rows)
    ]


def test_format_few_shot_legacy(
    oblique_data: Dataset,
    bert_tokenizer: PreTrainedTokenizerFast,
    shots: list[FewShot],
) -> None:
    ds = Obl(oblique_data, bert_tokenizer, "decoder")

    assert ds._format_few_shot_legacy([]) == ""
    assert ds._format_few_shot_legacy(shots) == (
        f"{shots[0].user} {shots[0].label}\n\n{shots[1].user} {shots[1].label}\n"
    )


def test_format_few_shot_chat_builds_turns(
    oblique_data: Dataset,
    llama_tokenizer: PreTrainedTokenizerFast,
    shots: list[FewShot],
) -> None:
    ds = Obl(oblique_data, llama_tokenizer, "decoder")

    assert ds._format_few_shot_chat(llama_tokenizer, []) == []
    assert ds._format_few_shot_chat(llama_tokenizer, shots) == [
        {"role": "user", "content": shots[0].user},
        {"role": "assistant", "content": shots[0].label},
        {"role": "user", "content": shots[1].user},
        {"role": "assistant", "content": shots[1].label},
    ]


def test_format_few_shot_chat_requires_template(
    oblique_data: Dataset,
    bert_tokenizer: PreTrainedTokenizerFast,
    shots: list[FewShot],
) -> None:
    ds = Obl(oblique_data, bert_tokenizer, "decoder")

    with pytest.raises(RuntimeError, match="chat template"):
        ds._format_few_shot_chat(bert_tokenizer, shots)


def test_encode_prompt_legacy_round_trips(
    oblique_data: Dataset,
    gpt2_tokenizer: PreTrainedTokenizerFast,
    shots: list[FewShot],
) -> None:
    ds = Obl(oblique_data, gpt2_tokenizer, "decoder")
    encoded = ds._encode_prompt(
        arch="decoder",
        tokenizer=gpt2_tokenizer,
        user_prompt="USER",
        few_shots=shots,
        max_len=None,
    )

    assert gpt2_tokenizer.decode(encoded["input_ids"]) == (
        f"{Obl.system_prompt}\n{shots[0].user} {shots[0].label}\n"
        f"\n{shots[1].user} {shots[1].label}\n\nUSER"
    )


@pytest.mark.parametrize("arch", ["encoder", "decoder", "encoder-decoder"])
def test_encode_prompt_generation_flag(
    arch: Architecture,
    oblique_data: Dataset,
    gemma_tokenizer: PreTrainedTokenizerFast,
) -> None:
    ds = Obl(oblique_data, gemma_tokenizer, arch)
    encoded = ds._encode_prompt(
        arch=arch,
        tokenizer=gemma_tokenizer,
        user_prompt="USER",
        few_shots=[],
        max_len=None,
    )
    decoded = gemma_tokenizer.decode(encoded["input_ids"])
    assert isinstance(decoded, str)

    assert "USER" in decoded
    if arch == "encoder":
        assert "<start_of_turn>model" not in decoded
    else:
        assert decoded.rstrip().endswith("<start_of_turn>model")


def test_encode_answer_legacy_appends_label(
    oblique_data: Dataset,
    gpt2_tokenizer: PreTrainedTokenizerFast,
) -> None:
    from eh30pt.datasets.registry import get_labels

    label = get_labels("oblique")[0]
    ds = Obl(oblique_data, gpt2_tokenizer, "decoder")
    encoded = ds._encode_answer(
        tokenizer=gpt2_tokenizer,
        user_prompt="USER",
        label=label,
        few_shots=[],
        max_len=None,
    )

    assert gpt2_tokenizer.decode(encoded["input_ids"]) == (
        f"{Obl.system_prompt}\n\nUSER {label}"
    )


def test_encode_answer_template(
    oblique_data: Dataset,
    gemma_tokenizer: PreTrainedTokenizerFast,
) -> None:
    from eh30pt.datasets.registry import get_labels

    label = get_labels("oblique")[0]
    ds = Obl(oblique_data, gemma_tokenizer, "decoder")
    encoded = ds._encode_answer(
        tokenizer=gemma_tokenizer,
        user_prompt="USER",
        label=label,
        few_shots=[],
        max_len=None,
    )
    decoded = gemma_tokenizer.decode(encoded["input_ids"])
    assert isinstance(decoded, str)

    assert "USER" in decoded
    assert decoded.rstrip().endswith(f"{label}<end_of_turn>")


def test_tokenize_decoder_masks_prompt(
    oblique_data: Dataset,
    oblique_sample: OblRow,
    gpt2_tokenizer: PreTrainedTokenizerFast,
) -> None:
    ds = Obl(oblique_data, gpt2_tokenizer, "decoder")
    encoded = ds.tokenize(oblique_sample, gpt2_tokenizer)

    user = Obl._format_user_prompt(
        oblique_sample["sentence"], oblique_sample["removed"]
    )
    label = oblique_sample["label"]
    prompt_ids = ds._encode_prompt(
        arch="decoder",
        tokenizer=gpt2_tokenizer,
        user_prompt=user,
        few_shots=[],
        max_len=None,
    )["input_ids"]
    answer_ids = ds._encode_answer(
        tokenizer=gpt2_tokenizer,
        user_prompt=user,
        label=label,
        few_shots=[],
        max_len=None,
    )["input_ids"]

    assert encoded["labels"] == [-100] * len(prompt_ids) + answer_ids[len(prompt_ids) :]
    assert encoded["truncated"] is False


def test_tokenize_encoder_prompt_and_int_label(
    oblique_data: Dataset,
    oblique_sample: OblRow,
    bert_tokenizer: PreTrainedTokenizerFast,
) -> None:
    ds = Obl(oblique_data, bert_tokenizer, "encoder")
    encoded = ds.tokenize(oblique_sample, bert_tokenizer)

    assert encoded["labels"] == Obl.label2id[oblique_sample["label"]]
    assert encoded["truncated"] is False
    decoded = bert_tokenizer.decode(encoded["input_ids"])
    assert oblique_sample["removed"].split()[0].lower() in decoded.lower()
    assert "[SEP]" in decoded
    assert "lausest eemaldatud" in decoded.lower()
    assert "märgendamise" not in decoded.lower()


def test_tokenize_encoder_decoder_suffix_labels(
    oblique_data: Dataset,
    oblique_sample: OblRow,
    t5_tokenizer: PreTrainedTokenizerFast,
) -> None:
    ds = Obl(oblique_data, t5_tokenizer, "encoder-decoder")
    encoded = ds.tokenize(oblique_sample, t5_tokenizer)

    user = Obl._format_user_prompt(
        oblique_sample["sentence"], oblique_sample["removed"]
    )
    label = oblique_sample["label"]
    prompt_ids = ds._encode_prompt(
        arch="encoder-decoder",
        tokenizer=t5_tokenizer,
        user_prompt=user,
        few_shots=[],
        max_len=None,
    )["input_ids"]
    answer_ids = ds._encode_answer(
        tokenizer=t5_tokenizer,
        user_prompt=user,
        label=label,
        few_shots=[],
        max_len=None,
    )["input_ids"]

    assert t5_tokenizer.eos_token_id is not None
    idx = len(prompt_ids) - int(answer_ids[-1] == t5_tokenizer.eos_token_id)

    assert encoded["labels"] == answer_ids[idx:]
    assert all(label != SENTINEL_TOKEN for label in encoded["labels"])
    assert encoded["truncated"] is False


def test_tokenize_multinerd_encoder_int_label(
    nerd_data: Dataset,
    bert_tokenizer: PreTrainedTokenizerFast,
) -> None:
    ds = Multinerd(nerd_data, bert_tokenizer, "encoder")
    first = next(iter(ds.samples()))
    label = cast("MultinerdTag", first.label)
    row: MultinerdRow = {
        "sentence": first.user.split("\n")[0].removeprefix("Sentence: "),
        "entity": first.user.split("\n")[1].removeprefix("Entity: "),
        "label": label,
    }
    encoded = ds.tokenize(row, bert_tokenizer)

    assert encoded["labels"] == Multinerd.label2id[first.label]
    decoded = bert_tokenizer.decode(encoded["input_ids"])
    assert isinstance(decoded, str)
    assert (
        first.user.split("\n")[1].removeprefix("Entity: ").split()[0].lower()
        in decoded.lower()
    )
    assert "identify the ner tag of the entity" in decoded.lower()
    assert "possible tags are" not in decoded.lower()


def test_tokenize_over_request(
    oblique_data: Dataset,
    oblique_sample: OblRow,
    bert_tokenizer: PreTrainedTokenizerFast,
) -> None:
    ds = Obl(oblique_data, bert_tokenizer, "decoder")

    with pytest.raises(ValueError, match="more few shot examples than defined"):
        ds.tokenize(oblique_sample, bert_tokenizer, n_shot=99)


def test_samples_oblique(
    oblique_data: Dataset,
    bert_tokenizer: PreTrainedTokenizerFast,
) -> None:
    from eh30pt.datasets.registry import get_labels

    ds = Obl(oblique_data, bert_tokenizer, "decoder")
    samples = list(ds.samples())

    assert len(samples) == len(oblique_data)
    assert {sample.label for sample in samples} <= set(get_labels("oblique"))
    for idx, sample in enumerate(samples):
        row = oblique_data[idx]
        assert sample.user == Obl._format_user_prompt(row["sentence"], row["removed"])
        assert sample.label == row["label"]
        assert sample.id == idx


def test_samples_multinerd_expands_entities(
    nerd_data: Dataset,
    bert_tokenizer: PreTrainedTokenizerFast,
) -> None:
    from eh30pt.datasets.registry import get_labels

    ds = Multinerd(nerd_data, bert_tokenizer, "decoder")
    samples = list(ds.samples())

    assert samples
    assert {sample.label for sample in samples} <= set(get_labels("multinerd"))
    for sample in samples:
        assert sample.user.startswith("Sentence: ")
        assert "\nEntity: " in sample.user
        assert sample.user.endswith("\nTag:")
        assert sample.id is not None


def test_total_multinerd_counts_entities(
    nerd_data: Dataset,
    bert_tokenizer: PreTrainedTokenizerFast,
) -> None:
    ds = Multinerd(nerd_data, bert_tokenizer, "decoder")

    assert ds.total() == len(list(ds.samples()))


def test_getitem_tokenizes_row(
    oblique_data: Dataset,
    bert_tokenizer: PreTrainedTokenizerFast,
) -> None:
    ds = Obl(oblique_data, bert_tokenizer, "decoder")

    assert ds[0] == next(iter(ds))
    assert len(ds) == ds.total()


def test_classifier_reiterates_its_rows(
    classifier_rows: list[DiscontClassifierRow],
    bert_tokenizer: PreTrainedTokenizerFast,
) -> None:
    ds = DiscontClassifier(classifier_rows, bert_tokenizer, "decoder")

    first = list(ds.samples())
    expected = [
        FewShot(
            user=DiscontClassifier._format_user(row),
            label=row["label"],
            id=row["id"],
        )
        for row in classifier_rows
    ]

    assert first == expected

    second = list(ds.samples())

    assert [sample.user for sample in second] == [sample.user for sample in first]


def test_extract_rows_are_indexable(
    bert_tokenizer: PreTrainedTokenizerFast,
) -> None:
    rows: list[DiscontExtractRow] = [
        {"id": idx, "text": shot.user} for idx, shot in enumerate(_Extract.examples[:2])
    ]
    ds = DiscontExtract(rows, bert_tokenizer, "decoder")

    assert len(ds) == len(rows)
    assert list(ds.samples()) == [
        FewShot(user=row["text"], label="", id=row["id"]) for row in rows
    ]
    assert ds[0] == next(iter(ds))


def test_load_discont_extract_loads_rows(
    bert_tokenizer: PreTrainedTokenizerFast,
    database: None,
) -> None:
    _ = database
    splits = load_discont_extract(bert_tokenizer, "decoder", source="trigger")

    assert list(splits.train.samples()) == []
    assert list(splits.eval.samples()) == []
    assert len(splits.test) > 0
    assert len(splits.test) == len(list(splits.test.samples()))


def test_load_classifier_reads_exported_labels_without_a_server(
    bert_tokenizer: PreTrainedTokenizerFast, tmp_path: Path
) -> None:
    """Splits stream from an export-labels parquet, never touching the API."""
    n_tasks = 10
    export_labels = tuple(DiscontClassifier.en2et)
    real_shots = _Extract.examples
    fields = [cast("dict[str, str]", json.loads(shot.label)) for shot in real_shots]
    fallback_drug = next(field["drug"] for field in fields if field["drug"])
    labels = tmp_path / "labels.parquet"
    df = ExtractionFrameSchema.validate(
        pl.DataFrame(
            {
                "project_id": [5] * n_tasks,
                "task_id": list(range(1, n_tasks + 1)),
                "annotation_id": list(range(1, n_tasks + 1)),
                "result_id": [f"res-{i}" for i in range(1, n_tasks + 1)],
                "annotator": [42] * n_tasks,
                "from_name": "choices",
                "text": [real_shots[i % len(real_shots)].user for i in range(n_tasks)],
                "phrase": [
                    fields[i % len(fields)]["phrase"]
                    or real_shots[i % len(real_shots)].user
                    for i in range(n_tasks)
                ],
                "drug": [
                    fields[i % len(fields)]["drug"] or fallback_drug
                    for i in range(n_tasks)
                ],
                "reason": [fields[i % len(fields)]["reason"] for i in range(n_tasks)],
                "label": [
                    export_labels[i % len(export_labels)] for i in range(n_tasks)
                ],
            }
        )
    )

    df.write_parquet(labels)

    splits = load_discont_classifier(
        bert_tokenizer,
        "decoder",
        seed=0,
        data_path=labels,
    )

    samples = [sample for split in splits for sample in split.samples()]

    assert len(samples) == n_tasks
    assert {sample.label for sample in samples} == {
        DiscontClassifier.en2et[export_labels[i % len(export_labels)]]
        for i in range(n_tasks)
    }
    assert sorted(
        sample_id for sample in samples if (sample_id := sample.id) is not None
    ) == list(range(1, n_tasks + 1))


def test_load_classifier_rejects_a_missing_data_path(
    bert_tokenizer: PreTrainedTokenizerFast, tmp_path: Path
) -> None:
    """A labels path pointing nowhere fails at load, before any iteration."""
    with pytest.raises(FileNotFoundError):
        load_discont_classifier(
            bert_tokenizer, "decoder", seed=0, data_path=tmp_path / "missing.parquet"
        )

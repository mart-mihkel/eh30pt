"""Prompting and few-shot formatting integration tests with real tokenizers."""

from __future__ import annotations

from collections.abc import Iterator
from typing import TYPE_CHECKING, Final, cast

import polars as pl
import pytest
from datasets import Dataset

from eh30pt.constants import SENTINEL_TOKEN
from eh30pt.datasets.discontinuation.classify import (
    DiscontClassifier,
    DiscontClassifierRow,
    load_discont_classifier,
)
from eh30pt.datasets.discontinuation.extract import (
    DiscontExtract,
    DiscontExtractRow,
    load_discont_extract,
)
from eh30pt.datasets.multinerd import Multinerd, MultinerdRow
from eh30pt.datasets.oblique import Obl, OblRow
from eh30pt.labeling.loaders.extract import ExtractionFrameSchema
from eh30pt.types import Architecture, FewShot

if TYPE_CHECKING:
    from collections.abc import Iterator
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
    from sqlalchemy import create_engine as _create_engine
    from sqlalchemy.orm import Session as _Session

    from eh30pt.datasets.discontinuation.extract import DiscontExtract as _Extract
    from eh30pt.datasets.registry import get_labels as _get_labels
    from eh30pt.esthealth.queries import select_trigger_events as _select_trigger
    from eh30pt.settings import PostgresSettings as _PostgresSettings

    labels = _get_labels("discont-classifier/extract-labels")
    engine = _create_engine(_PostgresSettings().connection_url)
    try:
        with _Session(engine) as session:
            events = list(_select_trigger(session, page_size=100, limit=20))
    finally:
        engine.dispose()
    texts = [event.text for event in events if event.text][:_n_fixture_rows]
    assert len(texts) == _n_fixture_rows
    valid = [
        shot
        for shot in _Extract.extract_shots
        if shot.extractions["phrase"] and shot.extractions["drug"]
    ][:_n_fixture_rows]
    assert len(valid) == _n_fixture_rows
    return [
        {
            "id": events[idx].id,
            "text": texts[idx],
            "phrase": valid[idx].extractions["phrase"],
            "drug": valid[idx].extractions["drug"],
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


def test_set_epoch_ignored(
    oblique_data: Dataset,
    bert_tokenizer: PreTrainedTokenizerFast,
) -> None:
    ds = Obl(oblique_data, bert_tokenizer, "decoder")

    assert ds.set_epoch(3) is None


def test_getitem_forbidden(
    oblique_data: Dataset,
    bert_tokenizer: PreTrainedTokenizerFast,
) -> None:
    ds = Obl(oblique_data, bert_tokenizer, "decoder")

    with pytest.raises(TypeError, match="indexing"):
        ds[0]


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


def test_extract_streams_rows_lazily(
    bert_tokenizer: PreTrainedTokenizerFast,
) -> None:
    from eh30pt.datasets.discontinuation.extract import DiscontExtract as _Extract

    real: list[DiscontExtractRow] = [
        {"id": idx, "text": shot.user}
        for idx, shot in enumerate(_Extract.extract_shots[:2])
    ]
    calls: list[int] = []

    def _rows() -> Iterator[DiscontExtractRow]:
        calls.append(1)

        yield from real

    ds = DiscontExtract(_rows, bert_tokenizer, "decoder")

    assert calls == []

    first = list(ds.samples())

    assert calls == [1]
    assert first == [FewShot(user=row["text"], label="", id=row["id"]) for row in real]

    second = list(ds.samples())

    assert calls == [1, 1]
    assert [sample.user for sample in second] == [row["text"] for row in real]


def test_load_discont_extract_opens_no_connection(
    bert_tokenizer: PreTrainedTokenizerFast,
) -> None:
    splits = load_discont_extract(bert_tokenizer, "decoder", source="trigger")

    assert list(splits.train.samples()) == []
    assert list(splits.eval.samples()) == []


def test_load_classifier_reads_exported_labels_without_a_server(
    bert_tokenizer: PreTrainedTokenizerFast, tmp_path: Path
) -> None:
    """Splits stream from an export-labels parquet, never touching the API."""
    from eh30pt.datasets.discontinuation.extract import DiscontExtract as _Extract

    n_tasks = 10
    export_labels = tuple(DiscontClassifier.label2margend)
    real_shots = _Extract.extract_shots
    fallback_drug = next(
        s.extractions["drug"] for s in real_shots if s.extractions["drug"]
    )
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
                    real_shots[i % len(real_shots)].extractions["phrase"]
                    or real_shots[i % len(real_shots)].user
                    for i in range(n_tasks)
                ],
                "drug": [
                    real_shots[i % len(real_shots)].extractions["drug"] or fallback_drug
                    for i in range(n_tasks)
                ],
                "reason": [
                    real_shots[i % len(real_shots)].extractions["reason"]
                    for i in range(n_tasks)
                ],
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
        DiscontClassifier.label2margend[export_labels[i % len(export_labels)]]
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

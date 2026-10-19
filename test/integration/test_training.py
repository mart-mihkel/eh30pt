"""Trainer construction integration tests with real tokenizers."""

from __future__ import annotations

from typing import TYPE_CHECKING, Final, cast

import pytest

from eh30pt.constants import SENTINEL_TOKEN
from eh30pt.datasets.oblique import Obl
from eh30pt.modeling.collator import get_collator
from eh30pt.modeling.training.constraints import _label_completions
from eh30pt.modeling.training.spec import _build_spec
from eh30pt.types import DataSplits

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from transformers import PreTrainedModel, PreTrainedTokenizerFast, Trainer

    from eh30pt.modeling.training.spec import _Spec

_default_warmup: Final = 0.06
_custom_warmup: Final = 0.1
_train_steps: Final = 500
_eval_steps: Final = 50


@pytest.fixture
def oblique_splits(gpt2_tokenizer: PreTrainedTokenizerFast) -> DataSplits:
    """Return train, eval, and test splits over two real oblique rows."""
    from datasets import Dataset as _Dataset

    from eh30pt.datasets.oblique import Obl as _Obl

    raw = _Dataset.from_csv(_Obl.data_permalink, sep=";")
    raw = raw.rename_column("type", "label")
    raw = raw.map(lambda row: {"label": _Obl.label_entoet[row["label"]]})
    first_label = raw[0]["label"]
    second_idx = next(
        idx for idx in range(1, len(raw)) if raw[idx]["label"] != first_label
    )
    data = raw.select([0, second_idx])
    split = Obl(data, gpt2_tokenizer, "decoder")

    return DataSplits(train=split, eval=split, test=split)


def test_build_spec_hyperparameters(
    oblique_splits: DataSplits,
    oblique_gpt2_model: PreTrainedModel,
    gpt2_tokenizer: PreTrainedTokenizerFast,
) -> None:
    spec = _build_spec(
        model=oblique_gpt2_model,
        tokenizer=gpt2_tokenizer,
        data=oblique_splits,
        arch="decoder",
        collate_fn=get_collator(gpt2_tokenizer, "decoder"),
        metrics_fn=lambda _pred: {},
        output_dir="log",
        eval_steps=_eval_steps,
        train_steps=_train_steps,
    )

    assert spec.warmup_ratio == _default_warmup
    assert spec.train_steps == _train_steps
    assert spec.eval_steps == _eval_steps
    assert (
        _build_spec(
            model=oblique_gpt2_model,
            tokenizer=gpt2_tokenizer,
            data=oblique_splits,
            arch="decoder",
            collate_fn=get_collator(gpt2_tokenizer, "decoder"),
            metrics_fn=lambda _pred: {},
            output_dir="log",
            eval_steps=_eval_steps,
            train_steps=_train_steps,
            warmup_ratio=_custom_warmup,
        ).warmup_ratio
        == _custom_warmup
    )


def test_build_spec_aligns_save_with_eval_for_best_checkpoints(
    oblique_splits: DataSplits,
    oblique_gpt2_model: PreTrainedModel,
    gpt2_tokenizer: PreTrainedTokenizerFast,
) -> None:
    """load_best requires matching save and eval strategies."""
    spec = _build_spec(
        model=oblique_gpt2_model,
        tokenizer=gpt2_tokenizer,
        data=oblique_splits,
        arch="decoder",
        collate_fn=get_collator(gpt2_tokenizer, "decoder"),
        metrics_fn=lambda _pred: {},
        output_dir="log",
        eval_steps=_eval_steps,
        train_steps=_train_steps,
        checkpoints=True,
    )

    assert spec.load_best is True
    assert spec.save_strategy == spec.eval_strategy == "steps"
    assert spec.save_steps == _eval_steps


def test_build_spec_disables_saving_without_checkpoints(
    oblique_splits: DataSplits,
    oblique_gpt2_model: PreTrainedModel,
    gpt2_tokenizer: PreTrainedTokenizerFast,
) -> None:
    """Metrics-only runs must neither save checkpoints nor load the best."""
    spec = _build_spec(
        model=oblique_gpt2_model,
        tokenizer=gpt2_tokenizer,
        data=oblique_splits,
        arch="decoder",
        collate_fn=get_collator(gpt2_tokenizer, "decoder"),
        metrics_fn=lambda _pred: {},
        output_dir="log",
        eval_steps=_eval_steps,
        train_steps=_train_steps,
        checkpoints=False,
    )

    assert spec.load_best is False
    assert spec.save_strategy == "no"


def test_build_spec_evaluates_every_epoch(
    oblique_splits: DataSplits,
    oblique_gpt2_model: PreTrainedModel,
    gpt2_tokenizer: PreTrainedTokenizerFast,
) -> None:
    """'epoch' moves both eval and best-checkpoint saving to epoch boundaries."""
    spec = _build_spec(
        model=oblique_gpt2_model,
        tokenizer=gpt2_tokenizer,
        data=oblique_splits,
        arch="decoder",
        collate_fn=get_collator(gpt2_tokenizer, "decoder"),
        metrics_fn=lambda _pred: {},
        output_dir="log",
        eval_steps="epoch",
        train_steps=_train_steps,
        checkpoints=True,
    )

    assert spec.eval_strategy == "epoch"
    assert spec.save_strategy == "epoch"
    assert spec.load_best is True


def test_build_spec_epoch_eval_without_checkpoints(
    oblique_splits: DataSplits,
    oblique_gpt2_model: PreTrainedModel,
    gpt2_tokenizer: PreTrainedTokenizerFast,
) -> None:
    """Epoch eval without checkpointing still evaluates but never saves."""
    spec = _build_spec(
        model=oblique_gpt2_model,
        tokenizer=gpt2_tokenizer,
        data=oblique_splits,
        arch="decoder",
        collate_fn=get_collator(gpt2_tokenizer, "decoder"),
        metrics_fn=lambda _pred: {},
        output_dir="log",
        eval_steps="epoch",
        train_steps=_train_steps,
        checkpoints=False,
    )

    assert spec.eval_strategy == "epoch"
    assert spec.save_strategy == "no"
    assert spec.load_best is False


def test_label_completions_collects_unmasked_tails(
    oblique_splits: DataSplits,
) -> None:
    first, second = _label_completions(oblique_splits)

    assert first != second
    assert all(SENTINEL_TOKEN not in completion for completion in (first, second))


def test_label_completions_dedupes(
    gpt2_tokenizer: PreTrainedTokenizerFast,
) -> None:
    from datasets import Dataset as _Dataset

    from eh30pt.datasets.oblique import Obl as _Obl

    raw = _Dataset.from_csv(_Obl.data_permalink, sep=";")
    raw = raw.rename_column("type", "label")
    raw = raw.map(lambda row: {"label": _Obl.label_entoet[row["label"]]})
    data = raw.select(range(1))
    split = _Obl(data, gpt2_tokenizer, "decoder")
    splits = DataSplits(train=split, eval=split, test=split)

    assert len(_label_completions(splits)) == 1


def test_label_completions_skips_fully_masked() -> None:
    from datasets import Dataset as _Dataset

    from eh30pt.datasets.oblique import Obl as _Obl
    from eh30pt.modeling.tokenizer import load_tokenizer

    capped = load_tokenizer("hf-internal-testing/tiny-random-gpt2")
    capped.model_max_length = 8
    raw = _Dataset.from_csv(_Obl.data_permalink, sep=";")
    raw = raw.rename_column("type", "label")
    raw = raw.map(lambda row: {"label": _Obl.label_entoet[row["label"]]})
    data = raw.select(range(2))
    split = _Obl(data, capped, "decoder")
    splits = DataSplits(train=split, eval=split, test=split)

    assert _label_completions(splits) == ()


def test_build_spec_collects_completions_when_constrained(
    oblique_splits: DataSplits,
    oblique_gpt2_model: PreTrainedModel,
    gpt2_tokenizer: PreTrainedTokenizerFast,
) -> None:
    spec = _build_spec(
        model=oblique_gpt2_model,
        tokenizer=gpt2_tokenizer,
        data=oblique_splits,
        arch="decoder",
        collate_fn=get_collator(gpt2_tokenizer, "decoder"),
        metrics_fn=lambda _pred: {},
        output_dir="log",
        eval_steps=_eval_steps,
        train_steps=_train_steps,
        constrain=True,
    )

    assert spec.constrain is True
    assert spec.label_completions


def test_build_spec_drops_completions_when_unconstrained(
    oblique_splits: DataSplits,
    oblique_gpt2_model: PreTrainedModel,
    gpt2_tokenizer: PreTrainedTokenizerFast,
) -> None:
    """An empty completion set is what makes the trainers skip constraining."""
    spec = _build_spec(
        model=oblique_gpt2_model,
        tokenizer=gpt2_tokenizer,
        data=oblique_splits,
        arch="decoder",
        collate_fn=get_collator(gpt2_tokenizer, "decoder"),
        metrics_fn=lambda _pred: {},
        output_dir="log",
        eval_steps=_eval_steps,
        train_steps=_train_steps,
        constrain=False,
    )

    assert spec.constrain is False
    assert spec.label_completions == ()


def test_build_spec_is_unconstrained_by_default(
    oblique_splits: DataSplits,
    oblique_gpt2_model: PreTrainedModel,
    gpt2_tokenizer: PreTrainedTokenizerFast,
) -> None:
    spec = _build_spec(
        model=oblique_gpt2_model,
        tokenizer=gpt2_tokenizer,
        data=oblique_splits,
        arch="decoder",
        collate_fn=get_collator(gpt2_tokenizer, "decoder"),
        metrics_fn=lambda _pred: {},
        output_dir="log",
        eval_steps=_eval_steps,
        train_steps=_train_steps,
    )

    assert spec.constrain is False
    assert spec.label_completions == ()


def test_constrained_seq2seq_trainer_omits_processor_without_completions(
    gpt2_tokenizer: PreTrainedTokenizerFast,
    tmp_path: Path,
) -> None:
    """An empty completion set must not mask generation down to the pad token."""
    from datasets import Dataset as _Dataset
    from transformers import AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments

    from eh30pt.datasets.oblique import Obl as _Obl
    from eh30pt.modeling.training.seq2seq import ConstrainedSeq2SeqTrainer
    from test.conftest import models

    raw = _Dataset.from_csv(_Obl.data_permalink, sep=";")
    raw = raw.rename_column("type", "label")
    raw = raw.map(lambda row: {"label": _Obl.label_entoet[row["label"]]})
    data = raw.select(range(2))
    split = Obl(data, gpt2_tokenizer, "encoder-decoder")
    model = AutoModelForSeq2SeqLM.from_pretrained(models["t5"])
    args = Seq2SeqTrainingArguments(
        output_dir=str(tmp_path),
        report_to=[],
        predict_with_generate=True,
        max_steps=1,
    )

    def _gen_kwargs(completions: tuple[tuple[int, ...], ...]) -> dict[str, object]:
        trainer = ConstrainedSeq2SeqTrainer(
            model=model,
            args=args,
            data_collator=get_collator(gpt2_tokenizer, "encoder-decoder"),
            train_dataset=None,
            eval_dataset=split,
            compute_metrics=lambda _pred: {},
            callbacks=[],
            label_completions=completions,
            pad_token_id=cast("int", gpt2_tokenizer.pad_token_id),
        )

        return trainer._gen_kwargs

    assert _gen_kwargs(()) == {}
    assert "logits_processor" in _gen_kwargs(((1, 2),))


def test_get_trainer_builds_per_architecture(
    oblique_splits: DataSplits,
    oblique_gpt2_model: PreTrainedModel,
    gpt2_tokenizer: PreTrainedTokenizerFast,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from transformers import Trainer

    import eh30pt.modeling.training.factory as factory_mod
    from eh30pt.modeling.training.factory import get_trainer

    out = str(tmp_path)
    seen: list[str] = []

    def _metrics(_pred: object, *, compute_result: bool = True) -> dict[str, float]:
        return {"computed": 1.0} if compute_result else {}

    real_encoder = factory_mod._encoder_trainer
    real_decoder = factory_mod._decoder_trainer
    real_seq2seq = factory_mod._seq2seq_trainer

    def _record(
        kind: str, fn: Callable[[_Spec], Trainer]
    ) -> Callable[[_Spec], Trainer]:
        def _wrap(spec: _Spec) -> Trainer:
            seen.append(kind)
            return fn(spec)

        return _wrap

    monkeypatch.setattr(
        factory_mod, "_encoder_trainer", _record("encoder", real_encoder)
    )
    monkeypatch.setattr(
        factory_mod, "_decoder_trainer", _record("decoder", real_decoder)
    )
    monkeypatch.setattr(
        factory_mod, "_seq2seq_trainer", _record("seq2seq", real_seq2seq)
    )

    encoder = get_trainer(
        model=oblique_gpt2_model,
        tokenizer=gpt2_tokenizer,
        data=oblique_splits,
        arch="encoder",
        collate_fn=get_collator(gpt2_tokenizer, "encoder"),
        metrics_fn=_metrics,
        output_dir=out,
        eval_steps=0,
        train_steps=10,
    )

    assert type(encoder) is Trainer
    assert seen == ["encoder"]

    sentinel_decoder = object()
    sentinel_seq2seq = object()
    monkeypatch.setattr(factory_mod, "_decoder_trainer", lambda _spec: sentinel_decoder)
    monkeypatch.setattr(factory_mod, "_seq2seq_trainer", lambda _spec: sentinel_seq2seq)

    assert (
        get_trainer(
            model=oblique_gpt2_model,
            tokenizer=gpt2_tokenizer,
            data=oblique_splits,
            arch="decoder",
            collate_fn=get_collator(gpt2_tokenizer, "decoder"),
            metrics_fn=_metrics,
            output_dir=out,
            eval_steps=0,
            train_steps=10,
        )
        is sentinel_decoder
    )

    assert (
        get_trainer(
            model=oblique_gpt2_model,
            tokenizer=gpt2_tokenizer,
            data=oblique_splits,
            arch="encoder-decoder",
            collate_fn=get_collator(gpt2_tokenizer, "encoder-decoder"),
            metrics_fn=_metrics,
            output_dir=out,
            eval_steps=0,
            train_steps=10,
        )
        is sentinel_seq2seq
    )


def test_maybe_add_early_stopping_branches(
    oblique_splits: DataSplits,
    oblique_gpt2_model: PreTrainedModel,
    gpt2_tokenizer: PreTrainedTokenizerFast,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    from transformers import EarlyStoppingCallback

    from eh30pt.modeling.training.factory import get_trainer

    out = str(tmp_path)

    def _metrics(_pred: object, *, compute_result: bool = True) -> dict[str, float]:
        return {"computed": 1.0} if compute_result else {}

    with caplog.at_level("WARNING", logger="eh30pt"):
        get_trainer(
            model=oblique_gpt2_model,
            tokenizer=gpt2_tokenizer,
            data=oblique_splits,
            arch="encoder",
            collate_fn=get_collator(gpt2_tokenizer, "encoder"),
            metrics_fn=_metrics,
            output_dir=out,
            eval_steps=0,
            train_steps=10,
            patience=2,
        )

    assert "not using early stopping" in caplog.text
    caplog.clear()

    trainer = get_trainer(
        model=oblique_gpt2_model,
        tokenizer=gpt2_tokenizer,
        data=oblique_splits,
        arch="encoder",
        collate_fn=get_collator(gpt2_tokenizer, "encoder"),
        metrics_fn=_metrics,
        output_dir=out,
        eval_steps=_eval_steps,
        train_steps=10,
        patience=2,
    )

    assert any(
        isinstance(cb, EarlyStoppingCallback)
        for cb in trainer.callback_handler.callbacks
    )

    epoch_trainer = get_trainer(
        model=oblique_gpt2_model,
        tokenizer=gpt2_tokenizer,
        data=oblique_splits,
        arch="encoder",
        collate_fn=get_collator(gpt2_tokenizer, "encoder"),
        metrics_fn=_metrics,
        output_dir=out,
        eval_steps="epoch",
        train_steps=10,
        patience=2,
    )

    assert any(
        isinstance(cb, EarlyStoppingCallback)
        for cb in epoch_trainer.callback_handler.callbacks
    )

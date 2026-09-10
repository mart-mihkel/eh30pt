"""Trainer construction integration tests with real tokenizers."""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

import pytest
import torch
from datasets import Dataset

from eh30pt.constants import SENTINEL_TOKEN
from eh30pt.datasets.oblique import Obl
from eh30pt.modeling.collator import get_collator
from eh30pt.modeling.trainer import _build_spec, _label_completions
from eh30pt.types import DataSplits

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerFast

_default_warmup: Final = 0.06
_custom_warmup: Final = 0.1
_train_steps: Final = 500
_eval_steps: Final = 50


@pytest.fixture
def oblique_splits(gpt2_tokenizer: PreTrainedTokenizerFast) -> DataSplits:
    """Return train, eval, and test splits over two real oblique rows."""
    data = Dataset.from_list(
        [
            {"sentence": "Tere maailm", "removed": "Tere", "label": "vaba"},
            {"sentence": "Kass jookseb", "removed": "Kass", "label": "seotud"},
        ]
    )
    split = Obl(data, gpt2_tokenizer, "decoder")

    return DataSplits(train=split, eval=split, test=split)


def test_build_spec_hyperparameters(
    oblique_splits: DataSplits,
    gpt2_tokenizer: PreTrainedTokenizerFast,
) -> None:
    spec = _build_spec(
        model=torch.nn.Linear(2, 2),
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
            model=torch.nn.Linear(2, 2),
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
    gpt2_tokenizer: PreTrainedTokenizerFast,
) -> None:
    """load_best requires matching save and eval strategies."""
    spec = _build_spec(
        model=torch.nn.Linear(2, 2),
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


def test_label_completions_collects_unmasked_tails(
    oblique_splits: DataSplits,
) -> None:
    first, second = _label_completions(oblique_splits)

    assert first != second
    assert all(SENTINEL_TOKEN not in completion for completion in (first, second))


def test_label_completions_dedupes(
    gpt2_tokenizer: PreTrainedTokenizerFast,
) -> None:
    data = Dataset.from_list(
        [{"sentence": "Tere maailm", "removed": "Tere", "label": "vaba"}]
    )
    split = Obl(data, gpt2_tokenizer, "decoder")
    splits = DataSplits(train=split, eval=split, test=split)

    assert len(_label_completions(splits)) == 1


def test_label_completions_skips_fully_masked() -> None:
    from eh30pt.modeling.tokenizer import load_tokenizer

    capped = load_tokenizer("hf-internal-testing/tiny-random-gpt2")
    capped.model_max_length = 8
    data = Dataset.from_list(
        [
            {"sentence": "Tere maailm", "removed": "Tere", "label": "vaba"},
            {"sentence": "Kass jookseb", "removed": "Kass", "label": "seotud"},
        ]
    )
    split = Obl(data, capped, "decoder")
    splits = DataSplits(train=split, eval=split, test=split)

    assert _label_completions(splits) == ()

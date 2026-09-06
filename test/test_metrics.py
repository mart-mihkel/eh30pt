"""Tests for the metrics accumulator."""

from typing import TYPE_CHECKING, cast

import numpy as np
import pytest
from transformers import EvalPrediction

from eh30pt.modeling.metrics import MetricsAccumulator, _filter_gibberish

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerFast

    from eh30pt.types import Architecture


class _StubTokenizer:
    """Offline tokenizer double decoding ids as space-separated text."""

    pad_token_id = 0

    def batch_decode(
        self, sequences: list[np.ndarray], *, skip_special_tokens: bool = True
    ) -> list[str]:
        del skip_special_tokens
        return [" ".join(str(token) for token in seq) for seq in sequences]


def _tokenizer() -> PreTrainedTokenizerFast:
    return cast("PreTrainedTokenizerFast", _StubTokenizer())


def _accumulator(arch: Architecture) -> MetricsAccumulator:
    return MetricsAccumulator(_tokenizer(), arch)


def _eval_pred_logits(
    labels: list[int] | list[list[int]],
    preds: list[list[int]],
    n_classes: int,
) -> EvalPrediction:
    """EvalPrediction with logits argmaxing to `preds` (encoder/decoder)."""
    logits = np.zeros((len(preds), len(preds[0]), n_classes))
    for i, row in enumerate(preds):
        for j, token in enumerate(row):
            logits[i, j, token] = 10.0
    return EvalPrediction(predictions=logits, label_ids=np.array(labels))


def _eval_pred_cls(
    labels: list[int], preds: list[int], n_classes: int = 3
) -> EvalPrediction:
    """EvalPrediction with class logits (encoder)."""
    logits = np.zeros((len(preds), n_classes))
    for i, token in enumerate(preds):
        logits[i, token] = 10.0
    return EvalPrediction(predictions=logits, label_ids=np.array(labels))


def _eval_pred_ids(labels: list[list[int]], preds: list[list[int]]) -> EvalPrediction:
    """EvalPrediction with pre-decoded id arrays (seq2seq generation)."""
    return EvalPrediction(predictions=np.array(preds), label_ids=np.array(labels))


def test_filter_gibberish() -> None:
    ref = [" yes", " no", " yes"]
    pred = [" dog", " no", " yes"]
    assert _filter_gibberish(ref, pred) == ["<gibberish>", " no", " yes"]


def test_encoder_accumulates_batches() -> None:
    acc = _accumulator("encoder")

    assert acc(_eval_pred_cls([0, 2], [0, 2]), compute_result=False) == {}
    metrics = acc(_eval_pred_cls([1], [1]), compute_result=True)

    assert metrics["accuracy"] == 1.0
    assert {"precision", "recall", "f1"} <= set(metrics)


def test_instances_do_not_share_state() -> None:
    first = _accumulator("encoder")
    second = _accumulator("encoder")

    assert first(_eval_pred_cls([0], [1]), compute_result=False) == {}
    assert second(_eval_pred_cls([0], [0]), compute_result=True)["accuracy"] == 1.0
    assert first(_eval_pred_cls([1], [0]), compute_result=True)["accuracy"] == 0.0


def test_failed_computation_resets_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    acc = _accumulator("encoder")

    def _boom(*args: object, **kwargs: object) -> dict[str, float]:
        del args, kwargs
        msg = "boom"
        raise RuntimeError(msg)

    monkeypatch.setattr(
        "eh30pt.modeling.metrics._compute_classification_metrics", _boom
    )
    with pytest.raises(RuntimeError, match="boom"):
        acc(_eval_pred_cls([0], [1]), compute_result=True)

    monkeypatch.undo()
    assert acc(_eval_pred_cls([0], [0]), compute_result=True)["accuracy"] == 1.0


def test_decoder_ignores_prompt_positions() -> None:
    acc = _accumulator("decoder")

    # wrong prediction where the label is masked out must not count
    metrics = acc(
        _eval_pred_logits([[5, -100, 6]], [[9, 6, 6]], 10), compute_result=True
    )

    assert metrics["accuracy"] == 1.0


def test_decoder_trims_virtual_prefix_tokens() -> None:
    acc = _accumulator("decoder")

    # predictions run 2 timesteps longer than labels, as happens when the
    # model's output includes positions for prompt-tuning's virtual tokens
    metrics = acc(_eval_pred_logits([[0, 1]], [[9, 9, 0, 1]], 10), compute_result=True)

    assert "accuracy" in metrics


def test_seq2seq_masks_padding() -> None:
    acc = _accumulator("encoder-decoder")

    # trailing pad tokens must not count toward the metrics
    metrics = acc(_eval_pred_ids([[4, 4]], [[4, 4, 0]]), compute_result=True)

    assert metrics["accuracy"] == 1.0

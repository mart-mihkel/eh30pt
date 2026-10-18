"""Per-architecture Trainer construction."""

from typing import TYPE_CHECKING

from transformers import EarlyStoppingCallback

from eh30pt.logging import logger
from eh30pt.modeling.training.encoder import _encoder_trainer
from eh30pt.modeling.training.seq2seq import _seq2seq_trainer
from eh30pt.modeling.training.sft import _decoder_trainer
from eh30pt.modeling.training.spec import _build_spec

if TYPE_CHECKING:
    from pathlib import Path

    from torch.nn import Module
    from transformers import (
        DataCollator,
        PreTrainedTokenizerFast,
        Trainer,
        TrainerCallback,
    )

    from eh30pt.types import Architecture, ComputeMetricsFn, DataSplits, EvalSteps


def get_trainer(
    model: Module,
    *,
    tokenizer: PreTrainedTokenizerFast,
    data: DataSplits,
    arch: Architecture,
    collate_fn: DataCollator,
    metrics_fn: ComputeMetricsFn,
    output_dir: Path | str,
    eval_steps: EvalSteps,
    patience: int = 0,
    tolerance: float = 0.01,
    train_steps: int = 0,
    learning_rate: float = 5e-5,
    batch_size: int = 8,
    warmup_ratio: float = 0.06,
    run_name: str = "default",
    report_to: str = "none",
    checkpoints: bool = False,
    callbacks: tuple[TrainerCallback, ...] = (),
    constrain: bool = False,
) -> Trainer:
    """Build a Trainer for the architecture, adding early stopping if requested."""
    spec = _build_spec(
        model,
        tokenizer=tokenizer,
        data=data,
        arch=arch,
        collate_fn=collate_fn,
        metrics_fn=metrics_fn,
        output_dir=output_dir,
        eval_steps=eval_steps,
        train_steps=train_steps,
        learning_rate=learning_rate,
        batch_size=batch_size,
        warmup_ratio=warmup_ratio,
        run_name=run_name,
        report_to=report_to,
        checkpoints=checkpoints,
        callbacks=callbacks,
        constrain=constrain,
    )

    if arch == "encoder-decoder":
        trainer = _seq2seq_trainer(spec)
    elif arch == "decoder":
        trainer = _decoder_trainer(spec)
    else:
        trainer = _encoder_trainer(spec)

    _maybe_add_early_stopping(
        trainer,
        eval_enabled=spec.eval_strategy != "no",
        patience=patience,
        tolerance=tolerance,
    )

    return trainer


def _maybe_add_early_stopping(
    trainer: Trainer,
    *,
    eval_enabled: bool,
    patience: int,
    tolerance: float,
) -> None:
    """Attach early stopping when evaluation and patience are enabled."""
    if patience > 0 and not eval_enabled:
        logger.warning("not using early stopping because not running evaluation")
    elif eval_enabled and patience > 0:
        logger.info(
            "using early stopping with %d patience and %.2f tolerance",
            patience,
            tolerance,
        )

        trainer.add_callback(
            EarlyStoppingCallback(
                early_stopping_patience=patience,
                early_stopping_threshold=tolerance,
            )
        )

"""Trainer construction, training arguments, and seq2seq quirks."""

from typing import TYPE_CHECKING, cast

import torch
from transformers import (
    DataCollator,
    EarlyStoppingCallback,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    TrainerCallback,
)
from transformers.trainer import Trainer
from transformers.trainer_utils import get_last_checkpoint
from transformers.training_args import TrainingArguments

from eh30pt.constants import LOGDIR
from eh30pt.logging import logger

if TYPE_CHECKING:
    from collections.abc import Callable

    from datasets.dataset_dict import DatasetDict
    from torch.nn import Module
    from torch.utils.data import Dataset

    from eh30pt.types import Architecture, ComputeMetricsFn


def get_args(
    arch: Architecture,
    *,
    do_eval: bool,
    epochs: int = 0,
    learning_rate: float = 5e-5,
    batch_size: int = 8,
    run_name: str = "default",
    report_to: str = "none",
    checkpoints: bool = False,
) -> TrainingArguments:
    """Build Trainer arguments, using seq2seq args for encoder-decoder models."""
    have_cuda = torch.cuda.is_available()
    have_mps = torch.mps.is_available()
    have_accel = have_cuda or have_mps
    optim = "adamw_8bit" if have_cuda else "adamw_torch_fused"
    eval_strategy = "epoch" if do_eval else "no"
    save_strategy = "epoch" if checkpoints else "no"
    out_dir = LOGDIR / run_name

    if arch == "encoder-decoder":
        logger.debug("using seq2seq training args")
        args = Seq2SeqTrainingArguments(
            run_name=run_name,
            report_to=report_to,
            output_dir=str(out_dir),
            save_strategy=save_strategy,
            save_total_limit=1,
            eval_strategy=eval_strategy,
            eval_on_start=do_eval,
            batch_eval_metrics=True,
            remove_unused_columns=False,
            logging_steps=100,
            metric_for_best_model="f1",
            learning_rate=learning_rate,
            optim=optim,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            use_cpu=not have_accel,
            bf16_full_eval=have_accel,
            bf16=have_accel,
            predict_with_generate=True,
            generation_max_length=8,
        )
    else:
        logger.debug("using regular training args")
        args = TrainingArguments(
            run_name=run_name,
            report_to=report_to,
            output_dir=str(out_dir),
            save_strategy=save_strategy,
            save_total_limit=1,
            eval_strategy=eval_strategy,
            eval_on_start=do_eval,
            batch_eval_metrics=True,
            remove_unused_columns=False,
            logging_steps=100,
            metric_for_best_model="f1",
            learning_rate=learning_rate,
            optim=optim,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            use_cpu=not have_accel,
            bf16_full_eval=have_accel,
            bf16=have_accel,
        )

    return args


def get_trainer(
    model: Module,
    data: DatasetDict,
    arch: Architecture,
    collate_fn: DataCollator,
    metrics_fn: ComputeMetricsFn,
    *,
    do_eval: bool,
    patience: int = 0,
    tolerance: float = 0.01,
    epochs: int = 0,
    learning_rate: float = 5e-5,
    batch_size: int = 8,
    run_name: str = "default",
    report_to: str = "none",
    checkpoints: bool = False,
    callbacks: tuple[TrainerCallback, ...] = (),
) -> Trainer:
    """Build a Trainer for the architecture, adding early stopping if requested."""
    args = get_args(
        arch,
        do_eval=do_eval,
        epochs=epochs,
        learning_rate=learning_rate,
        batch_size=batch_size,
        run_name=run_name,
        report_to=report_to,
        checkpoints=checkpoints,
    )

    _metrics_fn = cast("Callable", metrics_fn)
    train_dataset = data.get("train")
    eval_dataset = cast("Dataset", data.get("validation"))

    if arch == "encoder-decoder":
        logger.debug("using seq2seq trainer")
        trainer_cls = Seq2SeqTrainer
    else:
        logger.debug("using regular trainer")
        trainer_cls = Trainer

    trainer = trainer_cls(
        args=args,
        model=model,
        data_collator=collate_fn,
        eval_dataset=eval_dataset,
        train_dataset=train_dataset,
        compute_metrics=_metrics_fn,
        callbacks=list(callbacks),
    )

    if patience > 0 and not do_eval:
        logger.warning("not using early stopping because not running evaluation")
    elif do_eval and patience > 0:
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

    return trainer


def find_checkpoint(run_name: str) -> str | None:
    """Find the latest checkpoint of a previous run of `run_name`, if any exists."""
    out_dir = LOGDIR / run_name
    if not out_dir.is_dir():
        logger.warning("no output dir '%s', training from scratch", out_dir)
        return None

    checkpoint = get_last_checkpoint(str(out_dir))
    if checkpoint is None:
        logger.warning("no checkpoint in '%s', training from scratch", out_dir)
        return None

    logger.info("resuming from checkpoint '%s'", checkpoint)
    return checkpoint

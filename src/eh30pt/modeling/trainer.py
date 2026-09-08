"""Per-architecture Trainer construction."""

from typing import TYPE_CHECKING, NamedTuple, cast

import torch
from transformers import (
    DataCollator,
    EarlyStoppingCallback,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    TrainerCallback,
)
from transformers.trainer import Trainer
from transformers.training_args import TrainingArguments
from trl import SFTConfig, SFTTrainer

from eh30pt.logging import logger

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from datasets.arrow_dataset import Dataset as HFDataset
    from datasets.dataset_dict import DatasetDict
    from peft import PeftModel
    from torch.nn import Module
    from torch.utils.data import Dataset
    from transformers import PreTrainedModel, PreTrainedTokenizerFast

    from eh30pt.types import Architecture, ComputeMetricsFn


class _Spec(NamedTuple):
    """Resolved datasets, hyperparameters, and derived loop settings."""

    model: Module
    tokenizer: PreTrainedTokenizerFast
    train_dataset: HFDataset | None
    eval_dataset: Dataset
    collate_fn: DataCollator
    metrics_fn: Callable
    output_dir: str
    do_eval: bool
    epochs: int
    learning_rate: float
    batch_size: int
    run_name: str
    report_to: str
    checkpoints: bool
    callbacks: list[TrainerCallback]
    optim: str
    eval_strategy: str
    save_strategy: str
    load_best: bool
    have_accel: bool


def _encoder_trainer(spec: _Spec) -> Trainer:
    """Build the classification Trainer."""
    logger.debug("using regular trainer")
    args = TrainingArguments(
        run_name=spec.run_name,
        report_to=spec.report_to,
        output_dir=spec.output_dir,
        save_strategy=spec.save_strategy,
        save_total_limit=1,
        eval_strategy=spec.eval_strategy,
        eval_on_start=spec.do_eval,
        batch_eval_metrics=True,
        remove_unused_columns=False,
        logging_steps=100,
        load_best_model_at_end=spec.load_best,
        metric_for_best_model="f1",
        learning_rate=spec.learning_rate,
        optim=spec.optim,
        num_train_epochs=spec.epochs,
        per_device_train_batch_size=spec.batch_size,
        per_device_eval_batch_size=spec.batch_size,
        use_cpu=not spec.have_accel,
        bf16_full_eval=spec.have_accel,
        bf16=spec.have_accel,
    )

    return Trainer(
        args=args,
        model=spec.model,
        data_collator=spec.collate_fn,
        eval_dataset=spec.eval_dataset,
        train_dataset=spec.train_dataset,
        compute_metrics=spec.metrics_fn,
        callbacks=spec.callbacks,
    )


def _decoder_trainer(spec: _Spec) -> Trainer:
    """Build the SFT Trainer for causal models."""
    logger.debug("using SFT trainer")
    args = SFTConfig(
        run_name=spec.run_name,
        report_to=spec.report_to,
        output_dir=spec.output_dir,
        save_strategy=spec.save_strategy,
        save_total_limit=1,
        eval_strategy=spec.eval_strategy,
        eval_on_start=spec.do_eval,
        batch_eval_metrics=True,
        remove_unused_columns=False,
        logging_steps=100,
        load_best_model_at_end=spec.load_best,
        metric_for_best_model="f1",
        learning_rate=spec.learning_rate,
        optim=spec.optim,
        num_train_epochs=spec.epochs,
        per_device_train_batch_size=spec.batch_size,
        per_device_eval_batch_size=spec.batch_size,
        use_cpu=not spec.have_accel,
        bf16_full_eval=spec.have_accel,
        bf16=spec.have_accel,
        # Datasets are already tokenized with -100-masked labels,
        # skip TRL's text preparation and train on labels as is.
        dataset_kwargs={"skip_prepare_dataset": True},
        packing=False,
        # Chunked loss skips full logits during eval,
        # leaving logit-based metrics with empty predictions.
        loss_type="nll",
    )

    trainer = SFTTrainer(
        model=cast("PreTrainedModel | PeftModel", spec.model),
        args=args,
        data_collator=spec.collate_fn,
        train_dataset=spec.train_dataset,
        eval_dataset=cast("HFDataset | None", spec.eval_dataset),
        processing_class=spec.tokenizer,
        compute_metrics=spec.metrics_fn,
        callbacks=spec.callbacks,
    )

    return cast("Trainer", trainer)


def _seq2seq_trainer(spec: _Spec) -> Trainer:
    """Build the seq2seq Trainer for encoder-decoder models."""
    logger.debug("using seq2seq trainer")
    args = Seq2SeqTrainingArguments(
        run_name=spec.run_name,
        report_to=spec.report_to,
        output_dir=spec.output_dir,
        save_strategy=spec.save_strategy,
        save_total_limit=1,
        eval_strategy=spec.eval_strategy,
        eval_on_start=spec.do_eval,
        batch_eval_metrics=True,
        remove_unused_columns=False,
        logging_steps=100,
        load_best_model_at_end=spec.load_best,
        metric_for_best_model="f1",
        learning_rate=spec.learning_rate,
        optim=spec.optim,
        num_train_epochs=spec.epochs,
        per_device_train_batch_size=spec.batch_size,
        per_device_eval_batch_size=spec.batch_size,
        use_cpu=not spec.have_accel,
        bf16_full_eval=spec.have_accel,
        bf16=spec.have_accel,
        predict_with_generate=True,
        generation_max_length=8,
    )

    return Seq2SeqTrainer(
        args=args,
        model=spec.model,
        data_collator=spec.collate_fn,
        eval_dataset=spec.eval_dataset,
        train_dataset=spec.train_dataset,
        compute_metrics=spec.metrics_fn,
        callbacks=spec.callbacks,
    )


def _maybe_add_early_stopping(
    trainer: Trainer,
    *,
    do_eval: bool,
    patience: int,
    tolerance: float,
) -> None:
    """Attach early stopping when evaluation and patience are enabled."""
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


def get_trainer(
    model: Module,
    *,
    tokenizer: PreTrainedTokenizerFast,
    data: DatasetDict,
    arch: Architecture,
    collate_fn: DataCollator,
    metrics_fn: ComputeMetricsFn,
    output_dir: Path | str,
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
    """
    Build a Trainer for the architecture, adding early stopping if requested.

    Parameters
    ----------
    model:
        Model to train or evaluate.
    tokenizer:
        Tokenizer passed as the SFT processing class for decoders.
    data:
        Tokenized dataset splits (`train`, `validation`).
    arch:
        Model architecture, selecting regular, seq2seq, or SFT Trainer.
    collate_fn:
        Data collator batching tokenized examples.
    metrics_fn:
        Metric function scoring evaluation predictions.
    output_dir:
        Directory for checkpoints and trainer state.
    do_eval:
        Run evaluation during training.
    patience:
        Eval epochs without improvement before stopping, 0 disables.
    tolerance:
        Minimum eval improvement resetting the patience counter.
    epochs:
        Number of training epochs.
    learning_rate:
        Optimizer learning rate.
    batch_size:
        Training and evaluation batch size.
    run_name:
        Display name reported to trackers.
    report_to:
        Tracker integrations enabled on the Trainer.
    checkpoints:
        Save epoch checkpoints (also required to restore the best model).
    callbacks:
        Extra Trainer callbacks.

    Returns
    -------
    Trainer
        Configured Trainer for the architecture.

    """
    have_cuda = torch.cuda.is_available()
    have_mps = torch.mps.is_available()
    have_accel = have_cuda or have_mps
    do_checkpoints = do_eval and checkpoints

    spec = _Spec(
        model=model,
        tokenizer=tokenizer,
        train_dataset=data.get("train"),
        eval_dataset=cast("Dataset", data.get("validation")),
        collate_fn=collate_fn,
        metrics_fn=cast("Callable", metrics_fn),
        output_dir=str(output_dir),
        do_eval=do_eval,
        epochs=epochs,
        learning_rate=learning_rate,
        batch_size=batch_size,
        run_name=run_name,
        report_to=report_to,
        checkpoints=checkpoints,
        callbacks=list(callbacks),
        optim="adamw_8bit" if have_cuda else "adamw_torch_fused",
        eval_strategy="epoch" if do_eval else "no",
        save_strategy="epoch" if checkpoints else "no",
        load_best=do_checkpoints,
        have_accel=have_accel,
    )

    if arch == "encoder-decoder":
        trainer = _seq2seq_trainer(spec)
    elif arch == "decoder":
        trainer = _decoder_trainer(spec)
    else:
        trainer = _encoder_trainer(spec)

    _maybe_add_early_stopping(
        trainer,
        do_eval=do_eval,
        patience=patience,
        tolerance=tolerance,
    )

    return trainer

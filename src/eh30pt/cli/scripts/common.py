"""Shared training/evaluation flow backed by an open MLflow run."""

from typing import TYPE_CHECKING, cast

from eh30pt.logging import logger
from eh30pt.modeling.trainer import get_trainer

if TYPE_CHECKING:
    from datasets.dataset_dict import DatasetDict
    from peft import PeftModel
    from torch.utils.data import Dataset
    from transformers import DataCollator, PreTrainedModel, PreTrainedTokenizerFast

    from eh30pt.tracking import RunContext
    from eh30pt.types import Architecture, ComputeMetricsFn


def train_and_evaluate(
    ctx: RunContext,
    model: PreTrainedModel | PeftModel,
    *,
    tokenizer: PreTrainedTokenizerFast,
    data: DatasetDict,
    arch: Architecture,
    collate_fn: DataCollator,
    metrics_fn: ComputeMetricsFn,
    registered_name: str,
    eval_steps: int,
    patience: int,
    tolerance: float,
    train_steps: int,
    learning_rate: float,
    batch_size: int,
    warmup_ratio: float,
) -> dict[str, float]:
    """Train model under run context, then evaluate, log, and register the best."""
    trainer = get_trainer(
        model=model,
        tokenizer=tokenizer,
        data=data,
        arch=arch,
        collate_fn=collate_fn,
        metrics_fn=metrics_fn,
        output_dir=ctx.dir,
        eval_steps=eval_steps,
        patience=patience,
        tolerance=tolerance,
        train_steps=train_steps,
        learning_rate=learning_rate,
        batch_size=batch_size,
        warmup_ratio=warmup_ratio,
        run_name=ctx.run_name or "eh30pt",
        report_to="mlflow",
        checkpoints=True,
    )

    logger.debug("starting trainer")
    trainer.train(resume_from_checkpoint=ctx.find_checkpoint())

    logger.debug("starting test eval")
    test = cast("Dataset", data["test"])
    test_metrics = trainer.evaluate(test, metric_key_prefix="test")

    ctx.finish_training(
        registered_name,
        model=model,
        tokenizer=tokenizer,
        test_metrics=dict(test_metrics),
    )

    return test_metrics


def evaluate_and_log(
    ctx: RunContext,
    model: PreTrainedModel | PeftModel,
    *,
    tokenizer: PreTrainedTokenizerFast,
    data: DatasetDict,
    arch: Architecture,
    collate_fn: DataCollator,
    metrics_fn: ComputeMetricsFn,
    batch_size: int,
) -> dict[str, float]:
    """Evaluate model on the test split and log the returned metrics."""
    trainer = get_trainer(
        model=model,
        tokenizer=tokenizer,
        data=data,
        arch=arch,
        collate_fn=collate_fn,
        metrics_fn=metrics_fn,
        output_dir=ctx.dir,
        eval_steps=0,
        batch_size=batch_size,
        run_name=ctx.run_name or "eh30pt",
        report_to="mlflow",
    )

    logger.debug("starting test eval")
    test = cast("Dataset", data["test"])
    test_metrics = trainer.evaluate(test, metric_key_prefix="test")
    ctx.log_metrics(dict(test_metrics))
    return test_metrics


def log_prompt_version(
    ctx: RunContext,
    dataset: str,
    arch: Architecture,
    system_prompt: str,
) -> None:
    """Register the system prompt, recording its version on fresh runs."""
    prompt_version = ctx.register_prompt_version(
        f"{dataset}-{arch}",
        system_prompt,
        commit_message=f"tracked from run {ctx.run_name}",
    )

    if not ctx.resumed:
        ctx.log_params({"prompt_version": prompt_version.version})

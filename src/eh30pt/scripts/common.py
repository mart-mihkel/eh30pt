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
    do_eval: bool,
    patience: int,
    tolerance: float,
    epochs: int,
    learning_rate: float,
    batch_size: int,
    warmup_ratio: float,
) -> dict[str, float]:
    """
    Train `model` under `ctx`, then evaluate, log, and register the best.

    Checkpoints land in the run's directory and training resumes from its
    latest checkpoint when present.

    Parameters
    ----------
    ctx : RunContext
        Open run tracking the training.
    model : PreTrainedModel | PeftModel
        Model (or PEFT adapter) to train.
    tokenizer : PreTrainedTokenizerFast
        Tokenizer saved with the final model.
    data : DatasetDict
        Tokenized dataset splits, including `test`.
    arch : Architecture
        Model architecture, selecting the trainer flavor.
    collate_fn : DataCollator
        Data collator batching tokenized examples.
    metrics_fn : ComputeMetricsFn
        Metric function scoring evaluation predictions.
    registered_name : str
        Registered model name for the best-model comparison.
    do_eval : bool
        Run evaluation during training.
    patience : int
        Eval epochs without improvement before stopping, 0 disables.
    tolerance : float
        Minimum eval improvement resetting the patience counter.
    epochs : int
        Number of training epochs.
    learning_rate : float
        Optimizer learning rate.
    batch_size : int
        Training and evaluation batch size.
    warmup_ratio : float
        Fraction of training steps for learning rate warmup.

    Returns
    -------
    dict[str, float]
        Test metrics under the `test_` prefix.

    """
    trainer = get_trainer(
        model=model,
        tokenizer=tokenizer,
        data=data,
        arch=arch,
        collate_fn=collate_fn,
        metrics_fn=metrics_fn,
        output_dir=ctx.dir,
        do_eval=do_eval,
        patience=patience,
        tolerance=tolerance,
        epochs=epochs,
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
    """
    Evaluate `model` on the test split and log the returned metrics.

    Parameters
    ----------
    ctx : RunContext
        Open run tracking the evaluation.
    model : PreTrainedModel | PeftModel
        Model (or PEFT adapter) to evaluate.
    tokenizer : PreTrainedTokenizerFast
        Tokenizer passed as the SFT processing class for decoders.
    data : DatasetDict
        Tokenized dataset splits, including `test`.
    arch : Architecture
        Model architecture, selecting the trainer flavor.
    collate_fn : DataCollator
        Data collator batching tokenized examples.
    metrics_fn : ComputeMetricsFn
        Metric function scoring evaluation predictions.
    batch_size : int
        Evaluation batch size.

    Returns
    -------
    dict[str, float]
        Test metrics under the `test_` prefix.

    """
    trainer = get_trainer(
        model=model,
        tokenizer=tokenizer,
        data=data,
        arch=arch,
        collate_fn=collate_fn,
        metrics_fn=metrics_fn,
        output_dir=ctx.dir,
        do_eval=False,
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
    """
    Register the system prompt, recording its version on fresh runs.

    Parameters
    ----------
    ctx : RunContext
        Open run to link the prompt version to.
    dataset : str
        Dataset naming the prompt registry entry together with `arch`.
    arch : Architecture
        Architecture naming the prompt registry entry together with `dataset`.
    system_prompt : str
        Prompt template text to commit.

    """
    prompt_version = ctx.register_prompt_version(
        f"{dataset}-{arch}",
        system_prompt,
        commit_message=f"tracked from run {ctx.run_name}",
    )

    if not ctx.resumed:
        ctx.log_params({"prompt_version": prompt_version.version})

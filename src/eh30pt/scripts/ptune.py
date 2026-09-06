"""Prompt-tune a pretrained model and run test evaluation."""

from typing import TYPE_CHECKING, cast

import mlflow
from transformers import AutoConfig

from eh30pt.datasets.registry import get_sys_prompt, load_data
from eh30pt.logging import logger
from eh30pt.modeling.arch import infer_architecture
from eh30pt.modeling.collator import get_collator
from eh30pt.modeling.metrics import MetricsAccumulator
from eh30pt.modeling.tokenizer import load_tokenizer
from eh30pt.modeling.trainer import find_checkpoint, get_trainer
from eh30pt.modeling.tuning import get_n_virtual, prepare_pt_model
from eh30pt.scripts.tracking import run_name, start_run

if TYPE_CHECKING:
    from peft import PromptTuningConfig
    from torch.utils.data import Dataset

    from eh30pt.types import Architecture, DatasetName, PrefixInit


def prompt_tune(
    model_path: str,
    dataset: DatasetName,
    prefix_init: PrefixInit,
    n_shot: int,
    *,
    arch: Architecture | None = None,
    train_samples: int | None,
    val_samples: int | None,
    do_eval: bool,
    patience: int,
    tolerance: float,
    resume: bool = True,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    mlflow_experiment: str,
    mlflow_run_name: str | None,
    mlflow_tracking_uri: str,
) -> None:
    """Prompt-tune `model_path` on `dataset` and evaluate on the test split."""
    logger.info("loading config for '%s'", model_path)
    config = AutoConfig.from_pretrained(model_path)
    arch = infer_architecture(config, arch)

    tokenizer = load_tokenizer(model_path)
    collate_fn = get_collator(tokenizer, arch)
    metrics_fn = MetricsAccumulator(tokenizer, arch)

    sys_prompt = get_sys_prompt(dataset, tokenizer, arch)
    n_virtual = get_n_virtual(tokenizer, sys_prompt)

    ds = load_data(
        tokenizer,
        dataset,
        arch,
        n_shot,
        n_train_samples=train_samples,
        n_val_samples=val_samples,
        n_virtual=n_virtual,
    )
    data = ds.data

    if dataset in {"boolq", "wic"}:
        logger.warning("using superglue validation data for test, labels are private")
        data["test"] = data["validation"]

    logger.info(
        "preparing prompt tuning model for '%s' with %s prefix initialization",
        model_path,
        prefix_init,
    )

    model = prepare_pt_model(prefix_init, tokenizer, model_path, arch, ds)

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    ptcfg = cast("PromptTuningConfig", model.peft_config["default"])

    method = f"prompt-tune-{prefix_init}"
    if mlflow_run_name is None:
        mlflow_run_name = run_name(dataset, model_path, method, train_samples)

    logger.info("total parameters %d", total)
    logger.info("trainable parameters %d", trainable)
    logger.info("virtual tokens %d", ptcfg.num_virtual_tokens)
    logger.info("tracking '%s' of experiment '%s'", mlflow_run_name, mlflow_experiment)

    checkpoint = find_checkpoint(mlflow_run_name) if resume else None

    start_run(
        mlflow_experiment,
        mlflow_run_name,
        mlflow_tracking_uri,
        resume=checkpoint is not None,
    )

    mlflow.log_param("n_shot", n_shot)
    mlflow.log_param("dataset", dataset)
    mlflow.log_param("architecture", arch)
    mlflow.log_param("base_model", model_path)
    mlflow.log_param("prefix_init", prefix_init)
    mlflow.log_param("system_prompt", ds.system_prompt)
    mlflow.log_param("method", method)
    mlflow.log_param("n_virtual", ptcfg.num_virtual_tokens)
    mlflow.log_metric("train_samples", len(data["train"]))
    mlflow.log_metric("validation_samples", len(data["validation"]) if do_eval else 0)
    mlflow.log_metric("test_samples", len(data["test"]))
    mlflow.log_metric("total_parameters", total)
    mlflow.log_metric("trainable_parameters", trainable)

    trainer = get_trainer(
        model=model,
        data=data,
        arch=arch,
        collate_fn=collate_fn,
        metrics_fn=metrics_fn,
        do_eval=do_eval,
        patience=patience,
        tolerance=tolerance,
        epochs=epochs,
        learning_rate=learning_rate,
        batch_size=batch_size,
        run_name=mlflow_run_name,
        report_to="mlflow",
        checkpoints=True,
    )

    logger.debug("start trainer")
    trainer.train(resume_from_checkpoint=checkpoint)

    logger.debug("start test eval")
    test = cast("Dataset", data["test"])
    trainer.evaluate(test, metric_key_prefix="test")

    mlflow.end_run()

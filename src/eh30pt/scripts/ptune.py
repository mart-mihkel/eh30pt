"""Prompt-tune a pretrained model and run test evaluation."""

from typing import TYPE_CHECKING, cast

from transformers import AutoConfig

from eh30pt.datasets.registry import get_sys_prompt, load_data
from eh30pt.logging import logger
from eh30pt.modeling.arch import infer_architecture
from eh30pt.modeling.collator import get_collator
from eh30pt.modeling.metrics import MetricsAccumulator
from eh30pt.modeling.tokenizer import cap_tokenizer_length, load_tokenizer
from eh30pt.modeling.tuning import get_n_virtual, prepare_pt_model
from eh30pt.scripts.common import log_prompt_version, train_and_evaluate
from eh30pt.tracking import build_run_name, open_run

if TYPE_CHECKING:
    from peft import PromptTuningConfig

    from eh30pt.types import Architecture, DatasetName, PrefixInit, PromptTuneRunConfig


def main(
    model_path: str,
    dataset: DatasetName,
    prefix_init: PrefixInit,
    n_shot: int,
    *,
    arch: Architecture | None = None,
    eval_steps: int,
    patience: int,
    tolerance: float,
    resume: bool = True,
    train_steps: int,
    batch_size: int,
    learning_rate: float,
    warmup_ratio: float,
    mlflow_experiment: str,
    mlflow_run_name: str | None,
    mlflow_tracking_uri: str,
    seed: int | None = None,
) -> None:
    """Prompt-tune `model_path` on `dataset` and evaluate on the test split."""
    logger.info("loading config for '%s'", model_path)
    config = AutoConfig.from_pretrained(model_path)
    arch = infer_architecture(config, arch)

    tokenizer = load_tokenizer(model_path)
    cap_tokenizer_length(tokenizer, config)

    collate_fn = get_collator(tokenizer, arch)
    metrics_fn = MetricsAccumulator(tokenizer, arch)

    sys_prompt = get_sys_prompt(dataset, tokenizer, arch)
    n_virtual = get_n_virtual(tokenizer, sys_prompt)

    data = load_data(
        tokenizer,
        dataset,
        arch,
        n_shot=n_shot,
        n_virtual=n_virtual,
    )

    tokenized = data.data

    if dataset in {"boolq", "wic"}:
        logger.warning("using superglue validation data for test, labels are private")
        tokenized["test"] = tokenized["validation"]

    logger.info(
        "preparing prompt tuning model for '%s' with %s prefix initialization",
        model_path,
        prefix_init,
    )

    model = prepare_pt_model(prefix_init, tokenizer, model_path, arch, data)

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    ptcfg = cast("PromptTuningConfig", model.peft_config["default"])

    method = f"prompt-tune-{prefix_init}"
    if mlflow_run_name is None:
        mlflow_run_name = build_run_name(dataset, model_path, method, None)

    logger.info("total parameters %d", total)
    logger.info("trainable parameters %d", trainable)
    logger.info("virtual tokens %d", ptcfg.num_virtual_tokens)
    logger.info("tracking '%s' of experiment '%s'", mlflow_run_name, mlflow_experiment)

    run_config: PromptTuneRunConfig = {
        "method": method,
        "dataset": dataset,
        "architecture": arch,
        "base_model": model_path,
        "prefix_init": prefix_init,
        "n_virtual": ptcfg.num_virtual_tokens,
        "n_shot": n_shot,
        "eval_steps": eval_steps,
        "patience": patience,
        "tolerance": tolerance,
        "train_steps": train_steps,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "warmup_ratio": warmup_ratio,
        "seed": seed,
        "system_prompt": data.system_prompt,
        "train_samples": len(tokenized["train"]),
        "validation_samples": len(tokenized["validation"]) if eval_steps > 0 else 0,
        "test_samples": len(tokenized["test"]),
        "total_parameters": total,
        "trainable_parameters": trainable,
    }

    with open_run(
        mlflow_experiment,
        mlflow_run_name,
        mlflow_tracking_uri,
        resume=resume,
        config=run_config,
    ) as ctx:
        log_prompt_version(ctx, dataset, arch, data.system_prompt)
        train_and_evaluate(
            ctx,
            model,
            tokenizer=tokenizer,
            data=tokenized,
            arch=arch,
            collate_fn=collate_fn,
            metrics_fn=metrics_fn,
            registered_name=f"{mlflow_experiment}-{dataset}-{arch}-{method}",
            eval_steps=eval_steps,
            patience=patience,
            tolerance=tolerance,
            train_steps=train_steps,
            learning_rate=learning_rate,
            batch_size=batch_size,
            warmup_ratio=warmup_ratio,
        )

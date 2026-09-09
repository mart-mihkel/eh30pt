"""Run few-shot test evaluation for a pretrained model."""

from typing import TYPE_CHECKING, cast

from transformers import AutoConfig

from eh30pt.cli.scripts.common import evaluate_and_log, log_prompt_version
from eh30pt.datasets.registry import load_data
from eh30pt.logging import logger
from eh30pt.modeling.arch import infer_architecture
from eh30pt.modeling.collator import get_collator
from eh30pt.modeling.loader import load_model
from eh30pt.modeling.metrics import MetricsAccumulator
from eh30pt.modeling.tokenizer import cap_tokenizer_length, load_tokenizer
from eh30pt.tracking import build_run_name, open_run

if TYPE_CHECKING:
    from datasets.splits import Split

    from eh30pt.types import Architecture, DatasetName, FewShotRunConfig


def main(
    model_path: str,
    dataset: DatasetName,
    n_shot: int,
    *,
    arch: Architecture | None = None,
    batch_size: int,
    mlflow_experiment: str,
    mlflow_run_name: str | None,
    mlflow_tracking_uri: str,
    seed: int | None = None,
) -> None:
    """Run few-shot test evaluation of `model_path` on `dataset`."""
    logger.info("loading config for '%s'", model_path)
    config = AutoConfig.from_pretrained(model_path)
    arch = infer_architecture(config, arch)

    tokenizer = load_tokenizer(model_path)
    cap_tokenizer_length(tokenizer, config)

    collate_fn = get_collator(tokenizer, arch)
    metrics_fn = MetricsAccumulator(tokenizer, arch)

    split = cast("Split", {"test": "test"})
    data = load_data(tokenizer, dataset, arch, n_shot=n_shot, split=split)
    tokenized = data.data

    if dataset in {"boolq", "wic"}:
        logger.warning("using superglue validation data, test labels are private")
        tokenized["test"] = tokenized["validation"]

    logger.info("loading '%s'", model_path)
    model = load_model(tokenizer, model_path, data, arch, head_only=False)

    total = sum(p.numel() for p in model.parameters())

    method = f"{n_shot}-shot"
    if mlflow_run_name is None:
        mlflow_run_name = build_run_name(dataset, model_path, method, train_samples=0)

    logger.info("tracking '%s' of experiment '%s'", mlflow_run_name, mlflow_experiment)

    run_config: FewShotRunConfig = {
        "method": method,
        "dataset": dataset,
        "architecture": arch,
        "base_model": model_path,
        "n_shot": n_shot,
        "batch_size": batch_size,
        "seed": seed,
        "one_conv": False,
        "system_prompt": data.system_prompt,
        "train_samples": 0,
        "validation_samples": 0,
        "test_samples": len(tokenized["test"]),
        "total_parameters": total,
        "trainable_parameters": 0,
    }

    with open_run(
        mlflow_experiment,
        mlflow_run_name,
        mlflow_tracking_uri,
        resume=False,
        config=run_config,
    ) as ctx:
        log_prompt_version(ctx, dataset, arch, data.system_prompt)
        evaluate_and_log(
            ctx,
            model,
            tokenizer=tokenizer,
            data=tokenized,
            arch=arch,
            collate_fn=collate_fn,
            metrics_fn=metrics_fn,
            batch_size=batch_size,
        )

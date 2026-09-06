"""Fine-tune a pretrained model and run test evaluation."""

from typing import TYPE_CHECKING

from transformers import AutoConfig

from eh30pt.datasets.registry import load_data
from eh30pt.logging import logger
from eh30pt.modeling.arch import infer_architecture
from eh30pt.modeling.collator import get_collator
from eh30pt.modeling.loader import load_model
from eh30pt.modeling.metrics import MetricsAccumulator
from eh30pt.modeling.tokenizer import load_tokenizer
from eh30pt.scripts.common import log_prompt_version, train_and_evaluate
from eh30pt.tracking import build_run_name, open_run

if TYPE_CHECKING:
    from eh30pt.types import Architecture, DatasetName, FineTuneRunConfig


def main(
    model_path: str,
    dataset: DatasetName,
    n_shot: int,
    *,
    arch: Architecture | None = None,
    head_only: bool,
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
    seed: int | None = None,
) -> None:
    """Fine-tune `model_path` on `dataset` and evaluate on the test split."""
    logger.info("loading config for '%s'", model_path)
    config = AutoConfig.from_pretrained(model_path)
    arch = infer_architecture(config, arch)

    tokenizer = load_tokenizer(model_path)
    collate_fn = get_collator(tokenizer, arch)
    metrics_fn = MetricsAccumulator(tokenizer, arch)
    ds = load_data(
        tokenizer,
        dataset,
        arch,
        n_shot,
        n_train_samples=train_samples,
        n_val_samples=val_samples,
    )
    data = ds.data

    if dataset in {"boolq", "wic"}:
        logger.warning("using superglue validation data for test, labels are private")
        data["test"] = data["validation"]

    logger.info("loading '%s'", model_path)
    model = load_model(tokenizer, model_path, ds, arch, head_only=head_only)

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    method = "cls-head" if head_only else "fine-tune"
    if mlflow_run_name is None:
        mlflow_run_name = build_run_name(dataset, model_path, method, train_samples)

    logger.info("total parameters %d", total)
    logger.info("trainable parameters %d", trainable)
    logger.info("tracking '%s' of experiment '%s'", mlflow_run_name, mlflow_experiment)

    run_config: FineTuneRunConfig = {
        "method": method,
        "dataset": dataset,
        "architecture": arch,
        "base_model": model_path,
        "head_only": head_only,
        "n_shot": n_shot,
        "train_subset": train_samples,
        "val_subset": val_samples,
        "do_eval": do_eval,
        "patience": patience,
        "tolerance": tolerance,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "seed": seed,
        "system_prompt": ds.system_prompt,
        "train_samples": len(data["train"]),
        "validation_samples": len(data["validation"]) if do_eval else 0,
        "test_samples": len(data["test"]),
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
        log_prompt_version(ctx, dataset, arch, ds.system_prompt)
        train_and_evaluate(
            ctx,
            model,
            tokenizer=tokenizer,
            data=data,
            arch=arch,
            collate_fn=collate_fn,
            metrics_fn=metrics_fn,
            registered_name=f"{mlflow_experiment}-{dataset}-{arch}-{method}",
            do_eval=do_eval,
            patience=patience,
            tolerance=tolerance,
            epochs=epochs,
            learning_rate=learning_rate,
            batch_size=batch_size,
        )

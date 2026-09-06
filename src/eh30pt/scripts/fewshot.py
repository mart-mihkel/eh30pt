"""Run few-shot test evaluation for a pretrained model."""

from typing import TYPE_CHECKING, cast

import mlflow
from transformers import AutoConfig

from eh30pt.datasets.registry import load_data
from eh30pt.logging import logger
from eh30pt.modeling.arch import infer_architecture
from eh30pt.modeling.collator import get_collator
from eh30pt.modeling.loader import load_model
from eh30pt.modeling.metrics import MetricsAccumulator
from eh30pt.modeling.tokenizer import load_tokenizer
from eh30pt.modeling.trainer import get_trainer
from eh30pt.scripts.tracking import run_name

if TYPE_CHECKING:
    from datasets.splits import Split
    from torch.utils.data import Dataset

    from eh30pt.types import Architecture, DatasetName


def few_shot(
    model_path: str,
    dataset: DatasetName,
    n_shot: int,
    *,
    arch: Architecture | None = None,
    batch_size: int,
    mlflow_experiment: str,
    mlflow_run_name: str | None,
    mlflow_tracking_uri: str,
) -> None:
    """Run few-shot test evaluation of `model_path` on `dataset`."""
    logger.info("loading config for '%s'", model_path)
    config = AutoConfig.from_pretrained(model_path)
    arch = infer_architecture(config, arch)

    tokenizer = load_tokenizer(model_path)
    collate_fn = get_collator(tokenizer, arch)
    metrics_fn = MetricsAccumulator(tokenizer, arch)

    split = cast("Split", {"test": "test"})
    ds = load_data(tokenizer, dataset, arch, n_shot, split=split)
    data = ds.data

    if dataset in {"boolq", "wic"}:
        logger.warning("using superglue validation data, test labels are private")
        data["test"] = data["validation"]

    logger.info("loading '%s'", model_path)
    model = load_model(tokenizer, model_path, ds, arch, head_only=False)

    total = sum(p.numel() for p in model.parameters())

    method = f"{n_shot}-shot"
    if mlflow_run_name is None:
        mlflow_run_name = run_name(dataset, model_path, method, train_samples=0)

    logger.info("tracking '%s' of experiment '%s'", mlflow_run_name, mlflow_experiment)

    mlflow.set_tracking_uri(mlflow_tracking_uri)
    mlflow.set_experiment(mlflow_experiment)
    mlflow.start_run(run_name=mlflow_run_name)
    mlflow.log_param("n_shot", n_shot)
    mlflow.log_param("dataset", dataset)
    mlflow.log_param("architecture", arch)
    mlflow.log_param("base_model", model_path)
    mlflow.log_param("method", method)
    mlflow.log_param("system_prompt", ds.system_prompt)
    mlflow.log_metric("train_samples", 0)
    mlflow.log_metric("validation_samples", 0)
    mlflow.log_metric("test_samples", len(data["test"]))
    mlflow.log_metric("total_parameters", total)
    mlflow.log_metric("trainable_parameters", 0)

    trainer = get_trainer(
        model=model,
        data=data,
        arch=arch,
        collate_fn=collate_fn,
        metrics_fn=metrics_fn,
        do_eval=False,
        patience=0,
        batch_size=batch_size,
        run_name=mlflow_run_name,
        report_to="mlflow",
    )

    logger.debug("start test eval")
    test = cast("Dataset", data["test"])
    trainer.evaluate(test, metric_key_prefix="test")

    mlflow.end_run()

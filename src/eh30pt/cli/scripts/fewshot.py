"""Run few-shot test evaluation for a pretrained model."""

from typing import TYPE_CHECKING

from transformers import AutoConfig

from eh30pt.datasets.registry import (
    assert_classification,
    get_sys_prompt,
    load_data,
)
from eh30pt.logging import logger
from eh30pt.modeling.arch import infer_architecture
from eh30pt.modeling.collator import get_collator
from eh30pt.modeling.loader import load_model
from eh30pt.modeling.metrics import MetricsAccumulator
from eh30pt.modeling.tokenizer import cap_tokenizer_length, load_tokenizer
from eh30pt.tracking.context import RunContext
from eh30pt.tracking.resume import build_run_name

if TYPE_CHECKING:
    from eh30pt.settings import CommonSettings, MlflowSettings
    from eh30pt.types import FewShotRunConfig, Method


def main(
    common: CommonSettings,
    *,
    batch_size: int,
    constrain: bool,
    mlflow: MlflowSettings,
) -> None:
    """Run few-shot test evaluation of `model_path` on `dataset`."""
    assert_classification(common.dataset, "metrics need labels")

    logger.info("loading config for '%s'", common.model)
    config = AutoConfig.from_pretrained(common.model)
    arch = infer_architecture(config, common.arch)

    tokenizer = load_tokenizer(common.model)
    cap_tokenizer_length(tokenizer, config)

    collate_fn = get_collator(tokenizer, arch)
    metrics_fn = MetricsAccumulator(tokenizer, arch)

    splits = load_data(
        tokenizer,
        common.dataset,
        arch,
        n_shot=common.n_shot,
        seed=common.seed,
        data_path=common.data_path,
    )

    sys_prompt = get_sys_prompt(common.dataset, arch)
    model = load_model(tokenizer, common.model, common.dataset, arch, head_only=False)

    total = sum(p.numel() for p in model.parameters())

    method: Method = "few-shot"
    if mlflow.run_name is None:
        mlflow.run_name = build_run_name(
            common.model,
            "few-shot",
            f"{common.n_shot}-shot",
        )

    logger.info(
        "tracking '%s' of experiment '%s'",
        mlflow.run_name,
        mlflow.experiment,
    )

    config: FewShotRunConfig = {
        "method": method,
        "dataset": common.dataset,
        "architecture": arch,
        "base_model": common.model,
        "n_shot": common.n_shot,
        "batch_size": batch_size,
        "system_prompt": sys_prompt,
        "total_parameters": total,
        "trainable_parameters": 0,
        "constrain": constrain,
    }

    with RunContext(mlflow, config) as ctx:
        ctx.log_prompt_version(common.dataset, arch, sys_prompt)
        ctx.evaluate_and_log(
            model=model,
            tokenizer=tokenizer,
            data=splits,
            arch=arch,
            collate_fn=collate_fn,
            metrics_fn=metrics_fn,
            batch_size=batch_size,
            constrain=constrain,
        )

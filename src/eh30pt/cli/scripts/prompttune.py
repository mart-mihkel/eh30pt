"""Prompt-tune a pretrained model and run test evaluation."""

from typing import TYPE_CHECKING, cast

from transformers import AutoConfig

from eh30pt.datasets.registry import (
    assert_classification,
    get_sys_prompt,
    load_data,
)
from eh30pt.logging import logger
from eh30pt.modeling.arch import infer_architecture
from eh30pt.modeling.collator import get_collator
from eh30pt.modeling.metrics import MetricsAccumulator
from eh30pt.modeling.tokenizer import cap_tokenizer_length, load_tokenizer
from eh30pt.modeling.tuning import get_n_virtual, prepare_pt_model
from eh30pt.tracking.context import RunContext
from eh30pt.tracking.resume import build_run_name

if TYPE_CHECKING:
    from peft import PromptTuningConfig

    from eh30pt.settings import CommonSettings, MlflowSettings, TrainingSettings
    from eh30pt.types import (
        Method,
        PrefixInit,
        PromptTuneRunConfig,
    )


def main(
    common: CommonSettings,
    prefix_init: PrefixInit,
    *,
    training: TrainingSettings,
    mlflow: MlflowSettings,
) -> None:
    """Prompt-tune `model_path` on `dataset` and evaluate on the test split."""
    assert_classification(common.dataset, "training needs labels")

    logger.info("loading config for '%s'", common.model)
    config = AutoConfig.from_pretrained(common.model)
    arch = infer_architecture(config, common.arch)

    tokenizer = load_tokenizer(common.model)
    cap_tokenizer_length(tokenizer, config)

    collate_fn = get_collator(tokenizer, arch)
    metrics_fn = MetricsAccumulator(tokenizer, arch)

    sys_prompt = get_sys_prompt(common.dataset)
    n_virtual = get_n_virtual(tokenizer, sys_prompt)

    splits = load_data(
        tokenizer,
        common.dataset,
        arch,
        n_shot=common.n_shot,
        n_virtual=n_virtual,
        source=common.source,
        labeling_projects=common.labeling_projects,
    )

    model = prepare_pt_model(prefix_init, tokenizer, common.model, arch, common.dataset)

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    ptcfg = cast("PromptTuningConfig", model.peft_config["default"])

    method: Method = "prompt-tune"
    if mlflow.run_name is None:
        mlflow.run_name = build_run_name(common.model, method, prefix_init)

    logger.info("total parameters %d", total)
    logger.info("trainable parameters %d", trainable)
    logger.info("virtual tokens %d", ptcfg.num_virtual_tokens)
    logger.info(
        "tracking '%s' of experiment '%s'",
        mlflow.run_name,
        mlflow.experiment,
    )

    config: PromptTuneRunConfig = {
        "method": method,
        "dataset": common.dataset,
        "source": common.source,
        "architecture": arch,
        "base_model": common.model,
        "prefix_init": prefix_init,
        "n_virtual": ptcfg.num_virtual_tokens,
        "n_shot": common.n_shot,
        "eval_steps": training.eval_steps,
        "patience": training.patience,
        "tolerance": training.tolerance,
        "train_steps": training.train_steps,
        "batch_size": training.batch_size,
        "learning_rate": training.learning_rate,
        "warmup_ratio": training.warmup_ratio,
        "system_prompt": sys_prompt,
        "total_parameters": total,
        "trainable_parameters": trainable,
    }

    with RunContext(mlflow, config) as ctx:
        ctx.log_prompt_version(common.dataset, arch, sys_prompt)
        ctx.train_and_evaluate(
            model=model,
            tokenizer=tokenizer,
            data=splits,
            arch=arch,
            collate_fn=collate_fn,
            metrics_fn=metrics_fn,
            registered_name=f"{mlflow.experiment}-{common.dataset}-{arch}-{method}-{prefix_init}",
            training=training,
        )

"""Command line interface."""

import asyncio
from typing import TYPE_CHECKING

import rich_click as click

from eh30pt.cli.options.common import (
    ArchOption,
    DatasetOption,
    InferredDatasetOption,
    LangExtractNShotOption,
    LogLevelOption,
    ModelOption,
    NShotOption,
    OutputOption,
    SeedOption,
)
from eh30pt.cli.options.labeling import (
    LabelStudioUrlOption,
    OptionalProjectIdsOption,
    ProjectIdsOption,
    ProjectTitleOption,
)
from eh30pt.cli.options.lora import LoraAlphaOption, LoraDropoutOption, LoraRankOption
from eh30pt.cli.options.mlflow import (
    ExperimentOption,
    ExportRunOption,
    RunIdOption,
    RunNameOption,
    TrackingUriOption,
)
from eh30pt.cli.options.server import (
    CompletionDatasetOption,
    ConcurrencyOption,
    FromIdOption,
    LogIntervalOption,
    PromptArgument,
    ProviderOption,
    ShardSizeOption,
    SourceOption,
    SystemOption,
)
from eh30pt.cli.options.slurm import (
    DryRunOption,
    InnerCommandArgument,
    SlurmCpusOption,
    SlurmGresOption,
    SlurmJobNameOption,
    SlurmMemOption,
    SlurmOutputOption,
    SlurmPartitionOption,
    SlurmTimeOption,
)
from eh30pt.cli.options.training import (
    BatchSizeOption,
    EvalStepsOption,
    HeadOnlyOption,
    PatienceOption,
    PrefixInitOption,
    ToleranceOption,
    TrainStepsOption,
    WarmupRatioOption,
    learning_rate_option,
)
from eh30pt.logging import logger, setup_logging
from eh30pt.settings import (
    CommonSettings,
    LoRASettings,
    MlflowSettings,
    ServerSettings,
    StreamSettings,
    TrainingSettings,
)

if TYPE_CHECKING:
    from pathlib import Path

    from eh30pt.types import (
        Architecture,
        DatasetName,
        DataSource,
        LogLevel,
        PrefixInit,
        Provider,
    )


@click.group(
    "ehpt",
    context_settings={"help_option_names": ["-h", "--help"]},
    help="Interface for running scripts",
    no_args_is_help=True,
)
@click.rich_config(
    click.RichHelpConfiguration(
        theme="cargo-slim",
        command_groups={
            "ehpt": [
                {
                    "name": "Hugging Face",
                    "commands": ["fine-tune", "prompt-tune", "qlora", "few-shot"],
                },
                {
                    "name": "Llama",
                    "commands": ["completion", "evaluate", "generate", "langextract"],
                },
                {
                    "name": "Label-Studio",
                    "commands": ["create-task", "export-labels"],
                },
                {"name": "Meta", "commands": ["slurm"]},
            ]
        },
        option_groups={
            "ehpt *": [
                {"name": "Model", "options": ["--model", "--arch"]},
                {
                    "name": "Data",
                    "options": [
                        "--dataset",
                        "--source",
                        "--n-shot",
                        "--project",
                        "--from-id",
                    ],
                },
                {
                    "name": "Training",
                    "options": [
                        "--train-steps",
                        "--eval-steps",
                        "--batch-size",
                        "--learning-rate",
                        "--warmup-ratio",
                        "--patience",
                        "--tolerance",
                        "--head-only",
                        "--prefix-init",
                        "--lora-rank",
                        "--lora-alpha",
                        "--lora-dropout",
                    ],
                },
                {
                    "name": "Server",
                    "options": [
                        "--provider",
                        "--system",
                        "--concurrency",
                        "--shard-size",
                        "--log-interval",
                    ],
                },
                {
                    "name": "SLURM",
                    "options": [
                        "--job-name",
                        "--partition",
                        "--gres",
                        "--cpus",
                        "--mem",
                        "--time",
                        "--dry-run",
                    ],
                },
                {"name": "Output", "options": ["--output"]},
                {
                    "name": "MLflow",
                    "options": [
                        "--mlflow-tracking-uri",
                        "--mlflow-experiment",
                        "--mlflow-run-name",
                        "--mlflow-run-id",
                    ],
                },
                {
                    "name": "Label Studio",
                    "options": ["--label-studio-url", "--title"],
                },
                {"name": "Run", "options": ["--log-level", "--seed"]},
            ]
        },
    )
)
def app() -> None:
    """Set up logging before running a command."""
    setup_logging()


def set_seed(seed: int | None) -> None:
    """Fix randomness."""
    import random

    import numpy as np
    import torch

    if seed is None:
        logger.warning("randomness is not fixed")
        return

    random.seed(seed)
    np.random.seed(seed)  # noqa: NPY002
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if torch.mps.is_available():
        torch.mps.manual_seed(seed)


def log_accelerator() -> None:
    """Log torch acceleator."""
    import torch

    if torch.cuda.is_available():
        logger.info("cuda available")
    elif torch.mps.is_available():
        logger.info("metal available")
    else:
        logger.warning("no acceleator available")


@app.command("fine-tune", help="Fine-tune and run test evaluation")
@ModelOption
@DatasetOption
@HeadOnlyOption
@NShotOption
@TrainStepsOption
@EvalStepsOption
@BatchSizeOption
@learning_rate_option(5e-5)
@WarmupRatioOption
@PatienceOption
@ToleranceOption
@ArchOption
@SourceOption
@OptionalProjectIdsOption
@ExperimentOption
@RunNameOption
@RunIdOption
@TrackingUriOption
@LogLevelOption
@SeedOption
def fine_tune(
    *,
    model: str,
    dataset: DatasetName,
    head_only: bool,
    n_shot: int,
    train_steps: int,
    eval_steps: int,
    batch_size: int,
    learning_rate: float,
    warmup_ratio: float,
    patience: int,
    tolerance: float,
    arch: Architecture | None,
    source: DataSource | None,
    labeling_projects: tuple[int, ...] | None,
    mlflow_experiment: str,
    mlflow_run_name: str | None,
    mlflow_run_id: str | None,
    mlflow_tracking_uri: str,
    log_level: LogLevel,
    seed: int | None,
) -> None:
    """Fine-tune and run test evaluation."""
    from eh30pt.cli.scripts.finetune import main

    logger.setLevel(log_level.upper())
    set_seed(seed)
    log_accelerator()

    common = CommonSettings(
        model=model,
        dataset=dataset,
        arch=arch,
        n_shot=n_shot,
        source=source,
        labeling_projects=list(labeling_projects) if labeling_projects else None,
    )

    mlflow = MlflowSettings(
        tracking_uri=mlflow_tracking_uri,
        experiment=mlflow_experiment,
        run_name=mlflow_run_name,
        run_id=mlflow_run_id,
    )

    training = TrainingSettings(
        eval_steps=eval_steps,
        patience=patience,
        tolerance=tolerance,
        train_steps=train_steps,
        batch_size=batch_size,
        learning_rate=learning_rate,
        warmup_ratio=warmup_ratio,
    )

    main(
        common=common,
        head_only=head_only,
        training=training,
        mlflow=mlflow,
    )


@app.command("prompt-tune", help="Prompt-tune and run test evaluation")
@ModelOption
@DatasetOption
@PrefixInitOption
@NShotOption
@TrainStepsOption
@EvalStepsOption
@BatchSizeOption
@learning_rate_option(1e-3)
@WarmupRatioOption
@PatienceOption
@ToleranceOption
@ArchOption
@SourceOption
@OptionalProjectIdsOption
@ExperimentOption
@RunNameOption
@RunIdOption
@TrackingUriOption
@LogLevelOption
@SeedOption
def prompt_tune(
    *,
    model: str,
    dataset: DatasetName,
    prefix_init: PrefixInit,
    n_shot: int,
    train_steps: int,
    eval_steps: int,
    batch_size: int,
    learning_rate: float,
    warmup_ratio: float,
    patience: int,
    tolerance: float,
    arch: Architecture | None,
    source: DataSource | None,
    labeling_projects: tuple[int, ...] | None,
    mlflow_experiment: str,
    mlflow_run_name: str | None,
    mlflow_run_id: str | None,
    mlflow_tracking_uri: str,
    log_level: LogLevel,
    seed: int | None,
) -> None:
    """Prompt-tune and run test evaluation."""
    from eh30pt.cli.scripts.prompttune import main

    logger.setLevel(log_level.upper())
    set_seed(seed)
    log_accelerator()

    common = CommonSettings(
        model=model,
        dataset=dataset,
        arch=arch,
        n_shot=n_shot,
        source=source,
        labeling_projects=list(labeling_projects) if labeling_projects else None,
    )

    mlflow = MlflowSettings(
        tracking_uri=mlflow_tracking_uri,
        experiment=mlflow_experiment,
        run_name=mlflow_run_name,
        run_id=mlflow_run_id,
    )

    training = TrainingSettings(
        eval_steps=eval_steps,
        patience=patience,
        tolerance=tolerance,
        train_steps=train_steps,
        batch_size=batch_size,
        learning_rate=learning_rate,
        warmup_ratio=warmup_ratio,
    )

    main(
        common=common,
        prefix_init=prefix_init,
        training=training,
        mlflow=mlflow,
    )


@app.command("qlora", help="QLoRA-tune and run test evaluation")
@ModelOption
@DatasetOption
@LoraRankOption
@LoraAlphaOption
@LoraDropoutOption
@NShotOption
@TrainStepsOption
@EvalStepsOption
@BatchSizeOption
@learning_rate_option(2e-4)
@WarmupRatioOption
@PatienceOption
@ToleranceOption
@ArchOption
@SourceOption
@OptionalProjectIdsOption
@ExperimentOption
@RunNameOption
@RunIdOption
@TrackingUriOption
@LogLevelOption
@SeedOption
def qlora(
    *,
    model: str,
    dataset: DatasetName,
    lora_rank: int,
    lora_alpha: int,
    lora_dropout: float,
    n_shot: int,
    train_steps: int,
    eval_steps: int,
    batch_size: int,
    learning_rate: float,
    warmup_ratio: float,
    patience: int,
    tolerance: float,
    arch: Architecture | None,
    source: DataSource | None,
    labeling_projects: tuple[int, ...] | None,
    mlflow_experiment: str,
    mlflow_run_name: str | None,
    mlflow_run_id: str | None,
    mlflow_tracking_uri: str,
    log_level: LogLevel,
    seed: int | None,
) -> None:
    """QLoRA-tune and run test evaluation."""
    from eh30pt.cli.scripts.qlora import main

    logger.setLevel(log_level.upper())
    set_seed(seed)
    log_accelerator()

    common = CommonSettings(
        model=model,
        dataset=dataset,
        arch=arch,
        n_shot=n_shot,
        source=source,
        labeling_projects=list(labeling_projects) if labeling_projects else None,
    )

    lora = LoRASettings(
        rank=lora_rank,
        alpha=lora_alpha,
        dropout=lora_dropout,
    )

    mlflow = MlflowSettings(
        tracking_uri=mlflow_tracking_uri,
        experiment=mlflow_experiment,
        run_name=mlflow_run_name,
        run_id=mlflow_run_id,
    )

    training = TrainingSettings(
        eval_steps=eval_steps,
        patience=patience,
        tolerance=tolerance,
        train_steps=train_steps,
        batch_size=batch_size,
        learning_rate=learning_rate,
        warmup_ratio=warmup_ratio,
    )

    main(
        common=common,
        lora=lora,
        training=training,
        mlflow=mlflow,
    )


@app.command("few-shot", help="Run test evaluation with few-shot learning")
@ModelOption
@DatasetOption
@NShotOption
@BatchSizeOption
@ArchOption
@SourceOption
@ExperimentOption
@RunNameOption
@RunIdOption
@TrackingUriOption
@LogLevelOption
@SeedOption
def few_shot(
    *,
    model: str,
    dataset: DatasetName,
    n_shot: int,
    batch_size: int,
    arch: Architecture | None,
    source: DataSource | None,
    mlflow_experiment: str,
    mlflow_run_name: str | None,
    mlflow_run_id: str | None,
    mlflow_tracking_uri: str,
    log_level: LogLevel,
    seed: int | None,
) -> None:
    """Run test evaluation with few-shot learning."""
    from eh30pt.cli.scripts.fewshot import main

    logger.setLevel(log_level.upper())
    set_seed(seed)
    log_accelerator()

    common = CommonSettings(
        model=model,
        dataset=dataset,
        arch=arch,
        n_shot=n_shot,
        source=source,
    )

    mlflow = MlflowSettings(
        tracking_uri=mlflow_tracking_uri,
        experiment=mlflow_experiment,
        run_name=mlflow_run_name,
        run_id=mlflow_run_id,
    )

    main(common=common, batch_size=batch_size, mlflow=mlflow)


@app.command(
    "slurm",
    context_settings={"ignore_unknown_options": True, "allow_extra_args": True},
    help="Submit a training command as a SLURM job",
)
@click.pass_context
@InnerCommandArgument
@SlurmJobNameOption
@SlurmOutputOption
@SlurmPartitionOption
@SlurmGresOption
@SlurmCpusOption
@SlurmMemOption
@SlurmTimeOption
@DryRunOption
@LogLevelOption
def slurm(
    ctx: click.Context,
    command: str,
    *,
    job_name: str | None,
    output: str,
    partition: str | None,
    gres: str | None,
    cpus: int,
    mem: str,
    time: str,
    dry_run: bool,
    log_level: LogLevel,
) -> None:
    """
    Submit a training command as a SLURM job.

    Append the training command line after `--`.
    """
    from eh30pt.cli.scripts.slurm import main

    logger.setLevel(log_level.upper())

    if command not in ("fine-tune", "prompt-tune", "qlora", "few-shot"):
        msg = f"cannot submit '{command}', expected a training command"
        raise click.BadParameter(msg)

    main(
        command,
        list(ctx.args),
        job_name=job_name or command,
        output=output,
        partition=partition,
        gres=gres,
        cpus=cpus,
        mem=mem,
        time=time,
        dry_run=dry_run,
    )


@app.command("completion", help="Query a server and print the completion")
@PromptArgument
@SystemOption
@ProviderOption
@CompletionDatasetOption
@LogLevelOption
@SeedOption
def completion(
    *,
    user_prompt: str,
    system_prompt: str | None,
    provider: Provider,
    dataset: DatasetName | None,
    log_level: LogLevel,
    seed: int | None,
) -> None:
    """Query a server and print the completion."""
    from eh30pt.cli.scripts.completion import main
    from eh30pt.datasets.registry import get_response_schema, get_sys_prompt

    logger.setLevel(log_level.upper())
    if seed is None:
        logger.warning("randomness is not fixed")

    if system_prompt is None and dataset is not None:
        logger.debug("using '%s' system prompt", dataset)
        system_prompt = get_sys_prompt(dataset)

    if system_prompt is None:
        msg = "system prompt is required when --dataset is not given"
        raise click.BadParameter(msg, param_hint="--system")

    fmt = get_response_schema(dataset) if dataset is not None else None
    server = ServerSettings(provider=provider, format=fmt)

    asyncio.run(
        main(
            user_prompt=user_prompt,
            system_prompt=system_prompt,
            server=server,
            seed=seed,
        )
    )


@app.command("evaluate", help="Run few-shot test evaluation on a server")
@DatasetOption
@SourceOption
@ProviderOption
@NShotOption
@SystemOption
@ConcurrencyOption
@FromIdOption
@RunNameOption
@RunIdOption
@ExperimentOption
@TrackingUriOption
@LogLevelOption
@LogIntervalOption
@SeedOption
def evaluate(
    *,
    dataset: DatasetName,
    source: DataSource | None,
    provider: Provider,
    n_shot: int,
    system_prompt: str | None,
    concurrency: int,
    from_id: int | None,
    mlflow_run_name: str | None,
    mlflow_run_id: str | None,
    mlflow_experiment: str,
    mlflow_tracking_uri: str,
    log_level: LogLevel,
    log_interval: int,
    seed: int | None,
) -> None:
    """Run few-shot test evaluation on a server."""
    from eh30pt.cli.scripts.evaluate import main
    from eh30pt.datasets.registry import get_response_schema

    logger.setLevel(log_level.upper())
    if seed is None:
        logger.warning("randomness is not fixed")

    fmt = get_response_schema(dataset)
    server = ServerSettings(provider=provider, format=fmt)

    mlflow = MlflowSettings(
        tracking_uri=mlflow_tracking_uri,
        experiment=mlflow_experiment,
        run_name=mlflow_run_name,
        run_id=mlflow_run_id,
    )

    stream = StreamSettings(
        log_interval=log_interval,
        concurrency=concurrency,
        from_id=from_id,
    )

    asyncio.run(
        main(
            system_prompt=system_prompt,
            server=server,
            dataset=dataset,
            source=source,
            n_shot=n_shot,
            stream=stream,
            mlflow=mlflow,
            seed=seed,
        ),
    )


@app.command("generate", help="Generate predictions on a server")
@DatasetOption
@SourceOption
@ProviderOption
@NShotOption
@SystemOption
@ConcurrencyOption
@FromIdOption
@RunNameOption
@RunIdOption
@ExperimentOption
@TrackingUriOption
@LogLevelOption
@ShardSizeOption
@SeedOption
def generate(
    *,
    dataset: DatasetName,
    source: DataSource | None,
    provider: Provider,
    n_shot: int,
    system_prompt: str | None,
    concurrency: int,
    from_id: int | None,
    mlflow_run_name: str | None,
    mlflow_run_id: str | None,
    mlflow_experiment: str,
    mlflow_tracking_uri: str,
    log_level: LogLevel,
    shard_size: int,
    seed: int | None,
) -> None:
    """Generate predictions on a server and log them as an MLflow artifact."""
    from eh30pt.cli.scripts.generate import main
    from eh30pt.datasets.registry import get_response_schema

    logger.setLevel(log_level.upper())
    if seed is None:
        logger.warning("randomness is not fixed")

    fmt = get_response_schema(dataset)
    server = ServerSettings(provider=provider, format=fmt)

    mlflow = MlflowSettings(
        tracking_uri=mlflow_tracking_uri,
        experiment=mlflow_experiment,
        run_name=mlflow_run_name,
        run_id=mlflow_run_id,
    )

    stream = StreamSettings(
        shard_size=shard_size,
        concurrency=concurrency,
        from_id=from_id,
    )

    asyncio.run(
        main(
            system_prompt=system_prompt,
            server=server,
            dataset=dataset,
            source=source,
            n_shot=n_shot,
            stream=stream,
            mlflow=mlflow,
            seed=seed,
        ),
    )


@app.command("langextract", help="Generate predictions with langextract on a server")
@DatasetOption
@SourceOption
@ProviderOption
@LangExtractNShotOption
@SystemOption
@FromIdOption
@RunNameOption
@RunIdOption
@ExperimentOption
@TrackingUriOption
@LogLevelOption
@ShardSizeOption
@ConcurrencyOption
@SeedOption
def langextract(
    *,
    dataset: DatasetName,
    source: DataSource | None,
    provider: Provider,
    n_shot: int,
    system_prompt: str | None,
    from_id: int | None,
    mlflow_run_name: str | None,
    mlflow_run_id: str | None,
    mlflow_experiment: str,
    mlflow_tracking_uri: str,
    log_level: LogLevel,
    shard_size: int,
    concurrency: int,
    seed: int | None,
) -> None:
    """Generate predictions with langextract and log them as an artifact."""
    from eh30pt.cli.scripts.langextract import main

    logger.setLevel(log_level.upper())
    if seed is None:
        logger.warning("randomness is not fixed")

    server = ServerSettings(provider=provider)
    mlflow = MlflowSettings(
        tracking_uri=mlflow_tracking_uri,
        experiment=mlflow_experiment,
        run_name=mlflow_run_name,
        run_id=mlflow_run_id,
    )

    stream = StreamSettings(
        shard_size=shard_size,
        concurrency=concurrency,
        from_id=from_id,
    )

    asyncio.run(
        main(
            system_prompt=system_prompt,
            server=server,
            dataset=dataset,
            source=source,
            n_shot=n_shot,
            stream=stream,
            mlflow=mlflow,
            seed=seed,
        ),
    )


@app.command("create-task", help="Create a Label Studio project from prediction shards")
@ExportRunOption
@InferredDatasetOption
@ProjectTitleOption
@LabelStudioUrlOption
@TrackingUriOption
@LogLevelOption
def create_task(
    *,
    mlflow_run_id: str,
    dataset: DatasetName | None,
    title: str | None,
    label_studio_url: str,
    mlflow_tracking_uri: str,
    log_level: LogLevel,
) -> None:
    """Create a Label Studio project and import the run's tasks into it."""
    from eh30pt.cli.scripts.labeling import main
    from eh30pt.settings import LabelStudioSettings

    logger.setLevel(log_level.upper())

    studio = LabelStudioSettings(url=label_studio_url)
    mlflow = MlflowSettings(tracking_uri=mlflow_tracking_uri, run_id=mlflow_run_id)

    project = main(
        settings=studio,
        mlflow_settings=mlflow,
        dataset=dataset,
        title=title,
    )

    logger.info(
        "label studio project %d is ready at '%s'", project.id, label_studio_url
    )


@app.command(
    "export-labels", help="Export finished annotations from Label Studio projects"
)
@ProjectIdsOption
@OutputOption
@LabelStudioUrlOption
@LogLevelOption
def export_labels(
    *,
    labeling_projects: tuple[int, ...],
    output: Path | None,
    label_studio_url: str,
    log_level: LogLevel,
) -> None:
    """Read finished annotations into one Parquet frame."""
    from eh30pt.cli.scripts.labeling import export_labels
    from eh30pt.settings import LabelStudioSettings

    logger.setLevel(log_level.upper())

    studio = LabelStudioSettings(url=label_studio_url)
    export_labels(settings=studio, projects=list(labeling_projects), output=output)


if __name__ == "__main__":
    app()

"""Command line interface."""

import asyncio

from typer import BadParameter, Context, Typer

from eh30pt.cli.options import (
    ArchOption,
    BatchSizeOption,
    CompletionDatasetOption,
    ConcurrencyOption,
    DatasetOption,
    DryRunOption,
    EvalStepsOption,
    ExperimentOption,
    ExportRunOption,
    FromIdOption,
    HeadOnlyOption,
    InnerCommandArgument,
    LangExtractNShotOption,
    LearningRateOption,
    LogIntervalOption,
    LogLevelOption,
    LoraAlphaOption,
    LoraDropoutOption,
    LoraRankOption,
    ModelOption,
    NShotOption,
    OutputOption,
    PatienceOption,
    PrefixInitOption,
    PromptArgument,
    ProviderOption,
    RunIdOption,
    RunNameOption,
    SeedOption,
    ShardSizeOption,
    SlurmCpusOption,
    SlurmGresOption,
    SlurmJobNameOption,
    SlurmMemOption,
    SlurmOutputOption,
    SlurmPartitionOption,
    SlurmTimeOption,
    SourceOption,
    SystemOption,
    ToleranceOption,
    TrackingUriOption,
    TrainStepsOption,
    WarmupRatioOption,
)
from eh30pt.constants import LOGDIR
from eh30pt.logging import logger, setup_logging
from eh30pt.settings import (
    CommonSettings,
    LoRASettings,
    MlflowSettings,
    ServerSettings,
    StreamSettings,
    TrainingSettings,
)

app = Typer(
    context_settings={"help_option_names": ["-h", "--help"]},
    help="Interface for running scripts",
    no_args_is_help=True,
)


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


@app.callback()
def setup() -> None:
    """Set up logging before running a command."""
    setup_logging()


@app.command(help="Fine-tune and run test evaluation")
def fine_tune(
    *,
    model: ModelOption,
    dataset: DatasetOption,
    head_only: HeadOnlyOption = False,
    n_shot: NShotOption = 0,
    train_steps: TrainStepsOption,
    eval_steps: EvalStepsOption = 0,
    batch_size: BatchSizeOption = 8,
    learning_rate: LearningRateOption = 5e-5,
    warmup_ratio: WarmupRatioOption = 0.06,
    patience: PatienceOption = 0,
    tolerance: ToleranceOption = 0.01,
    arch: ArchOption = None,
    mlflow_experiment: ExperimentOption = "eh30pt",
    mlflow_run_name: RunNameOption = None,
    mlflow_run_id: RunIdOption = None,
    mlflow_tracking_uri: TrackingUriOption = "sqlite:///mlflow.db",
    log_level: LogLevelOption = "info",
    seed: SeedOption = None,
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

    main(common=common, head_only=head_only, training=training, mlflow=mlflow)


@app.command(help="Prompt-tune and run test evaluation")
def prompt_tune(
    *,
    model: ModelOption,
    dataset: DatasetOption,
    prefix_init: PrefixInitOption = "pretrained",
    n_shot: NShotOption = 0,
    train_steps: TrainStepsOption,
    eval_steps: EvalStepsOption = 0,
    batch_size: BatchSizeOption = 8,
    learning_rate: LearningRateOption = 1e-3,
    warmup_ratio: WarmupRatioOption = 0.06,
    patience: PatienceOption = 0,
    tolerance: ToleranceOption = 0.01,
    arch: ArchOption = None,
    mlflow_experiment: ExperimentOption = "eh30pt",
    mlflow_run_name: RunNameOption = None,
    mlflow_run_id: RunIdOption = None,
    mlflow_tracking_uri: TrackingUriOption = "sqlite:///mlflow.db",
    log_level: LogLevelOption = "info",
    seed: SeedOption = None,
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

    main(common=common, prefix_init=prefix_init, training=training, mlflow=mlflow)


@app.command(help="QLoRA-tune and run test evaluation")
def qlora(
    *,
    model: ModelOption,
    dataset: DatasetOption,
    lora_rank: LoraRankOption = 16,
    lora_alpha: LoraAlphaOption = 16,
    lora_dropout: LoraDropoutOption = 0.05,
    n_shot: NShotOption = 0,
    train_steps: TrainStepsOption,
    eval_steps: EvalStepsOption = 0,
    batch_size: BatchSizeOption = 8,
    learning_rate: LearningRateOption = 2e-4,
    warmup_ratio: WarmupRatioOption = 0.06,
    patience: PatienceOption = 0,
    tolerance: ToleranceOption = 0.01,
    arch: ArchOption = None,
    mlflow_experiment: ExperimentOption = "eh30pt",
    mlflow_run_name: RunNameOption = None,
    mlflow_run_id: RunIdOption = None,
    mlflow_tracking_uri: TrackingUriOption = "sqlite:///mlflow.db",
    log_level: LogLevelOption = "info",
    seed: SeedOption = None,
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

    main(common=common, lora=lora, training=training, mlflow=mlflow)


@app.command(help="Run test evaluation with few-shot learning")
def few_shot(
    *,
    model: ModelOption,
    dataset: DatasetOption,
    n_shot: NShotOption = 0,
    batch_size: BatchSizeOption = 8,
    arch: ArchOption = None,
    mlflow_experiment: ExperimentOption = "eh30pt",
    mlflow_run_name: RunNameOption = None,
    mlflow_run_id: RunIdOption = None,
    mlflow_tracking_uri: TrackingUriOption = "sqlite:///mlflow.db",
    log_level: LogLevelOption = "info",
    seed: SeedOption = None,
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
    )

    mlflow = MlflowSettings(
        tracking_uri=mlflow_tracking_uri,
        experiment=mlflow_experiment,
        run_name=mlflow_run_name,
        run_id=mlflow_run_id,
    )

    main(common=common, batch_size=batch_size, mlflow=mlflow)


@app.command(
    context_settings={"ignore_unknown_options": True, "allow_extra_args": True},
    help="Submit a training command as a SLURM job",
)
def slurm(
    ctx: Context,
    command: InnerCommandArgument,
    *,
    job_name: SlurmJobNameOption = None,
    output: SlurmOutputOption = "log/slurm/%j-%x.out",
    partition: SlurmPartitionOption = None,
    gres: SlurmGresOption = None,
    cpus: SlurmCpusOption = 8,
    mem: SlurmMemOption = "64G",
    time: SlurmTimeOption = "12:00:00",
    dry_run: DryRunOption = False,
    log_level: LogLevelOption = "info",
) -> None:
    """
    Submit a training command as a SLURM job.

    Append the training command line after `--`.
    """
    from eh30pt.cli.scripts.slurm import main

    logger.setLevel(log_level.upper())

    if command not in ("fine-tune", "prompt-tune", "qlora", "few-shot"):
        msg = f"cannot submit '{command}', expected a training command"
        raise BadParameter(msg)

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


@app.command(help="Query a server and print the completion")
def completion(
    *,
    user_prompt: PromptArgument,
    system_prompt: SystemOption = None,
    provider: ProviderOption = "llama",
    dataset: CompletionDatasetOption = None,
    log_level: LogLevelOption = "info",
    seed: SeedOption = None,
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
        raise BadParameter(msg, param_hint="--system")

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


@app.command(help="Run few-shot test evaluation on a server")
def evaluate(
    *,
    dataset: DatasetOption,
    source: SourceOption = "trigger",
    provider: ProviderOption = "llama",
    n_shot: NShotOption = 0,
    system_prompt: SystemOption = None,
    concurrency: ConcurrencyOption = 8,
    from_id: FromIdOption = None,
    mlflow_run_name: RunNameOption = None,
    mlflow_run_id: RunIdOption = None,
    mlflow_experiment: ExperimentOption = "eh30pt",
    mlflow_tracking_uri: TrackingUriOption = "sqlite:///mlflow.db",
    log_level: LogLevelOption = "info",
    log_interval: LogIntervalOption = 1000,
    seed: SeedOption = None,
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


@app.command(help="Generate predictions on a server")
def generate(
    *,
    dataset: DatasetOption,
    source: SourceOption = "trigger",
    provider: ProviderOption = "llama",
    n_shot: NShotOption = 0,
    system_prompt: SystemOption = None,
    concurrency: ConcurrencyOption = 8,
    from_id: FromIdOption = None,
    mlflow_run_name: RunNameOption = None,
    mlflow_run_id: RunIdOption = None,
    mlflow_experiment: ExperimentOption = "eh30pt",
    mlflow_tracking_uri: TrackingUriOption = "sqlite:///mlflow.db",
    log_level: LogLevelOption = "info",
    shard_size: ShardSizeOption = 10_000,
    seed: SeedOption = None,
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


@app.command(help="Generate predictions with langextract on a server")
def langextract(
    *,
    dataset: DatasetOption,
    source: SourceOption = "trigger",
    provider: ProviderOption = "llama",
    n_shot: LangExtractNShotOption = 1,
    system_prompt: SystemOption = None,
    from_id: FromIdOption = None,
    mlflow_run_name: RunNameOption = None,
    mlflow_run_id: RunIdOption = None,
    mlflow_experiment: ExperimentOption = "eh30pt",
    mlflow_tracking_uri: TrackingUriOption = "sqlite:///mlflow.db",
    log_level: LogLevelOption = "info",
    shard_size: ShardSizeOption = 10_000,
    concurrency: ConcurrencyOption = 8,
    seed: SeedOption = None,
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


@app.command(help="Export prediction shards as Label Studio tasks")
def export(
    *,
    mlflow_run_id: ExportRunOption,
    mlflow_tracking_uri: TrackingUriOption = "sqlite:///mlflow.db",
    output: OutputOption = None,
    log_level: LogLevelOption = "info",
) -> None:
    """Export prediction shards as Label Studio classification tasks."""
    from eh30pt.cli.scripts.export import main

    logger.setLevel(log_level.upper())

    if output is None:
        output = LOGDIR / f"task-{mlflow_run_id}.json"
        logger.debug("inferred output dir '%s'", output)

    mlflow = MlflowSettings(tracking_uri=mlflow_tracking_uri, run_id=mlflow_run_id)

    main(settings=mlflow, output=output)


if __name__ == "__main__":
    app()

"""Live model server querying and streaming options."""

import rich_click as click

# Unannotated on purpose: click decorators are generic over the decorated
# callable, which a variable annotation cannot express, so the precise type
# is left for inference.

prompt_rgument = click.argument("user_prompt", type=str)

system_prompt_option = click.option(
    "--system",
    "system_prompt",
    type=str,
    default=None,
    help="System prompt, defaults to the dataset prompt",
)

completion_dataset_option = click.option(
    "--dataset",
    "-d",
    "dataset",
    type=click.Choice(
        [
            "multinerd",
            "oblique",
            "discont-classifier",
            "discont-decider",
            "discont-extract",
            "discont-multiple",
        ]
    ),
    default=None,
    help="Constrain output to the dataset label set",
)

shard_size_option = click.option(
    "--shard-size",
    "shard_size",
    type=click.IntRange(min=1),
    default=10_000,
    show_default=True,
    help="Records per prediction shard",
)

log_interval_option = click.option(
    "--log-interval",
    "log_interval",
    type=click.IntRange(min=1),
    default=1000,
    show_default=True,
    help="Samples between running metric logs",
)

concurrency_option = click.option(
    "--concurrency",
    "-c",
    "concurrency",
    type=click.IntRange(min=1),
    default=8,
    show_default=True,
    help="Concurrent server requests",
)

provider_option = click.option(
    "--provider",
    "provider",
    type=click.Choice(["llama", "vllm"]),
    default="llama",
    show_default=True,
    help="Server provider",
)

data_source_option = click.option(
    "--source",
    "source",
    type=click.Choice(
        ["observed", "trigger", "discont", "switch", "usage-gap", "extract-labels"]
    ),
    default=None,
    help="Est-Health 30 query for discontinuation events",
)

from_id_option = click.option(
    "--from-id",
    "from_id",
    type=click.IntRange(min=0),
    default=None,
    help="Only stream rows with id greater than this value",
)

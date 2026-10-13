"""Live model server querying and streaming options."""

from typing import get_args

import rich_click as click

from eh30pt.types import DatasetName

prompt_argument = click.argument("user_prompt", type=str)

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
    type=click.Choice(get_args(DatasetName.__value__)),
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

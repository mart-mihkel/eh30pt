"""Live model server querying and streaming options."""

import rich_click as click

port_option = click.option(
    "--port",
    type=click.IntRange(min=1, max=65535),
    default=None,
    help="Local port: 9931 for llama, 9932 for System One, 8000 for vLLM by default",
)

systemone_option = click.option(
    "--systemone",
    is_flag=True,
    help="Use llama.cpp's System One decision endpoint for classification",
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
    default=2,
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

"""Live model server querying and streaming options."""

from typing import Annotated

from typer import Argument, Option

from eh30pt.types import DatasetName, DataSource, Provider

PromptArgument = Annotated[
    str,
    Argument(help="User prompt to send to the server"),
]

SystemOption = Annotated[
    str | None,
    Option("--system", help="System prompt, defaults to the dataset prompt"),
]

CompletionDatasetOption = Annotated[
    DatasetName.__value__ | None,
    Option("--dataset", "-d", help="Constrain output to the dataset label set"),
]

BaseUrlOption = Annotated[
    str,
    Option("--base-url", help="Base URL of the server"),
]

ShardSizeOption = Annotated[
    int,
    Option("--shard-size", min=1, help="Records per prediction shard"),
]

LogIntervalOption = Annotated[
    int,
    Option("--log-interval", min=1, help="Samples between running metric logs"),
]

ConcurrencyOption = Annotated[
    int,
    Option("--concurrency", "-c", min=1, help="Concurrent server requests"),
]

ProviderOption = Annotated[
    Provider.__value__,
    Option("--provider", help="Server provider"),
]

SourceOption = Annotated[
    DataSource.__value__ | None,
    Option("--source", help="Est-Health 30 query for discontinuation events"),
]

FromIdOption = Annotated[
    int | None,
    Option(
        "--from-id",
        min=0,
        help="Only stream rows with id greater than this value",
    ),
]

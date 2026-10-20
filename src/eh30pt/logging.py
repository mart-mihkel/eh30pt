"""Rich logging setup."""

import logging
import sys
import warnings
from io import StringIO
from typing import TYPE_CHECKING, TextIO, cast

import rich.traceback
import tqdm.std
from rich import box
from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table
from tqdm import TqdmExperimentalWarning

if TYPE_CHECKING:
    from collections import Counter
    from collections.abc import Sequence

logger = logging.getLogger("eh30pt")
"""Project logger."""


def format_counts(counts: Counter[int] | Counter[str]) -> str:
    """Format label counts as a compact `key=value` string for debug logs."""
    return " ".join(f"'{key}'={value}" for key, value in sorted(counts.items()))


def format_confusion_matrix(
    labels: Sequence[str],
    counts: Sequence[Sequence[int]],
) -> str:
    """Render a gold-by-predicted confusion matrix as a monospace table."""
    table = Table(box=box.SQUARE, header_style="bold")
    table.add_column("gold\\pred", justify="right")
    for label in labels:
        table.add_column(label, justify="right")

    for label, row in zip(labels, counts, strict=True):
        table.add_row(label, *(str(value) for value in row))

    buffer = StringIO()
    Console(file=buffer, width=100).print(table)
    return buffer.getvalue().rstrip()


class _TqdmSafeStream:
    """
    Buffer writes and flush per line through tqdm writes.

    Prevents the transformers trainer from mangling logs.
    """

    def __init__(self, stream: TextIO) -> None:
        self._stream = stream
        self._buffer = ""

    def write(self, text: str) -> int:
        self._buffer += text
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            tqdm.std.tqdm.write(line, file=self._stream)

        return len(text)

    def flush(self) -> None:
        self._stream.flush()

    def isatty(self) -> bool:
        return self._stream.isatty()


def rich_handler(stream: TextIO) -> RichHandler:
    """Build a rich handler writing to a stream."""
    return RichHandler(console=Console(file=stream), rich_tracebacks=True)


def _forward_filters(removed: logging.Handler, root: logging.Logger) -> None:
    """Move a stripped handler's filters onto the root rich handlers."""
    targets = [hdlr for hdlr in root.handlers if isinstance(hdlr, RichHandler)]
    for filt in removed.filters:
        for target in targets:
            if not any(type(existing) is type(filt) for existing in target.filters):
                target.addFilter(filt)


def route_library_loggers() -> None:
    """Route third-party library logs through the root rich handler."""
    try:
        from transformers.utils import logging as hf_logging

        hf_logging.disable_default_handler()
    except ImportError:
        pass

    loggers = (
        "transformers",
        "datasets",
        "huggingface_hub",
        "accelerate",
        "tokenizers",
        "trl",
        "peft",
        "mlflow",
    )

    root = logging.getLogger()
    for name in loggers:
        lib_logger = logging.getLogger(name)
        for handler in list(lib_logger.handlers):
            if not isinstance(handler, RichHandler):
                lib_logger.removeHandler(handler)
                _forward_filters(handler, root)

        lib_logger.propagate = True


def setup_logging() -> None:
    """Install the rich log handler, ascii tqdm bars, and rich tracebacks."""
    warnings.filterwarnings(
        "ignore",
        message="'pin_memory' argument is set as true but not supported on MPS.*",
        category=UserWarning,
    )

    warnings.filterwarnings(
        "ignore",
        message="rich is experimental/alpha",
        category=TqdmExperimentalWarning,
    )

    stream = cast("TextIO", _TqdmSafeStream(sys.stderr))
    root = logging.getLogger()
    has = any(isinstance(hdlr, RichHandler) for hdlr in root.handlers)
    if not has:
        root.addHandler(rich_handler(stream))

    route_library_loggers()

    rich.traceback.install(suppress=[rich, tqdm])
    logger.setLevel(logging.INFO)

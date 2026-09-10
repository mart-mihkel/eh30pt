"""Rich-backed logging setup."""

import logging
import sys
import warnings
from functools import partialmethod
from typing import TYPE_CHECKING, TextIO, cast

import rich.traceback
import tqdm.std
from rich.console import Console
from rich.logging import RichHandler

if TYPE_CHECKING:
    from collections import Counter

logger = logging.getLogger("eh30pt")
"""Project logger."""


def format_counts(counts: Counter[int] | Counter[str]) -> str:
    """Format label counts as a compact `key=value` string for debug logs."""
    return " ".join(f"'{key}'={value}" for key, value in sorted(counts.items()))


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


def setup_logging() -> None:
    """Install the rich log handler, ascii tqdm bars, and rich tracebacks."""
    warnings.filterwarnings(
        "ignore",
        message="'pin_memory' argument is set as true but not supported on MPS.*",
        category=UserWarning,
    )

    console = Console(
        file=cast("TextIO", _TqdmSafeStream(sys.stderr)),
        width=80,
    )

    handler = RichHandler(console=console, show_path=False, show_time=False)

    rich.traceback.install(suppress=[rich, tqdm])
    logging.basicConfig(format="%(message)s", handlers=[handler])
    tqdm.std.tqdm.__init__ = partialmethod(tqdm.std.tqdm.__init__, ascii=True)

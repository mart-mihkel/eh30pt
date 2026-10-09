"""Uv-style logging setup."""

import logging
import os
import sys
import warnings
from functools import partialmethod
from typing import TYPE_CHECKING, TextIO, cast

import rich.traceback
import tqdm.std

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


class _UvFormatter(logging.Formatter):
    """Prefix log lines with a lowercase level like uv does."""

    def __init__(self, *, use_color: bool = True) -> None:
        super().__init__()
        self._use_color = use_color

    def format(self, record: logging.LogRecord) -> str:
        message = record.getMessage()
        if record.levelno >= logging.CRITICAL:
            name, start = "critical", "\x1b[31m"
        elif record.levelno >= logging.ERROR:
            name, start = "error", "\x1b[31m"
        elif record.levelno >= logging.WARNING:
            name, start = "warning", "\x1b[33m"
        elif record.levelno >= logging.INFO:
            name, start = "info", "\x1b[32m"
        else:
            name, start = "debug", "\x1b[34m"

        prefix = f"{start}{name}\x1b[39m " if self._use_color else f"{name} "
        text = f"{prefix}{message}"
        if record.exc_info:
            if not record.exc_text:
                record.exc_text = self.formatException(record.exc_info)

            text = f"{text}\n{record.exc_text}"

        if record.stack_info:
            text = f"{text}\n{self.formatStack(record.stack_info)}"

        return text


def _supports_color(stream: TextIO) -> bool:
    """Report whether the stream wants ansi colors like anstream does."""
    if os.environ.get("NO_COLOR"):
        return False

    if os.environ.get("CLICOLOR_FORCE", "") not in ("", "0"):
        return True

    if os.environ.get("CLICOLOR", "") == "0":
        return False

    if os.environ.get("TERM", "") == "dumb":
        return False

    return stream.isatty()


def setup_logging() -> None:
    """Install the uv-style log handler, ascii tqdm bars, and rich tracebacks."""
    warnings.filterwarnings(
        "ignore",
        message="'pin_memory' argument is set as true but not supported on MPS.*",
        category=UserWarning,
    )

    stream = cast("TextIO", _TqdmSafeStream(sys.stderr))
    handler: logging.StreamHandler[TextIO] = logging.StreamHandler(stream)
    handler.setFormatter(_UvFormatter(use_color=_supports_color(sys.stderr)))

    rich.traceback.install(suppress=[rich, tqdm])
    root = logging.getLogger()
    has = any(isinstance(hdlr.formatter, _UvFormatter) for hdlr in root.handlers)
    if not has:
        root.addHandler(handler)

    logger.setLevel(logging.INFO)
    if not isinstance(tqdm.std.tqdm.__init__, partialmethod):
        tqdm.std.tqdm.__init__ = partialmethod(tqdm.std.tqdm.__init__, ascii=True)

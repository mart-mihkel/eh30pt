"""Tests for logging helpers."""

import io
import logging
from collections import Counter

from rich.logging import RichHandler

from eh30pt.logging import (
    _TqdmSafeStream,
    format_counts,
    logger,
    setup_logging,
)


def _record(level: int, message: str) -> logging.LogRecord:
    return logging.LogRecord(
        name="eh30pt",
        level=level,
        pathname=__file__,
        lineno=1,
        msg=message,
        args=(),
        exc_info=None,
    )


def test_format_counts_sorts_keys() -> None:
    assert format_counts(Counter({"b": 2, "a": 1})) == "'a'=1 'b'=2"


def test_tqdm_stream_buffers_partial_lines() -> None:
    stream = io.StringIO()
    safe = _TqdmSafeStream(stream)

    assert safe.write("partial") == len("partial")
    assert stream.getvalue() == ""

    assert safe.write(" line\n") == len(" line\n")
    assert "partial line" in stream.getvalue()


def test_tqdm_stream_flush_and_isatty() -> None:
    stream = io.StringIO()
    safe = _TqdmSafeStream(stream)

    safe.write("hello\n")
    safe.flush()

    assert "hello" in stream.getvalue()
    assert safe.isatty() == stream.isatty()


def _raise_boom() -> None:
    message = "boom"
    raise ValueError(message)


def test_setup_logging_preserves_root_handlers() -> None:
    root = logging.getLogger()
    handlers = list(root.handlers)
    level = logger.level
    marker = logging.StreamHandler(io.StringIO())
    try:
        root.addHandler(marker)
        setup_logging()
        setup_logging()
        assert marker in root.handlers
        rich_handlers = [
            handler for handler in root.handlers if isinstance(handler, RichHandler)
        ]
        assert len(rich_handlers) == 1
    finally:
        for handler in root.handlers:
            if handler not in handlers:
                root.removeHandler(handler)
        logger.setLevel(level)

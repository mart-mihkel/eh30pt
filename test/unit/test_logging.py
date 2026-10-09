"""Tests for logging helpers."""

import io
import logging
import sys
from collections import Counter

from eh30pt.logging import (
    _TqdmSafeStream,
    _UvFormatter,
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


def test_uv_formatter_plain_levels() -> None:
    formatter = _UvFormatter(use_color=False)

    assert formatter.format(_record(logging.DEBUG, "hello")) == "debug hello"
    assert formatter.format(_record(logging.INFO, "hello")) == "info hello"
    assert formatter.format(_record(logging.WARNING, "hello")) == "warning hello"
    assert formatter.format(_record(logging.ERROR, "hello")) == "error hello"
    assert formatter.format(_record(logging.CRITICAL, "hello")) == "critical hello"


def test_uv_formatter_colored_levels() -> None:
    formatter = _UvFormatter(use_color=True)

    assert (
        formatter.format(_record(logging.DEBUG, "hello"))
        == "\x1b[34mdebug\x1b[39m hello"
    )
    assert (
        formatter.format(_record(logging.INFO, "hello")) == "\x1b[32minfo\x1b[39m hello"
    )
    assert (
        formatter.format(_record(logging.WARNING, "hello"))
        == "\x1b[33mwarning\x1b[39m hello"
    )
    assert (
        formatter.format(_record(logging.ERROR, "hello"))
        == "\x1b[31merror\x1b[39m hello"
    )


def _raise_boom() -> None:
    message = "boom"
    raise ValueError(message)


def test_uv_formatter_appends_traceback() -> None:
    formatter = _UvFormatter(use_color=False)
    try:
        _raise_boom()
    except ValueError:
        record = logging.LogRecord(
            name="eh30pt",
            level=logging.ERROR,
            pathname=__file__,
            lineno=1,
            msg="failed",
            args=(),
            exc_info=sys.exc_info(),
        )

    text = formatter.format(record)
    assert text.startswith("error failed")
    assert "ValueError: boom" in text


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
        uv_handlers = [
            handler
            for handler in root.handlers
            if isinstance(handler.formatter, _UvFormatter)
        ]
        assert len(uv_handlers) == 1
    finally:
        for handler in root.handlers:
            if handler not in handlers:
                root.removeHandler(handler)
        logger.setLevel(level)

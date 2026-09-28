"""Tests for rich logging helpers."""

import io
from collections import Counter

from eh30pt.logging import _TqdmSafeStream, format_counts


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

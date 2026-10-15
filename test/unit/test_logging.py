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


def test_setup_logging_routes_huggingface_loggers_to_root() -> None:
    root = logging.getLogger()
    handlers = list(root.handlers)
    try:
        setup_logging()
        setup_logging()
        assert any(isinstance(handler, RichHandler) for handler in root.handlers)
        for name in ("transformers", "datasets", "huggingface_hub"):
            lib_logger = logging.getLogger(name)
            assert lib_logger.propagate
            assert all(
                isinstance(handler, RichHandler) for handler in lib_logger.handlers
            )
    finally:
        for handler in root.handlers:
            if handler not in handlers:
                root.removeHandler(handler)


def test_setup_logging_keeps_transformers_default_handler_disabled() -> None:
    from transformers.utils import logging as hf_logging

    root = logging.getLogger()
    handlers = list(root.handlers)
    try:
        setup_logging()
        hf_logging.get_logger("transformers.trainer")
        lib_logger = logging.getLogger("transformers")
        assert lib_logger.propagate
        assert all(isinstance(handler, RichHandler) for handler in lib_logger.handlers)
    finally:
        for handler in root.handlers:
            if handler not in handlers:
                root.removeHandler(handler)


def test_setup_logging_routes_mlflow_logger_to_root() -> None:
    import mlflow.utils.logging_utils as mlflow_logging

    root = logging.getLogger()
    handlers = list(root.handlers)
    mlflow_logger = logging.getLogger("mlflow")
    mlflow_handlers = list(mlflow_logger.handlers)
    mlflow_propagate = mlflow_logger.propagate
    try:
        plain = logging.StreamHandler()
        plain.addFilter(mlflow_logging.SuppressLogFilter())
        mlflow_logger.addHandler(plain)
        mlflow_logger.propagate = False
        setup_logging()
        assert mlflow_logger.propagate
        assert all(
            isinstance(handler, RichHandler) for handler in mlflow_logger.handlers
        )
        rich_handlers = [
            handler for handler in root.handlers if isinstance(handler, RichHandler)
        ]
        assert rich_handlers
        assert any(
            any(
                isinstance(filt, mlflow_logging.SuppressLogFilter)
                for filt in handler.filters
            )
            for handler in rich_handlers
        )
    finally:
        for handler in root.handlers:
            if handler not in handlers:
                root.removeHandler(handler)
        for handler in list(mlflow_logger.handlers):
            if handler not in mlflow_handlers:
                mlflow_logger.removeHandler(handler)
        mlflow_logger.propagate = mlflow_propagate

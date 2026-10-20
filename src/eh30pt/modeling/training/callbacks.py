"""Rich logging and CUDA metrics for trainer callbacks."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import mlflow
import torch
from rich.pretty import pretty_repr
from transformers import PrinterCallback, ProgressCallback, TrainerCallback

from eh30pt.logging import logger

if TYPE_CHECKING:
    from collections.abc import Callable

    from transformers import Trainer, TrainerControl, TrainerState, TrainingArguments


class RichProgressCallback(ProgressCallback):
    """Progress bars with metrics logged through rich instead of plain text."""

    def on_log(
        self,
        args: TrainingArguments,  # noqa: ARG002
        state: TrainerState,
        control: TrainerControl,  # noqa: ARG002
        logs: dict[str, float | int | str] | None = None,
        **kwargs: Any,  # noqa: ANN401, ARG002
    ) -> None:
        """Log metrics through the project logger, keeping the progress bars."""
        if not state.is_world_process_zero or logs is None:
            return

        formatted: dict[str, float | int | str] = {}
        for key, value in logs.items():
            if key == "total_flos":
                continue

            if isinstance(value, str) and len(value) > self.max_str_len:
                formatted[key] = (
                    f"[String too long to display, length: {len(value)} > "
                    f"{self.max_str_len}. Consider increasing `max_str_len` if needed.]"
                )
            elif isinstance(value, float):
                formatted[key] = f"{value:.4g}"
            else:
                formatted[key] = value

        logger.info("%s", pretty_repr(formatted, expand_all=True))


class CudaMetricsCallback(TrainerCallback):
    """Log CUDA utilization and memory metrics to the active MLflow run."""

    def __init__(self) -> None:
        """Start a callback that has not logged a step yet."""
        self._last_step: int | None = None

    def on_log(
        self,
        args: TrainingArguments,  # noqa: ARG002
        state: TrainerState,
        control: TrainerControl,  # noqa: ARG002
        **kwargs: Any,  # noqa: ANN401, ARG002
    ) -> None:
        """Log CUDA metrics whenever the trainer logs other metrics."""
        self._log(state)

    def on_train_end(
        self,
        args: TrainingArguments,  # noqa: ARG002
        state: TrainerState,
        control: TrainerControl,  # noqa: ARG002
        **kwargs: Any,  # noqa: ANN401, ARG002
    ) -> None:
        """Log final CUDA metrics after training finishes."""
        self._log(state)

    def _log(self, state: TrainerState) -> None:
        """Log current CUDA metrics once per global step."""
        if not state.is_world_process_zero:
            return
        if self._last_step == state.global_step:
            return
        if mlflow.active_run() is None:
            return

        metrics = _cuda_metrics()
        if not metrics:
            return

        mlflow.log_metrics(metrics, step=state.global_step)
        self._last_step = state.global_step


def _cuda_metrics() -> dict[str, float]:
    """Collect CUDA memory and device metrics for every visible GPU."""
    if not torch.cuda.is_available():
        return {}

    metrics = {}
    for index in range(torch.cuda.device_count()):
        pref = f"cuda/{index}/"

        metrics[f"{pref}memory_allocated_bytes"] = torch.cuda.memory_allocated(index)
        metrics[f"{pref}memory_reserved_bytes"] = torch.cuda.memory_reserved(index)
        metrics[f"{pref}max_memory_allocated_bytes"] = torch.cuda.max_memory_allocated(
            index
        )
        metrics[f"{pref}max_memory_reserved_bytes"] = torch.cuda.max_memory_reserved(
            index
        )

        readers: tuple[tuple[str, Callable[[], float]], ...] = (
            (
                "device_memory_used_bytes",
                lambda index=index: torch.cuda.device_memory_used(index),
            ),
            ("utilization_percent", lambda index=index: torch.cuda.utilization(index)),
            ("power_draw_milliwatts", lambda index=index: torch.cuda.power_draw(index)),
            ("temperature_celsius", lambda index=index: torch.cuda.temperature(index)),
            ("clock_rate_mhz", lambda index=index: torch.cuda.clock_rate(index)),
        )

        for name, _read in readers:
            try:
                metrics[f"{pref}{name}"] = _read()
            except (ModuleNotFoundError, RuntimeError) as error:
                logger.debug("could not read CUDA metric %s: %s", name, error)

    return metrics


def use_rich_logging(trainer: Trainer) -> None:
    """Replace plain metric printing with rich logging, keeping progress bars."""
    for callback in list(trainer.callback_handler.callbacks):
        if type(callback) in (ProgressCallback, PrinterCallback):
            trainer.callback_handler.remove_callback(callback)

    has = any(
        isinstance(callback, RichProgressCallback)
        for callback in trainer.callback_handler.callbacks
    )

    if not has:
        trainer.add_callback(RichProgressCallback())

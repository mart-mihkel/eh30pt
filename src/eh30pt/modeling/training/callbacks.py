"""Rich logging for trainer metrics."""

from typing import TYPE_CHECKING, Any

from rich.pretty import pretty_repr
from transformers import PrinterCallback, ProgressCallback

from eh30pt.logging import logger

if TYPE_CHECKING:
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

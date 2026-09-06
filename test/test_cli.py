"""Tests for CLI helper functions and command wiring."""

import random
import sys
import types
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
from typer.testing import CliRunner

from eh30pt.cli import _set_seed, app

runner = CliRunner()

_FINE_TUNE_LR = 5e-5
_PROMPT_TUNE_LR = 1e-3


def _stub_script(monkeypatch: pytest.MonkeyPatch, module: str, func: str) -> MagicMock:
    """Expose a mock `func` as `eh30pt.scripts.<module>` for lazy CLI imports."""
    fake = MagicMock()
    stub = types.ModuleType(f"eh30pt.scripts.{module}")
    setattr(stub, func, fake)
    monkeypatch.setitem(sys.modules, f"eh30pt.scripts.{module}", stub)
    return fake


def test_set_seed_is_deterministic() -> None:
    _set_seed(0)
    a = (random.random(), np.random.rand(), torch.rand(1).item())  # noqa: NPY002, S311

    _set_seed(0)
    b = (random.random(), np.random.rand(), torch.rand(1).item())  # noqa: NPY002, S311

    assert a == b


def test_set_seed_different_seeds_differ() -> None:
    _set_seed(0)
    a = torch.rand(1).item()

    _set_seed(1)
    b = torch.rand(1).item()

    assert a != b


@pytest.mark.skipif(not torch.mps.is_available(), reason="needs an MPS device")
def test_set_seed_is_deterministic_on_mps() -> None:
    _set_seed(0)
    a = torch.randn(4, device="mps")

    _set_seed(0)
    b = torch.randn(4, device="mps")

    assert torch.equal(a, b)


def test_fine_tune_command_forwards_args(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _stub_script(monkeypatch, "finetune", "main")
    epochs = 2

    result = runner.invoke(
        app,
        [
            "fine-tune",
            "--model",
            "my-model",
            "--dataset",
            "multinerd",
            "--epochs",
            str(epochs),
            "--head-only",
        ],
    )

    assert result.exit_code == 0, result.output
    fake.assert_called_once()
    kwargs = fake.call_args.kwargs
    assert kwargs["model_path"] == "my-model"
    assert kwargs["dataset"] == "multinerd"
    assert kwargs["epochs"] == epochs
    assert kwargs["head_only"] is True
    assert kwargs["learning_rate"] == _FINE_TUNE_LR


def test_fine_tune_command_resumes_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _stub_script(monkeypatch, "finetune", "main")

    result = runner.invoke(
        app,
        ["fine-tune", "--model", "my-model", "--dataset", "multinerd"],
    )

    assert result.exit_code == 0, result.output
    assert fake.call_args.kwargs["resume"] is True


def test_fine_tune_command_forwards_no_resume(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _stub_script(monkeypatch, "finetune", "main")

    result = runner.invoke(
        app,
        ["fine-tune", "--model", "my-model", "--dataset", "multinerd", "--no-resume"],
    )

    assert result.exit_code == 0, result.output
    assert fake.call_args.kwargs["resume"] is False


def test_prompt_tune_command_forwards_no_resume(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _stub_script(monkeypatch, "ptune", "main")

    result = runner.invoke(
        app,
        [
            "prompt-tune",
            "--model",
            "my-model",
            "--dataset",
            "multinerd",
            "--prefix-init",
            "random",
            "--no-resume",
        ],
    )

    assert result.exit_code == 0, result.output
    assert fake.call_args.kwargs["resume"] is False


def test_fine_tune_command_sets_seed_when_provided(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_script(monkeypatch, "finetune", "main")
    fake_set_seed = MagicMock()
    monkeypatch.setattr("eh30pt.cli._set_seed", fake_set_seed)

    result = runner.invoke(
        app,
        ["fine-tune", "--model", "my-model", "--dataset", "multinerd", "--seed", "7"],
    )

    assert result.exit_code == 0, result.output
    fake_set_seed.assert_called_once_with(7)


def test_fine_tune_command_forwards_seed(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _stub_script(monkeypatch, "finetune", "main")
    seed = 7

    result = runner.invoke(
        app,
        [
            "fine-tune",
            "--model",
            "my-model",
            "--dataset",
            "multinerd",
            "--seed",
            str(seed),
        ],
    )

    assert result.exit_code == 0, result.output
    assert fake.call_args.kwargs["seed"] == seed


def test_prompt_tune_command_forwards_args(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _stub_script(monkeypatch, "ptune", "main")

    result = runner.invoke(
        app,
        [
            "prompt-tune",
            "--model",
            "my-model",
            "--dataset",
            "multinerd",
            "--prefix-init",
            "random",
        ],
    )

    assert result.exit_code == 0, result.output
    fake.assert_called_once()
    kwargs = fake.call_args.kwargs
    assert kwargs["model_path"] == "my-model"
    assert kwargs["dataset"] == "multinerd"
    assert kwargs["prefix_init"] == "random"
    assert kwargs["learning_rate"] == _PROMPT_TUNE_LR


def test_few_shot_command_forwards_args(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _stub_script(monkeypatch, "fewshot", "main")
    n_shot = 3

    result = runner.invoke(
        app,
        [
            "few-shot",
            "--model",
            "my-model",
            "--dataset",
            "oblique",
            "--n-shot",
            str(n_shot),
        ],
    )

    assert result.exit_code == 0, result.output
    fake.assert_called_once()
    kwargs = fake.call_args.kwargs
    assert kwargs["model_path"] == "my-model"
    assert kwargs["dataset"] == "oblique"
    assert kwargs["n_shot"] == n_shot


def test_collect_metrics_command_forwards_args(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    fake = MagicMock()
    monkeypatch.setattr("eh30pt.tracking.collect_runs", fake)

    result = runner.invoke(
        app,
        ["collect", "--mlflow-experiment", "my-exp"],
    )

    assert result.exit_code == 0, result.output
    fake.assert_called_once_with(
        "my-exp",
        "sqlite:///mlflow.db",
        (),
        write_csv=True,
    )


def test_collect_command_forwards_metrics(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = MagicMock()
    monkeypatch.setattr("eh30pt.tracking.collect_runs", fake)

    result = runner.invoke(
        app,
        ["collect", "--metric", "eval_f1", "--metric", "grad_norm"],
    )

    assert result.exit_code == 0, result.output
    assert fake.call_args.args[2] == ("eval_f1", "grad_norm")


def test_fine_tune_rejects_unknown_dataset() -> None:
    result = runner.invoke(
        app,
        ["fine-tune", "--model", "my-model", "--dataset", "bogus"],
    )

    assert result.exit_code != 0


def test_fine_tune_rejects_negative_epochs() -> None:
    result = runner.invoke(
        app,
        [
            "fine-tune",
            "--model",
            "my-model",
            "--dataset",
            "multinerd",
            "--epochs",
            "-1",
        ],
    )

    assert result.exit_code != 0


def test_prompt_tune_requires_prefix_init() -> None:
    result = runner.invoke(
        app,
        ["prompt-tune", "--model", "my-model", "--dataset", "multinerd"],
    )

    assert result.exit_code != 0


def test_fine_tune_forwards_patience(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _stub_script(monkeypatch, "finetune", "main")
    patience = 3
    tolerance = 0.02

    result = runner.invoke(
        app,
        [
            "fine-tune",
            "--model",
            "my-model",
            "--dataset",
            "multinerd",
            "--do-eval",
            "--patience",
            str(patience),
            "--tolerance",
            str(tolerance),
        ],
    )

    assert result.exit_code == 0, result.output
    assert fake.call_args.kwargs["patience"] == patience
    assert fake.call_args.kwargs["tolerance"] == tolerance


def test_fine_tune_disables_early_stopping_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _stub_script(monkeypatch, "finetune", "main")

    result = runner.invoke(
        app,
        ["fine-tune", "--model", "my-model", "--dataset", "multinerd"],
    )

    assert result.exit_code == 0, result.output
    assert fake.call_args.kwargs["patience"] == 0


def test_fine_tune_rejects_negative_patience() -> None:
    result = runner.invoke(
        app,
        [
            "fine-tune",
            "--model",
            "my-model",
            "--dataset",
            "multinerd",
            "--patience",
            "-1",
        ],
    )

    assert result.exit_code != 0


def test_fine_tune_requires_model() -> None:
    result = runner.invoke(app, ["fine-tune", "--dataset", "multinerd"])

    assert result.exit_code != 0

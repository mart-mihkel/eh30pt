"""Tests for SLURM batch script rendering."""

from __future__ import annotations

from typing import TYPE_CHECKING, Final, Protocol

import pytest

from eh30pt.cli.scripts.slurm import build_sbatch, main

if TYPE_CHECKING:
    from pathlib import Path

_job: Final = "ft-oblique"
_project: Final = "/cluster/eh30pt"
_output: Final = "log/slurm/ft-oblique-%j.out"


class RenderScript(Protocol):
    """Render the standard fine-tune batch script."""

    def __call__(
        self,
        partition: str | None = ...,
        gres: str | None = ...,
        output: str = ...,
    ) -> str:
        """Render the script, optionally without partition and gres."""
        ...


@pytest.fixture
def make_render() -> RenderScript:
    """Return a factory rendering the standard fine-tune batch script."""

    def _factory(
        partition: str | None = "gpu",
        gres: str | None = "gpu:1",
        output: str = _output,
    ) -> str:
        return build_sbatch(
            "fine-tune",
            ["--model", "some-model", "--dataset", "oblique"],
            job_name=_job,
            partition=partition,
            gres=gres,
            cpus=8,
            mem="64G",
            time="12:00:00",
            project_dir=_project,
            output=output,
        )

    return _factory


def test_build_sbatch_renders_directives_and_command(
    make_render: RenderScript,
) -> None:
    script = make_render()

    assert script.startswith("#!/usr/bin/env bash\n")
    assert script.endswith("\n")
    assert "#SBATCH --job-name=ft-oblique" in script
    assert "#SBATCH --partition=gpu" in script
    assert "#SBATCH --gres=gpu:1" in script
    assert "#SBATCH --cpus-per-task=8" in script
    assert "#SBATCH --mem=64G" in script
    assert "#SBATCH --time=12:00:00" in script
    assert f"#SBATCH --output={_output}" in script
    assert "set -euo pipefail" in script
    assert f"cd {_project}" in script
    assert (
        "uv run --no-sync eh30pt fine-tune --model some-model --dataset oblique"
        in script
    )


def test_build_sbatch_omits_partition_and_gres_when_unset(
    make_render: RenderScript,
) -> None:
    script = make_render(partition=None, gres=None)

    assert "#SBATCH --partition" not in script
    assert "#SBATCH --gres" not in script


def test_build_sbatch_renders_custom_output(
    make_render: RenderScript,
) -> None:
    script = make_render(output="log/slurm/%j-%x.out")

    assert "#SBATCH --output=log/slurm/%j-%x.out" in script


def test_build_sbatch_renders_example_header() -> None:
    script = build_sbatch(
        "fine-tune",
        ["--model", "some-model", "--dataset", "multinerd"],
        job_name="deberta",
        partition="gpu",
        gres="gpu:h200-141g:1",
        cpus=32,
        mem="16GB",
        time="05:00:00",
        project_dir=_project,
        output="log/slurm/%j-%x.out",
    )

    assert "#SBATCH --output=log/slurm/%j-%x.out" in script
    assert "#SBATCH --gres=gpu:h200-141g:1" in script
    assert "#SBATCH --cpus-per-task=32" in script
    assert "#SBATCH --job-name=deberta" in script
    assert "#SBATCH --time=05:00:00" in script
    assert "#SBATCH --partition=gpu" in script
    assert "#SBATCH --mem=16GB" in script


def test_main_dry_run_writes_script_and_prints(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.chdir(tmp_path)
    main(
        "fine-tune",
        ["--model", "some-model"],
        job_name=_job,
        output=_output,
        partition="gpu",
        gres="gpu:1",
        cpus=8,
        mem="64G",
        time="12:00:00",
        dry_run=True,
    )

    path = tmp_path / "log" / "slurm" / f"{_job}.sbatch"
    assert path.is_file()

    out = capsys.readouterr().out
    assert out == path.read_text()
    assert "uv run --no-sync eh30pt fine-tune --model some-model" in out


def test_main_without_sbatch_raises(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PATH", str(tmp_path))

    with pytest.raises(RuntimeError, match="sbatch executable not found"):
        main(
            "fine-tune",
            ["--model", "some-model"],
            job_name=_job,
            output=_output,
            partition="gpu",
            gres="gpu:1",
            cpus=8,
            mem="64G",
            time="12:00:00",
            dry_run=False,
        )


def test_main_submits_and_logs_job_id(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    import os
    import stat

    monkeypatch.chdir(tmp_path)
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    sbatch = bin_dir / "sbatch"
    sbatch.write_text('#!/usr/bin/env bash\necho "Submitted batch job 123"\n')
    sbatch.chmod(sbatch.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}")

    with caplog.at_level("INFO", logger="eh30pt"):
        main(
            "fine-tune",
            ["--model", "some-model"],
            job_name=_job,
            output=_output,
            partition="gpu",
            gres="gpu:1",
            cpus=8,
            mem="64G",
            time="12:00:00",
            dry_run=False,
        )

    assert "submitted slurm job 123" in caplog.text


def test_main_submit_without_job_id_logs_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    import os
    import stat

    monkeypatch.chdir(tmp_path)
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    sbatch = bin_dir / "sbatch"
    sbatch.write_text('#!/usr/bin/env bash\necho "ok"\n')
    sbatch.chmod(sbatch.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}")

    with caplog.at_level("INFO", logger="eh30pt"):
        main(
            "fine-tune",
            ["--model", "some-model"],
            job_name=_job,
            output=_output,
            partition="gpu",
            gres="gpu:1",
            cpus=8,
            mem="64G",
            time="12:00:00",
            dry_run=False,
        )

    assert "sbatch:" in caplog.text

"""Export finished annotations from Label Studio projects."""

from typing import TYPE_CHECKING

import polars as pl
from returns.result import Failure, Success

from eh30pt.constants import LOGDIR
from eh30pt.labeling.registry import export_frame
from eh30pt.labeling.studio import StudioClient
from eh30pt.logging import logger

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from eh30pt.settings import LabelStudioSettings
    from eh30pt.types import DatasetName


def main(
    *,
    settings: LabelStudioSettings,
    projects: Sequence[int],
    dataset: DatasetName,
    output: Path | None = None,
) -> Path:
    """Write projects' annotations to Parquet."""
    dfs: list[pl.DataFrame] = []
    with StudioClient(settings) as studio:
        for project in projects:
            match studio.tasks(project):
                case Success(tasks):
                    pass
                case Failure(exc):
                    msg = f"could not read tasks of project {project}: {exc}"
                    raise RuntimeError(msg) from exc

            _df = export_frame(tasks, dataset, project)
            logger.info(
                "collected %d annotations from project %d",
                _df.height,
                project,
            )

            dfs.append(_df)

    df = pl.concat(dfs, how="diagonal")
    if output is None:
        joined = "-".join(str(id_) for id_ in projects)
        ids = f"labels-{joined}.parquet"
        output = LOGDIR / ids
        logger.debug("inferred output path '%s'", output)

    output.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(output)
    logger.info("wrote %d annotation rows to '%s'", df.height, output)

    return output

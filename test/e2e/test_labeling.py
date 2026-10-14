"""End to end tests for creating a Label Studio project over the API."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import polars as pl
import pytest
from returns.result import Failure

from eh30pt.datasets.discontinuation.extract import DiscontExtract
from eh30pt.datasets.registry import get_labels
from eh30pt.labeling.loaders.extract import ExtractionFrameSchema
from eh30pt.labeling.studio import StudioClient, StudioResult
from eh30pt.labeling.tasks.extract import extraction_controls, extraction_labeling
from eh30pt.settings import LabelStudioSettings

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

    from eh30pt.labeling.studio import ProjectRef


@pytest.fixture
def preds() -> pl.DataFrame:
    """Build prediction rows from the extraction dataset's own examples."""
    shots = DiscontExtract.extract_shots
    return pl.DataFrame(
        {
            "user": [shot.user for shot in shots],
            "prediction": [
                json.dumps(shot.extractions, ensure_ascii=False, sort_keys=True)
                for shot in shots
            ],
        }
    )


@pytest.fixture
def studio() -> LabelStudioSettings:
    """Return settings for a live server, xfailing when none is reachable."""
    settings = LabelStudioSettings()
    with StudioClient(settings) as client:
        if isinstance(client.projects(), Failure):
            pytest.xfail(f"label studio at '{settings.url}' is unreachable")

    return settings


@pytest.fixture
def project(studio: LabelStudioSettings, preds: pl.DataFrame) -> Iterator[ProjectRef]:
    """Create a throwaway project and delete it after the test."""
    labeling = extraction_labeling("discont-extract/trigger", preds, "run-1")

    with StudioClient(studio) as client:
        created = client.create("eh30pt-e2e", labeling.config).unwrap()
        yield created

        client.delete(created.id).unwrap()


def test_create_project_imports_tasks_with_pre_annotations(
    studio: LabelStudioSettings,
    project: ProjectRef,
    preds: pl.DataFrame,
) -> None:
    labeling = extraction_labeling("discont-extract/trigger", preds, "run-1")

    with StudioClient(studio) as client:
        client.validate(project.id, labeling.config).unwrap()
        imported = client.import_tasks(project.id, labeling.tasks).unwrap()

    assert imported.tasks == len(labeling.tasks)
    assert imported.predictions == len(labeling.tasks)

    with StudioClient(studio) as client:
        tasks = client.tasks(project.id).unwrap()

    assert len(tasks) == len(labeling.tasks)
    assert tasks[0]["data"] == labeling.tasks[0]["data"]
    assert (
        tasks[0]["data"]["drug"] == DiscontExtract.extract_shots[0].extractions["drug"]
    )


def test_create_task_creates_a_separate_project_per_call(
    studio: LabelStudioSettings,
    preds: pl.DataFrame,
) -> None:
    labeling = extraction_labeling("discont-extract/trigger", preds, "run-1")
    created: list[ProjectRef] = []

    _projects = 2
    with StudioClient(studio) as client:
        try:
            for index in (1, 2):
                project = client.create(f"eh30pt-e2e-{index}", labeling.config).unwrap()
                created.append(project)

            ids = {project.id for project in created}
            assert len(ids) == _projects

            titles = {project.title for project in client.projects().unwrap()}
            assert {"eh30pt-e2e-1", "eh30pt-e2e-2"} <= titles
        finally:
            for project in created:
                client.delete(project.id).unwrap()


def test_validate_rejects_a_malformed_label_config(
    studio: LabelStudioSettings,
    project: ProjectRef,
) -> None:
    with StudioClient(studio) as client:
        assert isinstance(client.validate(project.id, "<View><Text"), Failure)


def test_delete_removes_the_project(
    studio: LabelStudioSettings, preds: pl.DataFrame
) -> None:
    labeling = extraction_labeling("discont-extract/trigger", preds, "run-1")

    with StudioClient(studio) as client:
        created = client.create("eh30pt-e2e-delete", labeling.config).unwrap()
        assert created.id in {project.id for project in client.projects().unwrap()}

        client.delete(created.id).unwrap()
        assert created.id not in {project.id for project in client.projects().unwrap()}


def test_export_labels_writes_annotations_to_parquet(
    studio: LabelStudioSettings,
    project: ProjectRef,
    preds: pl.DataFrame,
    tmp_path: Path,
) -> None:
    from eh30pt.cli.scripts.export_labels import main

    labeling = extraction_labeling("discont-extract/trigger", preds, "run-1")
    output = tmp_path / "labels.parquet"
    labels = get_labels("discont-classifier/extract-labels")
    controls = extraction_controls()

    with StudioClient(studio) as client:
        client.import_tasks(project.id, labeling.tasks).unwrap()
        imported = client.tasks(project.id).unwrap()

        first: StudioResult = {
            "id": "1",
            "from_name": controls[0],
            "to_name": "text",
            "type": "choices",
            "value": {"choices": [labels[0]]},
        }

        second: StudioResult = {
            "id": "2",
            "from_name": controls[2],
            "to_name": "text",
            "type": "choices",
            "value": {"choices": [labels[1]]},
        }
        # Meta-box flags never export, so only the first choice below survives.

        client.annotate(imported[0]["id"], [first, second]).unwrap()

    main(
        settings=studio,
        projects=[project.id],
        dataset="discont-extract/trigger",
        output=output,
    )

    df = ExtractionFrameSchema.validate(pl.read_parquet(output))

    _rows = 1
    assert df.height == _rows
    assert df["label"].to_list() == [labels[0]]
    assert df["drug"].to_list() == [DiscontExtract.extract_shots[0].extractions["drug"]]


def test_export_labels_combines_projects_into_one_frame(
    studio: LabelStudioSettings,
    project: ProjectRef,
    preds: pl.DataFrame,
    tmp_path: Path,
) -> None:
    from eh30pt.cli.scripts.export_labels import main

    labeling = extraction_labeling("discont-extract/trigger", preds, "run-1")
    output = tmp_path / "labels.parquet"
    labels = get_labels("discont-classifier/extract-labels")
    controls = extraction_controls()

    _needed = 2
    assert len(labeling.tasks) >= _needed

    choice: StudioResult = {
        "id": "1",
        "from_name": controls[0],
        "to_name": "text",
        "type": "choices",
        "value": {"choices": [labels[1]]},
    }

    with StudioClient(studio) as client:
        other = client.create("eh30pt-e2e-other", labeling.config).unwrap()
        try:
            client.import_tasks(project.id, labeling.tasks[:1]).unwrap()
            client.import_tasks(other.id, labeling.tasks[1:]).unwrap()

            first = client.tasks(project.id).unwrap()
            second = client.tasks(other.id).unwrap()
            client.annotate(first[0]["id"], [choice]).unwrap()
            client.annotate(second[0]["id"], [choice]).unwrap()

            main(
                settings=studio,
                projects=[project.id, other.id],
                dataset="discont-extract/trigger",
                output=output,
            )
        finally:
            client.delete(other.id).unwrap()

    df = ExtractionFrameSchema.validate(pl.read_parquet(output))

    _rows = 2
    assert df.height == _rows
    assert df["label"].to_list() == [labels[1], labels[1]]

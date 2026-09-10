"""MLflow dataset inputs pointing at each dataset origin."""

from typing import TYPE_CHECKING

from mlflow.data.code_dataset_source import CodeDatasetSource
from mlflow.data.http_dataset_source import HTTPDatasetSource
from mlflow.data.huggingface_dataset_source import HuggingFaceDatasetSource
from mlflow.data.meta_dataset import MetaDataset

from eh30pt.datasets.oblique import Obl
from eh30pt.esthealth.models import CdaEvent, CdaTriggerEvent

if TYPE_CHECKING:
    from mlflow.data.dataset_source import DatasetSource

    from eh30pt.types import DatasetName


class MetadataOnlyDataset(MetaDataset):
    """Metadata-only dataset with an explicit profile for static typechecking."""

    @property
    def profile(self) -> None:
        """Return no summary statistics for metadata-only datasets."""
        return None


def dataset_source(dataset: DatasetName) -> DatasetSource:
    """Return the source recording where dataset originates."""
    if dataset == "multinerd":
        return HuggingFaceDatasetSource(path="Babelscape/multinerd")

    if dataset == "oblique":
        return HTTPDatasetSource(url=Obl.data_permalink)

    if dataset == "discont-decider":
        tags = {
            "database": "eh30",
            "tables": [CdaTriggerEvent.__tablename__],
            "schema": CdaTriggerEvent.__table__.schema,
        }

        return CodeDatasetSource(tags=tags)

    if dataset == "discont-extract":
        tags = {
            "database": "eh30",
            "tables": [CdaTriggerEvent.__tablename__],
            "schema": CdaTriggerEvent.__table__.schema,
        }

        return CodeDatasetSource(tags=tags)

    if dataset == "discont-classifier":
        tags = {
            "database": "eh30",
            "tables": [CdaEvent.__tablename__],
            "schema": CdaEvent.__table__.schema,
        }

        return CodeDatasetSource(tags=tags)

    msg = f"invalid dataset {dataset}"
    raise ValueError(msg)


def dataset_input(dataset: DatasetName) -> MetadataOnlyDataset:
    """Build a metadata-only input for dataset without fetching its rows."""
    return MetadataOnlyDataset(source=dataset_source(dataset), name=dataset)

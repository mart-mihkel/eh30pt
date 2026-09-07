"""Test configuration."""

from typing import TYPE_CHECKING, NamedTuple

import pytest
import torch

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from transformers import PreTrainedConfig, PreTrainedModel, PreTrainedTokenizerFast

    from eh30pt.datasets.multinerd import Multinerd
    from eh30pt.types import DatasetProtocol


class ModelSetup(NamedTuple):
    """Model and it's contemporaries."""

    config: PreTrainedConfig
    tokenizer: PreTrainedTokenizerFast
    dataset: Multinerd
    model: PreTrainedModel


models: Mapping[str, str] = {
    "bert": "hf-internal-testing/tiny-random-bert",
    "gpt2": "hf-internal-testing/tiny-random-gpt2",
    "t5": "hf-internal-testing/tiny-random-t5",
}


@pytest.fixture(scope="session")
def device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")

    if torch.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def _make_config_fixture(
    path: str,
) -> Callable[[], PreTrainedTokenizerFast]:
    @pytest.fixture(scope="session")
    def fixture() -> PreTrainedTokenizerFast:
        from transformers import AutoConfig

        return AutoConfig.from_pretrained(path)

    return fixture


def _make_tokenizer_fixture(
    path: str,
) -> Callable[[], PreTrainedTokenizerFast]:
    @pytest.fixture(scope="session")
    def fixture() -> PreTrainedTokenizerFast:
        from eh30pt.modeling.tokenizer import load_tokenizer

        return load_tokenizer(path)

    return fixture


def _make_dataset_fixture(
    name: str,
) -> Callable[[pytest.FixtureRequest], DatasetProtocol]:
    @pytest.fixture(scope="session")
    def fixture(request: pytest.FixtureRequest) -> DatasetProtocol:
        from eh30pt.datasets.registry import get_dataset
        from eh30pt.modeling.arch import infer_architecture

        config: PreTrainedConfig = request.getfixturevalue(f"{name}_config")
        tokenizer: PreTrainedTokenizerFast = request.getfixturevalue(
            f"{name}_tokenizer"
        )

        arch = infer_architecture(config)
        ds = get_dataset("multinerd", tokenizer, arch)
        ds.load()

        return ds

    return fixture


def _make_model_fixture(
    name: str,
    path: str,
) -> Callable[[pytest.FixtureRequest], PreTrainedModel]:
    @pytest.fixture(scope="session")
    def fixture(request: pytest.FixtureRequest) -> PreTrainedModel:
        from eh30pt.modeling.arch import infer_architecture
        from eh30pt.modeling.loader import load_model

        config: PreTrainedConfig = request.getfixturevalue(f"{name}_config")
        tokenizer: PreTrainedTokenizerFast = request.getfixturevalue(
            f"{name}_tokenizer"
        )

        data: DatasetProtocol = request.getfixturevalue(f"{name}_dataset")
        arch = infer_architecture(config)

        return load_model(tokenizer, path, data, arch, head_only=False)

    return fixture


for _name, _path in models.items():
    config_name = f"{_name}_config"
    tokenizer_name = f"{_name}_tokenizer"
    dataset_name = f"{_name}_dataset"
    model_name = f"{_name}_model"

    globals()[config_name] = _make_config_fixture(_path)
    globals()[tokenizer_name] = _make_tokenizer_fixture(_path)
    globals()[dataset_name] = _make_dataset_fixture(_name)
    globals()[model_name] = _make_model_fixture(_name, _path)


@pytest.fixture(params=list(models), scope="session", ids=list(models))
def model_setup(request: pytest.FixtureRequest) -> ModelSetup:
    name = request.param
    return ModelSetup(
        request.getfixturevalue(f"{name}_config"),
        request.getfixturevalue(f"{name}_tokenizer"),
        request.getfixturevalue(f"{name}_dataset"),
        request.getfixturevalue(f"{name}_model"),
    )

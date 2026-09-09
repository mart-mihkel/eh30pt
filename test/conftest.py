"""Test configuration."""

from typing import TYPE_CHECKING, Final, NamedTuple, cast

import datasets as ds
import pytest
import torch

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from datasets.splits import Split
    from transformers import PreTrainedConfig, PreTrainedModel, PreTrainedTokenizerFast

    from eh30pt.datasets.multinerd import Multinerd
    from eh30pt.datasets.oblique import Oblique
    from eh30pt.types import DatasetName, DatasetProtocol

ds.disable_caching()

models: Final[Mapping[str, str]] = {
    "bert": "hf-internal-testing/tiny-random-bert",
    "modernbert": "hf-internal-testing/tiny-random-ModernBertForSequenceClassification",
    "gpt2": "hf-internal-testing/tiny-random-gpt2",
    "gpt-neox": "hf-internal-testing/tiny-random-GPTNeoXForCausalLM",
    "llama": "hf-internal-testing/tiny-random-LlamaForCausalLM",
    # extremely slow or broken, not sure but should be fixed
    # "qwen": "hf-internal-testing/tiny-random-Qwen3MoeForCausalLM",  # noqa: ERA001
    # "gemma": "hf-internal-testing/tiny-random-Gemma3ForCausalLM",  # noqa: ERA001
    "t5": "hf-internal-testing/tiny-random-t5",
}

datasets: tuple[DatasetName, ...] = ("multinerd", "oblique")


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--run-slow",
        default=False,
        action="store_true",
        help="run tests marked as slow",
    )


def pytest_collection_modifyitems(
    config: pytest.Config,
    items: list[pytest.Item],
) -> None:
    if config.getoption("--run-slow"):
        return

    skip_slow = pytest.mark.skip(reason="pass --run-slow to run")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)


class ModelSetup(NamedTuple):
    """One model under test with its config, tokenizer, dataset, and weights."""

    config: PreTrainedConfig
    tokenizer: PreTrainedTokenizerFast
    dataset: Multinerd | Oblique
    model: PreTrainedModel


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
    dataset: DatasetName,
    model: str,
) -> Callable[[pytest.FixtureRequest], DatasetProtocol]:
    split = {
        "train": "train[:64]",
        "validation": "validation[:64]",
        "test": "test[:64]",
    }

    split = cast("Split", split)

    @pytest.fixture(scope="session")
    def fixture(request: pytest.FixtureRequest) -> DatasetProtocol:
        from eh30pt.datasets.registry import get_dataset
        from eh30pt.modeling.arch import infer_architecture

        config: PreTrainedConfig = request.getfixturevalue(f"{model}_config")
        tokenizer: PreTrainedTokenizerFast = request.getfixturevalue(
            f"{model}_tokenizer"
        )

        arch = infer_architecture(config)
        ds = get_dataset(dataset, tokenizer, arch)

        if dataset == "multinerd":
            cast("Multinerd", ds).load(split=split, filter_en=False)
        else:
            ds.load(split=split)

        return ds

    return fixture


def _make_model_fixture(
    dataset: DatasetName,
    model: str,
    path: str,
) -> Callable[[pytest.FixtureRequest], PreTrainedModel]:
    @pytest.fixture(scope="session")
    def fixture(request: pytest.FixtureRequest) -> PreTrainedModel:
        from eh30pt.modeling.arch import infer_architecture
        from eh30pt.modeling.loader import load_model

        config: PreTrainedConfig = request.getfixturevalue(f"{model}_config")
        tokenizer: PreTrainedTokenizerFast = request.getfixturevalue(
            f"{model}_tokenizer"
        )

        data: DatasetProtocol = request.getfixturevalue(f"{dataset}_{model}_dataset")
        arch = infer_architecture(config)

        return load_model(tokenizer, path, data, arch, head_only=False)

    return fixture


for _model, _path in models.items():
    globals()[f"{_model}_config"] = _make_config_fixture(_path)
    globals()[f"{_model}_tokenizer"] = _make_tokenizer_fixture(_path)

    for _dataset in datasets:
        globals()[f"{_dataset}_{_model}_dataset"] = _make_dataset_fixture(
            _dataset,
            _model,
        )

        globals()[f"{_dataset}_{_model}_model"] = _make_model_fixture(
            _dataset,
            _model,
            _path,
        )

_matrix = [(dataset, model) for dataset in datasets for model in models]
_matrix_ids = [f"{dataset}-{model}" for dataset, model in _matrix]


@pytest.fixture(params=_matrix, scope="session", ids=_matrix_ids)
def model_setup(request: pytest.FixtureRequest) -> ModelSetup:
    dataset, model = request.param
    return ModelSetup(
        request.getfixturevalue(f"{model}_config"),
        request.getfixturevalue(f"{model}_tokenizer"),
        request.getfixturevalue(f"{dataset}_{model}_dataset"),
        request.getfixturevalue(f"{dataset}_{model}_model"),
    )

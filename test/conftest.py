"""Test configuration."""

from typing import TYPE_CHECKING, cast

import pytest

if TYPE_CHECKING:
    from datasets.dataset_dict import DatasetDict
    from datasets.splits import Split
    from transformers import PreTrainedTokenizerFast

_SPLIT = {
    "train": "train[:10]",
    "validation": "validation[:10]",
    "test": "test[:10]",
}

_SPLIT_ESTNER = {
    "train": "train[:10]",
    "dev": "dev[:10]",
    "test": "test[:10]",
}


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--run-slow",
        action="store_true",
        default=False,
        help="also run tests marked 'slow'",
    )


def pytest_collection_modifyitems(
    config: pytest.Config,
    items: list[pytest.Item],
) -> None:
    if config.getoption("--run-slow"):
        return

    skip_slow = pytest.mark.skip(reason="use --run-slow to run")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)


@pytest.fixture(scope="session")
def bert_path() -> str:
    return "hf-internal-testing/tiny-random-bert"


@pytest.fixture(scope="session")
def gpt2_path() -> str:
    return "hf-internal-testing/tiny-random-gpt2"


@pytest.fixture(scope="session")
def t5_path() -> str:
    return "hf-internal-testing/tiny-random-t5"


@pytest.fixture(scope="session")
def llama_path() -> str:
    return "hf-internal-testing/tiny-random-llama4"


@pytest.fixture(scope="session")
def gemma_path() -> str:
    return "hf-internal-testing/tiny-random-Gemma3ForCausalLM"


@pytest.fixture(scope="session")
def qwen_path() -> str:
    return "Jiqing/tiny-random-qwen2"


@pytest.fixture(scope="session")
def bert_tokenizer(bert_path: str) -> PreTrainedTokenizerFast:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(bert_path)
    return cast("PreTrainedTokenizerFast", tokenizer)


@pytest.fixture(scope="session")
def gpt2_tokenizer(gpt2_path: str) -> PreTrainedTokenizerFast:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(gpt2_path)
    tokenizer = cast("PreTrainedTokenizerFast", tokenizer)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    return tokenizer


@pytest.fixture(scope="session")
def t5_tokenizer(t5_path: str) -> PreTrainedTokenizerFast:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(t5_path)
    return cast("PreTrainedTokenizerFast", tokenizer)


@pytest.fixture(scope="session")
def llama_tokenizer(llama_path: str) -> PreTrainedTokenizerFast:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(llama_path)
    tokenizer = cast("PreTrainedTokenizerFast", tokenizer)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    return tokenizer


@pytest.fixture(scope="session")
def gemma_tokenizer(gemma_path: str) -> PreTrainedTokenizerFast:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(gemma_path)
    tokenizer = cast("PreTrainedTokenizerFast", tokenizer)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    return tokenizer


@pytest.fixture(scope="session")
def qwen_tokenizer(qwen_path: str) -> PreTrainedTokenizerFast:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(qwen_path)
    tokenizer = cast("PreTrainedTokenizerFast", tokenizer)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    return tokenizer


@pytest.fixture(scope="session")
def boolq() -> DatasetDict:
    from datasets.load import load_dataset

    split = cast("Split", _SPLIT)
    data = load_dataset("aps/super_glue", "boolq", split=split)
    return cast("DatasetDict", data)


@pytest.fixture(scope="session")
def wic() -> DatasetDict:
    from datasets.load import load_dataset

    split = cast("Split", _SPLIT)
    data = load_dataset("aps/super_glue", "wic", split=split)
    return cast("DatasetDict", data)


@pytest.fixture(scope="session")
def estner() -> DatasetDict:
    from datasets.load import load_dataset

    split = cast("Split", _SPLIT_ESTNER)
    data = load_dataset("tartuNLP/EstNER", split=split)
    return cast("DatasetDict", data)


@pytest.fixture(scope="session")
def multinerd() -> DatasetDict:
    from datasets.load import load_dataset
    from datasets.utils.info_utils import VerificationMode

    split = cast("Split", _SPLIT)
    data = load_dataset(
        "Babelscape/multinerd",
        verification_mode=VerificationMode.NO_CHECKS,
        split=split,
    )

    return cast("DatasetDict", data)

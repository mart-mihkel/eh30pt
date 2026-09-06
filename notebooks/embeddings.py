import marimo

__generated_with = "0.23.15"
app = marimo.App()

with app.setup:
    from typing import TYPE_CHECKING, cast

    from transformers import (
        AutoModel,
        AutoTokenizer,
        PreTrainedModel,
        PreTrainedTokenizer,
    )

    from eh30pt.logging import logger, setup_logging

    if TYPE_CHECKING:
        from torch import Tensor

    setup_logging()


@app.function
def encode(
    text: str,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
) -> Tensor:
    prefix_tokenized = tokenizer(text, return_tensors="pt")
    token_ids = prefix_tokenized["input_ids"][0]
    return model.get_input_embeddings().forward(token_ids)


@app.function
def decode(
    embeddings: Tensor,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
) -> str | list[str]:
    voc_embeddings = cast("Tensor", model.get_input_embeddings().weight)
    similarity = embeddings @ voc_embeddings.T
    token_ids = similarity.argmax(dim=1)
    return tokenizer.decode(token_ids=token_ids)


@app.cell
def _():
    model_path = "hf-internal-testing/tiny-random-bert"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModel.from_pretrained(model_path)
    return model, tokenizer


@app.cell
def _(model, tokenizer):
    embeddings = encode(
        text="I am a horse",
        model=model,
        tokenizer=tokenizer,
    )

    decoded = decode(
        embeddings=embeddings,
        model=model,
        tokenizer=tokenizer,
    )

    logger.info(decoded)
    return


if __name__ == "__main__":
    app.run()

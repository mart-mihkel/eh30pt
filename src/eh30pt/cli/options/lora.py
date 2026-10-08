"""LoRA adapter options."""

from typing import Annotated

from typer import Option

LoraRankOption = Annotated[
    int,
    Option("--lora-rank", min=1, help="LoRA adapter rank"),
]

LoraAlphaOption = Annotated[
    int,
    Option("--lora-alpha", min=1, help="LoRA scaling factor"),
]

LoraDropoutOption = Annotated[
    float,
    Option("--lora-dropout", min=0, max=1, help="LoRA dropout probability"),
]

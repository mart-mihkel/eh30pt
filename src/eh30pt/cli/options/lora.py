"""LoRA adapter options."""

import rich_click as click

# Unannotated on purpose: click decorators are generic over the decorated
# callable, which a variable annotation cannot express, so the precise type
# is left for inference.
lora_rank_option = click.option(
    "--lora-rank",
    "lora_rank",
    type=click.IntRange(min=1),
    default=16,
    show_default=True,
    help="LoRA adapter rank",
)

lora_alpha_option = click.option(
    "--lora-alpha",
    "lora_alpha",
    type=click.IntRange(min=1),
    default=16,
    show_default=True,
    help="LoRA scaling factor",
)

lora_dropout_option = click.option(
    "--lora-dropout",
    "lora_dropout",
    type=click.FloatRange(min=0, max=1),
    default=0.05,
    show_default=True,
    help="LoRA dropout probability",
)

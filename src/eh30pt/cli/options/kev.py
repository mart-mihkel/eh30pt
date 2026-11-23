"""Kev-style decision model options."""

import rich_click as click

init_from_option = click.option(
    "--init-from",
    "init_from",
    type=str,
    default=None,
    help="Warm start from an existing decision checkpoint (local path or Hub id)",
)

head_dim_option = click.option(
    "--head-dim",
    "head_dim",
    type=click.IntRange(min=1),
    default=256,
    show_default=True,
    help="Pointer head projection size",
)

max_state_option = click.option(
    "--max-state",
    "max_state",
    type=click.IntRange(min=1),
    default=384,
    show_default=True,
    help="Token budget of the state, delimiter included",
)

max_branch_option = click.option(
    "--max-branch",
    "max_branch",
    type=click.IntRange(min=1),
    default=1024,
    show_default=True,
    help="Token budget of one state plus question row",
)

kev_device_option = click.option(
    "--device",
    "device",
    type=click.Choice(["auto", "cpu", "cuda", "mps"]),
    default="auto",
    show_default=True,
    help="Device to train on",
)

weight_decay_option = click.option(
    "--weight-decay",
    "weight_decay",
    type=click.FloatRange(min=0),
    default=0.01,
    show_default=True,
    help="AdamW weight decay",
)

import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium", auto_download=["html"])

with app.setup:
    import marimo as mo
    import plotnine as pn
    import polars as pl

    from eh30pt.logging import setup_logging

    setup_logging()


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Preamble
    """)
    return


@app.function
def theme(base_size: int = 11, base_family: str = "DejaVu Sans") -> pn.theme:
    background = "#FFFFFF"
    text = "#222222"
    axis = "#666666"
    grid = "#CCCCCC"

    return pn.theme_minimal(
        base_size=base_size,
        base_family=base_family,
    ) + pn.theme(
        panel_background=pn.element_rect(fill=background, color=background),
        plot_background=pn.element_rect(fill=background, color=background),
        panel_grid_major=pn.element_line(color=grid, size=0.4),
        panel_grid_minor=pn.element_blank(),
        axis_text=pn.element_text(color=text),
        axis_title=pn.element_text(weight="normal"),
        plot_title=pn.element_text(weight="normal", size=base_size),
        plot_subtitle=pn.element_text(size=base_size * 0.8),
        plot_caption=pn.element_text(size=base_size * 0.7, color=axis),
        legend_margin=2,
        legend_position="top",
        legend_box_background=pn.element_blank(),
        legend_key=pn.element_blank(),
        legend_title=pn.element_text(weight="normal"),
        legend_background=pn.element_rect(
            fill="#D8D8D8",
            color="#FFFFFF",
            alpha=0.25,
        ),
        strip_background=pn.element_rect(
            fill="#D8D8D8",
            color="#FFFFFF",
            alpha=0.25,
        ),
        strip_text=pn.element_text(weight="normal"),
        panel_border=pn.element_rect(color="#D8D8D8", alpha=0.25),
        figure_size=(6, 4),
    )


@app.cell
def _():
    shapes = ["o", "s", "D", "^", "v", "<", ">", "*", "p", "h", "8", "+", "x"]

    method_colors = {
        "5-shot": "#C4AD66",
        "cls-head": "#8C7530",
        "fine-tune": "#4878CF",
        "qlora": "#77BEDB",
        "prompt-tune-random": "#3E8A3A",
        "prompt-tune-pretrained": "#6ACC65",
    }

    method_labels = {
        "5-shot": "FS (5)",
        "cls-head": "Head",
        "fine-tune": "FT",
        "prompt-tune-pretrained": "PT",
    }

    # method_order = [
    #     "5-shot",
    #     "few-shot",
    #     "cls-head",
    #     "fine-tune",
    #     "prompt-tune-pretrained",
    # ]

    model_order = [
        "jhu-clsp/mmBERT-base",
        "google/gemma-3-4b-it",
        "Qwen/Qwen3.5-9B",
    ]

    model_labels = {
        "jhu-clsp/mmBERT-base": "mmBERT (307M)",
        "google/gemma-3-4b-it": "Gemma 3 (4.3B)",
        "Qwen/Qwen3.5-9B": "Qwen 3.5 (9B)",
    }

    arch_labels = {
        "encoder": "Encoder",
        "decoder": "Decoder",
    }

    # arch_order = ["encoder", "decoder", "encoder-decoder"]
    return (
        arch_labels,
        method_colors,
        method_labels,
        model_labels,
        model_order,
        shapes,
    )


@app.cell
def _():
    # dfeh = pl.from_pandas(
    #     mlflow.search_runs(experiment_ids=[mlflow.get_experiment("10").experiment_id])
    # )
    #
    # dfeh = dfeh.rename(
    #     {col: col.replace(".", "_") for col in dfeh.columns}
    # ).with_columns(
    #     pl.when(pl.col("params_method") == "prompt-tune")
    #     .then(pl.col("params_method") + pl.lit("-") + pl.col("params_prefix_init"))
    #     .otherwise(pl.col("params_method"))
    #     .cast(pl.Enum(method_order))
    #     .alias("params_method"),
    #     pl.col("params_base_model")
    #     .str.split("/")
    #     .list.last()
    #     .cast(pl.Enum(model_order)),
    #     pl.col("params_architecture").cast(pl.Enum(arch_order)),
    #     pl.col("params_train_size").fill_null(0).cast(pl.Int64),
    # )

    dfmn = pl.read_parquet("log/df-multinerd.parquet")
    return (dfmn,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Compute
    """)
    return


@app.cell
def _(dfmn):
    dfmn
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Plots
    """)
    return


@app.cell
def _(
    arch_labels,
    dfmn,
    method_colors,
    method_labels,
    model_labels,
    model_order,
    shapes,
):
    _sizes = [10.0, 100.0, 1000.0, 20000.0]
    _methods = ["cls-head", "fine-tune", "prompt-tune-pretrained"]

    _df_fewshot = dfmn.filter(
        pl.col("method") == "5-shot",
        pl.col("base_model").is_in(model_order),
    )

    _df = dfmn.filter(
        pl.col("method").is_in(_methods),
        pl.col("base_model").is_in(model_order),
    ).with_columns(pl.col("base_model").cast(pl.Enum(model_order)))

    _p = (
        pn.ggplot(_df)
        + pn.aes(
            x="train_samples",
            y="test_f1",
            fill="method",
        )
        + pn.labs(
            x="Trainset size",
            y="F1",
            fill="",
            color="",
            shape="",
        )
        + pn.scale_x_log10(
            expand=(0.2, 0),
            breaks=_sizes,
            labels=["10", "100", "1K", "20K"],
        )
        + pn.scale_y_continuous(
            breaks=[0, 0.25, 0.5, 0.75, 1.0],
            labels=["0%", "25%", "50%", "75%", "100%"],
            limits=(0, 1),
        )
        + pn.geom_line(
            pn.aes(color="method", linetype="method"),
            data=_df.filter(pl.col("method") != "5-shot"),
        )
        + pn.geom_hline(
            pn.aes(yintercept="test_f1", color="method", linetype="method"),
            data=_df_fewshot.with_columns(
                pl.col("base_model").cast(pl.Enum(model_order)),
            ),
            inherit_aes=False,
        )
        + pn.geom_point(
            pn.aes(color="method"),
            data=_df.filter(pl.col("method") != "5-shot"),
            stroke=0.3,
            size=2,
        )
        + pn.scale_color_manual(
            values=method_colors,
            breaks=list(method_labels),
            labels=method_labels,
        )
        + pn.scale_linetype_manual(
            values={
                "5-shot": "dashed",
                "cls-head": "solid",
                "fine-tune": "solid",
                "prompt-tune-pretrained": "solid",
            },
            labels=method_labels,
        )
        + pn.scale_fill_manual(values=method_colors, labels=method_labels)
        + pn.scale_shape_manual(values=shapes, labels=arch_labels)
        + pn.facet_grid(
            "~base_model",
            labeller=lambda s: model_labels.get(s, s),  # ty:ignore[invalid-argument-type]
        )
        + theme()
        + pn.theme(
            figure_size=(6, 2),
            legend_position="inside",
            legend_position_inside=(1, 0),
            legend_justification_inside=(1, 0),
            legend_background=pn.element_blank(),
            strip_background=pn.element_blank(),
        )
        + pn.guides(
            color=pn.guide_legend(
                nrow=1,
                override_aes={
                    "linetype": ["dashed", "solid", "solid", "solid"],
                    "shape": ["", "o", "o", "o"],
                },
            ),
            linetype="none",
            fill="none",
            shape="none",
        )
    )

    _p.save("log/multinerd-data-scaling.png", dpi=300)
    _p
    return


if __name__ == "__main__":
    app.run()

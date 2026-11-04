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
        legend_box_background=pn.element_blank(),
        legend_key=pn.element_blank(),
        legend_title=pn.element_text(weight="normal"),
        legend_position="inside",
        legend_position_inside=(1, 0),
        legend_justification_inside=(1, 0),
        legend_background=pn.element_blank(),
        strip_background=pn.element_blank(),
        strip_text=pn.element_text(weight="normal"),
        panel_border=pn.element_rect(color="#D8D8D8", alpha=0.25),
    )


@app.cell
def _():
    method_colors = {
        "5-shot": "#C4AD66",
        "few-shot": "#C4AD66",
        "cls-head": "#8C7530",
        "fine-tune": "#4878CF",
        "qlora": "#77BEDB",
        "prompt-tune-random": "#3E8A3A",
        "prompt-tune-pretrained": "#6ACC65",
    }

    method_labels = {
        "5-shot": "FS (5)",
        "few-shot": "FS (5)",
        "cls-head": "Head",
        "fine-tune": "FT",
        "prompt-tune-pretrained": "PT",
    }

    method_order = [
        "5-shot",
        "few-shot",
        "cls-head",
        "fine-tune",
        "qlora",
        "prompt-tune-random",
        "prompt-tune-pretrained",
    ]

    model_order = [
        "jhu-clsp/mmBERT-base",
        "mmBERT-base",
        "google/gemma-3-4b-it",
        "medgemma-4b-it",
        "Qwen/Qwen3.5-9B",
        "Qwen3.5-9B",
        "Llama-3.1-8B-Instruct",
        "Llama-3.1-EstLLM-8B-Instruct-1125",
    ]

    model_labels = {
        "jhu-clsp/mmBERT-base": "mmBERT (307M)",
        "mmBERT-base": "mmBERT (307M)",
        "medgemma-4b-it": "Medgemma (4.3B)",
        "google/gemma-3-4b-it": "Gemma 3 (4.3B)",
        "Qwen/Qwen3.5-9B": "Qwen 3.5 (9B)",
        "Qwen3.5-9B": "Qwen 3.5 (9B)",
        "Llama-3.1-8B-Instruct": "Llama 3.1 (8B)",
        "Llama-3.1-EstLLM-8B-Instruct-1125": "Llama 3.1 EstLLM (8B)",
    }

    arch_order = ["encoder", "decoder", "encoder-decoder"]
    return (
        arch_order,
        method_colors,
        method_labels,
        method_order,
        model_labels,
        model_order,
    )


@app.cell
def _(arch_order, method_order):
    dfmn = pl.read_parquet("log/df-multinerd-scaling.parquet").with_columns(
        pl.col("architecture").cast(pl.Enum(arch_order)),
        pl.col("method").cast(pl.Enum(method_order)),
    )

    dfeh = pl.read_parquet("log/df-eh30-scaling.parquet").with_columns(
        pl.col("params_architecture").cast(pl.Enum(arch_order)),
        pl.col("params_method").cast(pl.Enum(method_order)),
    )
    return dfeh, dfmn


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Plots
    """)
    return


@app.cell
def _(dfmn, method_colors, method_labels, model_labels, model_order):
    _sizes = [10.0, 100.0, 1000.0, 20000.0]
    _methods = ["5-shot", "cls-head", "fine-tune", "prompt-tune-pretrained"]

    _dffs = dfmn.filter(
        pl.col("method") == "5-shot",
        pl.col("base_model").is_in(model_order),
    )

    _df = dfmn.filter(
        pl.col("method").is_in(_methods[1:]),
        pl.col("base_model").is_in(model_order),
    ).with_columns(pl.col("base_model").cast(pl.Enum(model_order)))

    _p = (
        pn.ggplot(_df)
        + pn.aes(
            x="train_samples",
            y="test_f1",
            color="method",
            linetype="method",
        )
        + pn.labs(
            x="",
            y="F1",
            color="",
            linetype="",
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
        + pn.geom_line()
        + pn.geom_hline(
            pn.aes(yintercept="test_f1", color="method", linetype="method"),
            data=_dffs.with_columns(
                pl.col("base_model").cast(pl.Enum(model_order)),
            ),
            inherit_aes=False,
        )
        + pn.geom_point(stroke=0.3, size=2)
        + pn.facet_grid("~base_model", labeller=model_labels.get)
        + pn.scale_color_manual(
            values=method_colors,
            breaks=_methods,
            labels=[method_labels[method] for method in _methods],
        )
        + pn.scale_linetype_manual(
            values={
                method: "dashed" if method == "5-shot" else "solid"
                for method in method_colors
            },
            breaks=_methods,
            labels=[method_labels[method] for method in _methods],
        )
        + pn.guides(color=pn.guide_legend(nrow=1), linetype="none")
        + theme()
        + pn.theme(figure_size=(9, 2))
    )

    _p.save("log/multinerd-data-scaling.png", dpi=300)
    _p
    return


@app.cell
def _(dfeh, method_colors, method_labels, model_labels):
    _models = [
        "mmBERT-base",
        "medgemma-4b-it",
        "Qwen3.5-9B",
    ]

    _df = (
        dfeh.filter(
            pl.col("params_method").ne("qlora"),
            pl.col("params_base_model").is_in(_models),
            # pl.col("params_train_size").is_in([0, 100, 500, 1500, 2500])
        )
        .group_by(
            [
                "params_train_size",
                "params_method",
                "params_architecture",
                "params_base_model",
            ]
        )
        .agg(
            metrics_test_f1_mean=pl.col("metrics_test_f1").mean(),
            metrics_test_f1_low=pl.col("metrics_test_f1").quantile(0.025),
            metrics_test_f1_high=pl.col("metrics_test_f1").quantile(0.975),
        )
        .sort(
            [
                "params_train_size",
                "params_method",
                "params_architecture",
                "params_base_model",
            ]
        )
    )

    _methods = ["few-shot", "cls-head", "fine-tune", "prompt-tune-pretrained"]
    _dffs = _df.filter(pl.col("params_method").eq("few-shot"))
    _dfnfs = _df.filter(pl.col("params_method").ne("few-shot"))

    _p = (
        pn.ggplot(_dfnfs)
        + pn.aes(
            x="params_train_size",
            y="metrics_test_f1_mean",
            color="params_method",
            linetype="params_method",
        )
        + pn.geom_line()
        + pn.geom_hline(
            pn.aes(
                yintercept="metrics_test_f1_mean",
                color="params_method",
                linetype="params_method",
            ),
            data=_dffs,
            inherit_aes=False,
        )
        + pn.geom_point(stroke=0.3, size=2)
        + pn.facet_grid("~params_base_model", labeller=model_labels.get)
        + pn.scale_x_continuous(
            expand=(0.1, 0),
            breaks=[100, 500, 1000, 1500, 2000, 2500],
        )
        + pn.scale_y_continuous(
            breaks=[0, 0.25, 0.5, 0.75, 1.0],
            labels=["0%", "25%", "50%", "75%", "100%"],
            limits=(0, 1),
        )
        + pn.labs(
            x="",
            y="F1",
            color="",
            linetype="",
        )
        + pn.scale_color_manual(
            values=method_colors,
            breaks=_methods,
            labels=[method_labels[method] for method in _methods],
        )
        + pn.scale_linetype_manual(
            values={
                method: "dashed" if method == "few-shot" else "solid"
                for method in method_colors
            },
            breaks=_methods,
            labels=[method_labels[method] for method in _methods],
        )
        + pn.guides(color=pn.guide_legend(nrow=1), linetype="none")
        + theme()
        + pn.theme(figure_size=(9, 2))
    )

    _p.save("log/eh30-data-scaling.png", dpi=300)
    _p
    return


if __name__ == "__main__":
    app.run()

import marimo

__generated_with = "0.23.15"
app = marimo.App(width="medium")

with app.setup:
    from pathlib import Path
    from typing import TYPE_CHECKING, cast

    import marimo as mo
    import plotnine as pn
    import polars as pl

    from eh30pt.constants import LOGDIR
    from eh30pt.logging import setup_logging
    from eh30pt.tracking import collect_runs

    if TYPE_CHECKING:
        from collections.abc import Sequence

    setup_logging()


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Setup
    """)
    return


@app.cell
def _():
    dataset_dropdown = mo.ui.dropdown(
        ["multinerd", "obl"],
        value="multinerd",
        label="Dataset",
    )

    dataset_dropdown
    return (dataset_dropdown,)


@app.cell
def _(dataset_dropdown):
    _is_multinerd = dataset_dropdown.value == "multinerd"

    dataset = dataset_dropdown.value
    dataset_size = 20000 if _is_multinerd else None

    figpath = LOGDIR / "fig" / dataset
    return dataset, dataset_size, figpath


@app.cell
def _():
    shapes = ["o", "s", "D", "^", "v", "<", ">", "*", "p", "h", "8", "+", "x"]
    colors = ["#4878CF", "#6ACC65", "#D65F5F", "#B47CC7", "#C4AD66", "#77BEDB"]
    return colors, shapes


@app.cell
def _():
    method_labels = {
        "5-shot": "Näitepõhine (5)",
        "fine-tune": "Peenhäälestus",
        "cls-head": "Klassifitseerimispea",
        "prompt-tune-random": "Prompt-häälestus (juhuslik)",
        "prompt-tune-pretrained": "Prompt-häälestus (eeltreenitud)",
    }

    method_colors = {
        "5-shot": "#C4AD66",
        "cls-head": "#8C7530",
        "fine-tune": "#4878CF",
        "prompt-tune-random": "#3E8A3A",
        "prompt-tune-pretrained": "#6ACC65",
    }

    metric_labels = {
        "test_f1": "F1",
        "test_precision": "Täpsus",
        "test_recall": "Saagis",
    }

    model_order = [
        "distilbert",
        "modernbert",
        "deberta-v2",
        # "eurobert",
        "gemma3_text",
        "gpt_neox",
        "llama",
        "qwen3_5_text",
        "t5",
        # "t5gemma2",
    ]

    model_labels = {
        "distilbert": "DistilBERT",
        "modernbert": "mmBERT",
        "deberta-v2": "DeBERTa",
        # "eurobert": "EuroBERT",
        "gemma3_text": "Gemma 3",
        "gpt_neox": "GPT-NeoX",
        "llama": "Llama 3",
        "qwen3_5_text": "Qwen 3.5",
        "t5": "Flan-T5",
        # "t5gemma2": "T5Gemma2",
    }

    arch_order = ["encoder", "decoder", "encoder-decoder"]

    arch_labels = {
        "encoder": "Kooder",
        "decoder": "Dekooder",
        "encoder-decoder": "Kooder-dekooder",
    }

    arch_colors = {
        "encoder": "#4878CF",
        "decoder": "#6ACC65",
        "encoder-decoder": "#B47CC7",
    }
    return (
        arch_colors,
        arch_labels,
        arch_order,
        method_colors,
        method_labels,
        metric_labels,
        model_labels,
        model_order,
    )


@app.cell
def _():
    _background = "#FFFFFF"
    _text = "#222222"
    _axis = "#666666"
    _grid = "#CCCCCC"

    def theme(base_size=11, base_family="DejaVu Sans"):
        return pn.theme_minimal(
            base_size=base_size,
            base_family=base_family,
        ) + pn.theme(
            panel_background=pn.element_rect(fill=_background, color=_background),
            plot_background=pn.element_rect(fill=_background, color=_background),
            panel_grid_major=pn.element_line(color=_grid, size=0.4),
            panel_grid_minor=pn.element_blank(),
            axis_text=pn.element_text(color=_text),
            axis_title=pn.element_text(weight="normal"),
            plot_title=pn.element_text(weight="normal", size=base_size),
            plot_subtitle=pn.element_text(size=base_size * 0.8),
            plot_caption=pn.element_text(size=base_size * 0.7, color=_axis),
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

    return (theme,)


@app.cell
def _():
    collected = collect_runs("eh30pt", "sqlite:///mlflow.db", ("loss", "eval_loss"))
    return (collected,)


@app.cell
def _(arch_order, collected, dataset, dataset_size, figpath, model_order):
    Path(figpath).mkdir(parents=True)

    df_raw = collected.metrics.filter(
        pl.col("dataset") == dataset,
        pl.col("status") == "FINISHED",
        pl.col("model_type").is_in([*model_order, "gemma3"]),
    ).with_columns(
        pl.col("model_type")
        .replace({"gemma3": "gemma3_text"})
        .cast(pl.Enum(model_order)),
        pl.col("architecture").cast(pl.Enum(arch_order)),
    )

    if dataset_size is not None:
        df = df_raw.filter(pl.col("train_samples").is_in([0, dataset_size]))
    else:
        df = df_raw

    mo.md(f"Collected metrics of {len(df)} runs")
    return df, df_raw


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Tables
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Cost of compute
    """)
    return


@app.cell
def _(df):
    (
        df.with_columns(
            pl.col("end_time")
            .sub(pl.col("start_time"))
            .mul(1 / 1000)
            .alias("total_runtime"),
        )
        .select("train_runtime", "eval_runtime", "test_runtime", "total_runtime")
        .sum()
        .unpivot(variable_name="task", value_name="time")
        .with_columns(
            pl.col("time").mul(0.5 / 3600).round(2).alias("cost_eur"),
            pl.col("time").mul(1 / 3600).round(2).alias("gpu_hours"),
            pl.col("task").str.replace("_runtime", ""),
        )
        .select("task", "gpu_hours", "cost_eur")
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Parameters
    """)
    return


@app.cell
def _(df):
    (
        df.with_columns(pl.col("base_model").str.split("/").list.last().alias("model"))
        .pivot(
            on="method",
            index=[
                "model",
                "method",
                "architecture",
                "total_parameters",
                "n_virtual",
            ],
            values="trainable_parameters",
        )
        .with_columns(
            pl.when(pl.col("architecture") == "encoder")
            .then(pl.coalesce("cls-head", "fine-tune"))
            .otherwise(None)
            .alias("head_parameters"),
            pl.coalesce("prompt-tune-pretrained", "prompt-tune-random").alias(
                "prompt_parameters",
            ),
        )
        .select(
            "model",
            "method",
            "architecture",
            "total_parameters",
            "head_parameters",
            "prompt_parameters",
            "n_virtual",
        )
        .sort(["architecture", "total_parameters"])
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Perfomance
    """)
    return


@app.cell
def _(df):
    (
        df.with_columns(pl.col("base_model").str.split("/").list.last().alias("model"))
        .pivot(
            on="method",
            index=[
                "model",
                "method",
                "test_accuracy",
                "test_precision",
                "test_recall",
                "test_f1",
            ],
        )
        .select(
            "model",
            "method",
            pl.col("test_accuracy").mul(100).round(0),
            pl.col("test_precision").mul(100).round(0),
            pl.col("test_recall").mul(100).round(0),
            pl.col("test_f1").mul(100).round(0),
        )
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Performance scaling
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### All
    """)
    return


@app.cell
def _(
    arch_labels,
    df,
    figpath,
    method_colors,
    method_labels,
    model_labels,
    shapes,
    theme,
):
    _p = (
        pn.ggplot(df)
        + pn.aes(
            x="total_parameters",
            y="test_f1",
            fill="method",
            shape="architecture",
        )
        + pn.labs(x="Parameetrid", y="F1", fill="", color="", shape="")
        + pn.scale_x_log10(
            breaks=[10**i for i in range(6, 11)],
            labels=["1M", "10M", "100M", "1B", "10B"],
        )
        + pn.scale_y_continuous(
            breaks=[0, 0.25, 0.5, 0.75, 1.0],
            labels=["0%", "25%", "50%", "75%", "100%"],
            limits=(0, 1),
        )
        + pn.facet_wrap(
            "model_type",
            labeller=lambda s: model_labels.get(s, s),  # ty: ignore[invalid-argument-type]
        )
        + pn.geom_line(pn.aes(color="method"))
        + pn.geom_point(stroke=0.3, size=3, color="white")
        + pn.scale_color_manual(values=method_colors, labels=method_labels)
        + pn.scale_fill_manual(values=method_colors, labels=method_labels)
        + pn.scale_shape_manual(values=shapes, labels=arch_labels)
        + theme()
        + pn.theme(figure_size=(8, 7))
        + pn.guides(
            color=pn.guide_legend(ncol=2),
            shape=pn.guide_legend(ncol=1, override_aes={"color": "black"}),
        )
    )

    _p.save(figpath / "model-performance-scaling.png", dpi=300)
    _p
    return


@app.cell
def _(
    arch_labels,
    df,
    figpath,
    method_colors,
    method_labels,
    model_labels,
    shapes,
    theme,
):
    _arch_labels = arch_labels.copy()
    _arch_labels.pop("encoder-decoder")

    _df = df.filter(pl.col("model_type").is_in(["gpt_neox", "t5"]).not_()).with_columns(
        pl.col("architecture").cast(pl.Enum(list(_arch_labels.keys()))),
    )

    _p = (
        pn.ggplot(_df)
        + pn.aes(
            x="total_parameters",
            y="test_f1",
            fill="method",
            shape="architecture",
        )
        + pn.labs(x="Parameetrid", y="F1", fill="", color="", shape="")
        + pn.scale_x_log10(
            breaks=[10**i for i in range(6, 11)],
            labels=["1M", "10M", "100M", "1B", "10B"],
        )
        + pn.scale_y_continuous(
            breaks=[0, 0.25, 0.5, 0.75, 1.0],
            labels=["0%", "25%", "50%", "75%", "100%"],
            limits=(0, 1),
        )
        + pn.facet_wrap(
            "model_type",
            labeller=lambda s: model_labels.get(s, s),  # ty: ignore[invalid-argument-type]
        )
        + pn.geom_line(pn.aes(color="method"))
        + pn.geom_point(stroke=0.3, size=3, color="white")
        + pn.scale_color_manual(values=method_colors, labels=method_labels)
        + pn.scale_fill_manual(values=method_colors, labels=method_labels)
        + pn.scale_shape_manual(values=shapes, labels=_arch_labels)
        + theme()
        + pn.theme(figure_size=(8, 6))
        + pn.guides(
            color=pn.guide_legend(ncol=2),
            shape=pn.guide_legend(ncol=1, override_aes={"color": "black"}),
        )
    )

    _p.save(figpath / "other-eh30ptability-scaling.png", dpi=300)
    _p
    return


@app.cell
def _(
    arch_labels,
    df,
    figpath,
    method_colors,
    method_labels,
    metric_labels,
    model_labels,
    shapes,
    theme,
):
    _idx = [
        c for c in df.columns if c not in ["test_f1", "test_precision", "test_recall"]
    ]

    _df = df.unpivot(
        on=["test_f1", "test_precision", "test_recall"],
        index=_idx,
        variable_name="metric",
        value_name="value",
    ).with_columns(pl.col("metric").replace(metric_labels))

    _p = (
        pn.ggplot(_df)
        + pn.aes(
            x="total_parameters",
            y="value",
            fill="method",
            shape="architecture",
        )
        + pn.labs(
            x="Parameetrid",
            y="",
            fill="",
            color="",
            shape="",
        )
        + pn.scale_x_log10(
            breaks=[10**i for i in range(6, 11)],
            labels=["1M", "10M", "100M", "1B", "10B"],
        )
        + pn.scale_y_continuous(
            breaks=[0, 0.25, 0.5, 0.75, 1.0],
            labels=["0%", "25%", "50%", "75%", "100%"],
            limits=(0, 1),
        )
        + pn.facet_grid(
            "model_type ~ metric",
            labeller=lambda s: model_labels.get(s, s),  # ty: ignore[invalid-argument-type]
        )
        + pn.geom_line(pn.aes(color="method"))
        + pn.geom_point(stroke=0.3, size=3, color="white")
        + pn.scale_color_manual(values=method_colors, labels=method_labels)
        + pn.scale_fill_manual(values=method_colors, labels=method_labels)
        + pn.scale_shape_manual(values=shapes, labels=arch_labels)
        + theme()
        + pn.theme(figure_size=(6, 14))
        + pn.guides(
            color=pn.guide_legend(ncol=2),
            shape=pn.guide_legend(ncol=1, override_aes={"color": "black"}),
        )
    )

    _p.save(figpath / "model-performance-scaling-all-metrics.png", dpi=300)
    _p
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Encoder models
    """)
    return


@app.cell
def _(
    df,
    figpath,
    method_colors,
    method_labels,
    metric_labels,
    model_labels,
    theme,
):
    _idx = [
        c for c in df.columns if c not in ["test_f1", "test_precision", "test_recall"]
    ]

    _df = (
        df.filter(pl.col("architecture") == "encoder")
        .unpivot(
            on=["test_f1", "test_precision", "test_recall"],
            index=_idx,
            variable_name="metric",
            value_name="value",
        )
        .with_columns(pl.col("metric").replace(metric_labels))
    )

    _p = (
        pn.ggplot(_df)
        + pn.aes(x="total_parameters", y="value", fill="method")
        + pn.labs(x="Parameetrid", y="", fill="", color="")
        + pn.scale_x_log10(
            breaks=[10**i for i in range(6, 11)],
            labels=["1M", "10M", "100M", "1B", "10B"],
        )
        + pn.scale_y_continuous(
            breaks=[0, 0.25, 0.5, 0.75, 1.0],
            labels=["0%", "25%", "50%", "75%", "100%"],
            limits=(0, 1),
        )
        + pn.facet_grid(
            "metric ~ model_type",
            labeller=lambda s: model_labels.get(s, s),  # ty: ignore[invalid-argument-type]
        )
        + pn.geom_line(pn.aes(color="method"))
        + pn.geom_point(shape="o", stroke=0.3, size=3, color="white")
        + pn.scale_color_manual(values=method_colors, labels=method_labels)
        + pn.scale_fill_manual(values=method_colors, labels=method_labels)
        + theme()
        + pn.theme(figure_size=(8, 7))
        + pn.guides(color=pn.guide_legend(ncol=2))
    )

    _p.save(figpath / "encoder-performance-scaling.png", dpi=300)
    _p
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Decoder models
    """)
    return


@app.cell
def _(
    df,
    figpath,
    method_colors,
    method_labels,
    metric_labels,
    model_labels,
    theme,
):
    _idx = [
        c for c in df.columns if c not in ["test_f1", "test_precision", "test_recall"]
    ]

    _df = (
        df.filter(pl.col("architecture") == "decoder")
        .unpivot(
            on=["test_f1", "test_precision", "test_recall"],
            index=_idx,
            variable_name="metric",
            value_name="value",
        )
        .with_columns(pl.col("metric").replace(metric_labels))
    )

    _p = (
        pn.ggplot(_df)
        + pn.aes(x="total_parameters", y="value", fill="method")
        + pn.labs(x="Parameetrid", y="", fill="", color="")
        + pn.scale_x_log10(
            breaks=[10**i for i in range(6, 11)],
            labels=["1M", "10M", "100M", "1B", "10B"],
        )
        + pn.scale_y_continuous(
            breaks=[0, 0.25, 0.5, 0.75, 1.0],
            labels=["0%", "25%", "50%", "75%", "100%"],
            limits=(0, 1),
        )
        + pn.facet_grid(
            "metric ~ model_type",
            labeller=lambda s: model_labels.get(s, s),  # ty: ignore[invalid-argument-type]
        )
        + pn.geom_line(pn.aes(color="method"))
        + pn.geom_point(shape="s", stroke=0.3, size=3, color="white")
        + pn.scale_color_manual(values=method_colors, labels=method_labels)
        + pn.scale_fill_manual(values=method_colors, labels=method_labels)
        + theme()
        + pn.theme(figure_size=(8, 7))
        + pn.guides(color=pn.guide_legend(ncol=2))
    )

    _p.save(figpath / "decoder-performance-scaling.png", dpi=300)
    _p
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Encoder-decoder models
    """)
    return


@app.cell
def _(
    df,
    figpath,
    method_colors,
    method_labels,
    metric_labels,
    model_labels,
    theme,
):
    _idx = [
        c for c in df.columns if c not in ["test_f1", "test_precision", "test_recall"]
    ]

    _df = (
        df.filter(pl.col("architecture") == "encoder-decoder")
        .unpivot(
            on=["test_f1", "test_precision", "test_recall"],
            index=_idx,
            variable_name="metric",
            value_name="value",
        )
        .with_columns(pl.col("metric").replace(metric_labels))
    )

    _p = (
        pn.ggplot(_df)
        + pn.aes(x="total_parameters", y="value", fill="method")
        + pn.labs(x="Parameetrid", y="", fill="", color="")
        + pn.scale_x_log10(
            breaks=[10**i for i in range(6, 11)],
            labels=["1M", "10M", "100M", "1B", "10B"],
        )
        + pn.scale_y_continuous(
            breaks=[0, 0.25, 0.5, 0.75, 1.0],
            labels=["0%", "25%", "50%", "75%", "100%"],
            limits=(0, 1),
        )
        + pn.facet_grid(
            "metric ~ model_type",
            labeller=lambda s: model_labels.get(s, s),  # ty: ignore[invalid-argument-type]
        )
        + pn.geom_line(pn.aes(color="method"))
        + pn.geom_point(shape="D", stroke=0.3, size=3, color="white")
        + pn.scale_color_manual(values=method_colors, labels=method_labels)
        + pn.scale_fill_manual(values=method_colors, labels=method_labels)
        + theme()
        + pn.theme(figure_size=(8, 7))
        + pn.guides(color=pn.guide_legend(ncol=2))
    )

    _p.save(figpath / "encoder-decoder-performance-scaling.png", dpi=300)
    _p
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Instructability scaling
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    #### mmBERT
    """)
    return


@app.cell
def _(df, figpath, method_colors, method_labels, theme):
    _df = df.filter(pl.col("model_type") == "modernbert")

    _p = (
        pn.ggplot(_df)
        + pn.aes(x="total_parameters", y="test_f1", fill="method")
        + pn.labs(x="Parameetrid", y="F1", fill="", color="")
        + pn.scale_x_log10(
            expand=(0.2, 0),
            breaks=[1.5e8, 3e8],
            labels=["150M", "300M"],
        )
        + pn.scale_y_continuous(
            breaks=[0, 0.25, 0.5, 0.75, 1.0],
            labels=["0%", "25%", "50%", "75%", "100%"],
            limits=(0, 1),
        )
        + pn.geom_line(pn.aes(color="method"))
        + pn.geom_point(shape="o", stroke=0.3, size=3.5, color="white")
        + pn.scale_color_manual(values=method_colors, labels=method_labels)
        + pn.scale_fill_manual(values=method_colors, labels=method_labels)
        + theme()
        + pn.theme(figure_size=(5, 4))
        + pn.guides(color=pn.guide_legend(ncol=2))
    )

    _p.save(figpath / "mmbert-eh30ptability-scaling.png", dpi=300)
    _p
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    #### GPT-NeoX
    """)
    return


@app.cell
def _(df, figpath, method_colors, method_labels, theme):
    _df = df.filter(pl.col("model_type") == "gpt_neox")

    _p = (
        pn.ggplot(_df)
        + pn.aes(x="total_parameters", y="test_f1", fill="method")
        + pn.labs(x="Parameetrid", y="F1", fill="", color="")
        + pn.scale_x_log10(
            expand=(0.1, 0),
            labels=["100M", "1B", "10B"],
        )
        + pn.scale_y_continuous(
            breaks=[0, 0.25, 0.5, 0.75, 1.0],
            labels=["0%", "25%", "50%", "75%", "100%"],
            limits=(0, 1),
        )
        + pn.geom_line(pn.aes(color="method"))
        + pn.geom_point(shape="s", stroke=0.3, size=3.5, color="white")
        + pn.scale_color_manual(values=method_colors, labels=method_labels)
        + pn.scale_fill_manual(values=method_colors, labels=method_labels)
        + theme()
        + pn.theme(figure_size=(5, 4))
        + pn.guides(color=pn.guide_legend(ncol=2))
    )

    _p.save(figpath / "gpt-neox-eh30ptability-scaling.png", dpi=300)
    _p
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    #### Flan-T5
    """)
    return


@app.cell
def _(df, figpath, method_colors, method_labels, theme):
    _df = df.filter(pl.col("model_type") == "t5")

    _p = (
        pn.ggplot(_df)
        + pn.aes(x="total_parameters", y="test_f1", fill="method")
        + pn.labs(x="Parameetrid", y="F1", fill="", color="")
        + pn.scale_x_log10(expand=(0.1, 0), labels=["100M", "1B", "10B"])
        + pn.scale_y_continuous(
            breaks=[0, 0.25, 0.5, 0.75, 1.0],
            labels=["0%", "25%", "50%", "75%", "100%"],
            limits=(0, 1),
        )
        + pn.geom_line(pn.aes(color="method"))
        + pn.geom_point(shape="D", stroke=0.3, size=3.5, color="white")
        + pn.scale_color_manual(values=method_colors, labels=method_labels)
        + pn.scale_fill_manual(values=method_colors, labels=method_labels)
        + theme()
        + pn.theme(figure_size=(5, 4))
        + pn.guides(color=pn.guide_legend(ncol=2))
    )

    _p.save(figpath / "flan-t5-eh30ptability-scaling.png", dpi=300)
    _p
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Relative and absolute metrics change

    $$\Delta_{\text{prompt-tune}} = s_{\text{prompt-tune}} - s_{\text{few-shot}}$$

    $$\delta = \frac{\Delta_{\text{prompt-tune}}}{s_{\text{few-shot}}}$$
    """)
    return


@app.cell
def _(arch_colors, arch_labels, df, figpath, model_labels, shapes, theme):
    _metric_labels = {"f1": "F1", "recall": "Saagis", "precision": "Täpsus"}
    _metric_order = ["f1", "recall", "precision"]

    _df = (
        df.filter(
            pl.col("method").is_in(["5-shot", "cls-head", "prompt-tune-pretrained"]),
        )
        .with_columns(
            pl.col("total_parameters")
            .mean()
            .over(["base_model", "model_type", "architecture"]),
        )
        .pivot(
            index=["base_model", "model_type", "architecture", "total_parameters"],
            values=["test_f1", "test_recall", "test_precision"],
            on="method",
        )
        .with_columns(
            pl.coalesce(["test_f1_5-shot", "test_f1_cls-head"]).alias(
                "test_f1_baseline",
            ),
            pl.coalesce(["test_recall_5-shot", "test_recall_cls-head"]).alias(
                "test_recall_baseline",
            ),
            pl.coalesce(["test_precision_5-shot", "test_precision_cls-head"]).alias(
                "test_precision_baseline",
            ),
        )
        .with_columns(
            pl.col("test_f1_prompt-tune-pretrained")
            .sub(pl.col("test_f1_baseline"))
            .alias("pt_abs_delta_f1"),
            pl.col("test_recall_prompt-tune-pretrained")
            .sub(pl.col("test_recall_baseline"))
            .alias("pt_abs_delta_recall"),
            pl.col("test_precision_prompt-tune-pretrained")
            .sub(pl.col("test_precision_baseline"))
            .alias("pt_abs_delta_precision"),
        )
        .with_columns(
            pl.col("pt_abs_delta_f1")
            .mul(1 / pl.col("test_f1_baseline"))
            .alias("pt_rel_delta_f1"),
            pl.col("pt_abs_delta_recall")
            .mul(1 / pl.col("test_recall_baseline"))
            .alias("pt_rel_delta_recall"),
            pl.col("pt_abs_delta_precision")
            .mul(1 / pl.col("test_precision_baseline"))
            .alias("pt_rel_delta_precision"),
        )
        .select(
            "base_model",
            "model_type",
            "architecture",
            "total_parameters",
            "pt_abs_delta_f1",
            "pt_abs_delta_recall",
            "pt_abs_delta_precision",
            "pt_rel_delta_f1",
            "pt_rel_delta_recall",
            "pt_rel_delta_precision",
        )
        .unpivot(
            index=["base_model", "model_type", "architecture", "total_parameters"],
            variable_name="metric",
            value_name="value",
        )
        .with_columns(
            pl.when(pl.col("metric").str.starts_with("pt_abs_delta"))
            .then(pl.lit("pt_abs_delta"))
            .otherwise(pl.lit("pt_rel_delta"))
            .alias("measure"),
            pl.col("metric")
            .str.replace("pt_abs_delta_|pt_rel_delta_", "")
            .alias("metric"),
        )
        .pivot(
            index=[
                "base_model",
                "model_type",
                "architecture",
                "metric",
                "total_parameters",
            ],
            values="value",
            on="measure",
        )
        .with_columns(
            pl.col("metric").cast(pl.Enum(_metric_order)),
            (
                pl.col("metric").replace(_metric_labels)
                + pl.lit("\n")
                + pl.col("architecture")
            ).alias("facet_label"),
        )
    )

    _p = (
        pn.ggplot(_df)
        + pn.aes(
            x="pt_abs_delta",
            y="pt_rel_delta",
            shape="model_type",
            fill="architecture",
            size="total_parameters",
        )
        + pn.labs(
            x=r"$\Delta_{\text{PT}}$",
            y=r"$\delta_{\text{PT}}$",
            shape="",
            fill="",
            size="",
        )
        + pn.facet_grid(
            "metric ~ architecture",
            scales="free",
            labeller=lambda s: arch_labels.get(s, _metric_labels.get(s, s)),  # ty: ignore[invalid-argument-type]
        )
        + pn.scale_x_continuous(
            expand=(0.15, 0),
            labels=lambda ticks: [f"{int(100 * t)}%" for t in ticks],
        )
        + pn.scale_y_continuous(
            expand=(0.15, 0),
            labels=lambda ticks: [f"{int(100 * t)}%" for t in ticks],
        )
        + pn.scale_size_continuous(
            range=(2, 7),
            labels=lambda x: [f"{cast('float', v) / 1e9:.0f}B" for v in x],
        )
        + pn.geom_point(stroke=0.3, color="white")
        + pn.scale_color_manual(values=arch_colors, labels=arch_labels)
        + pn.scale_fill_manual(values=arch_colors, labels=arch_labels)
        + pn.scale_shape_manual(values=shapes, labels=model_labels)
        + theme()
        + pn.theme(
            panel_spacing_x=0.025,
            panel_spacing_y=0.025,
            figure_size=(8, 7),
        )
        + pn.guides(
            shape=pn.guide_legend(
                order=1,
                nrow=3,
                override_aes={"color": "black", "size": 4},
            ),
            size=pn.guide_legend(
                order=2,
                ncol=2,
                override_aes={"color": "black"},
            ),
            fill=pn.guide_legend(
                order=3,
                ncol=1,
                override_aes={"size": 4},
            ),
        )
    )

    _p.save(figpath / "relative-absolute-performance.png", dpi=300)
    _p
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Compute time scaling
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Side-by-side
    """)
    return


@app.cell
def _(colors, df, figpath, method_labels, model_labels, shapes, theme):
    _df = df.filter(
        pl.col("method").is_in(["fine-tune", "prompt-tune-pretrained"]),
        pl.col("model_type") != "t5gemma2",
    )

    _method_labels = method_labels.copy()
    _method_labels["prompt-tune-pretrained"] = "Prompt-häälestus"

    _p = (
        pn.ggplot(_df)
        + pn.aes(
            x="trainable_parameters",
            y="train_runtime",
            fill="method",
            shape="model_type",
        )
        + pn.labs(
            x="Treenitavad parameetrid",
            y="Treenimisaeg",
            fill="",
            shape="",
        )
        + pn.scale_x_log10(
            expand=(0.1, 0),
            breaks=[10**i for i in range(4, 11)],
            labels=["10K", "100K", "1M", "10M", "100M", "1B", "10B"],
        )
        + pn.scale_y_continuous(
            expand=(0.1, 0),
            labels=lambda ticks: [f"{cast('float', t) / 3600:.1f}h" for t in ticks],
        )
        + pn.geom_point(size=3.5, stroke=0.3, color="white")
        + pn.scale_color_manual(values=colors, labels=_method_labels)
        + pn.scale_fill_manual(values=colors, labels=_method_labels)
        + pn.scale_shape_manual(values=shapes, labels=model_labels)
        + theme()
        + pn.theme(figure_size=(5, 4))
        + pn.guides(
            shape=pn.guide_legend(
                order=1,
                nrow=3,
                override_aes={"color": "black", "size": 4},
            ),
            fill=pn.guide_legend(order=2, ncol=1),
        )
    )

    _p.save(figpath / "runtime-scaling-side-by-side.png", dpi=300)
    _p
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Comparison
    """)
    return


@app.cell
def _(arch_colors, arch_labels, df, figpath, model_labels, shapes, theme):
    _df = df.filter(
        pl.col("method").str.contains(r"fine-tune|prompt-tune-pretrained"),
    ).pivot(
        index=["base_model", "model_type", "architecture"],
        values=["train_runtime", "total_parameters"],
        on="method",
        aggregate_function="mean",
    )

    _max_time = _df.select("train_runtime_fine-tune").max().item()

    _p = (
        pn.ggplot(_df)
        + pn.aes(
            y="train_runtime_fine-tune",
            x="train_runtime_prompt-tune-pretrained",
            shape="model_type",
            fill="architecture",
            size="total_parameters_fine-tune",
        )
        + pn.labs(
            y="Peenhäälestus treenimisaeg",
            x="Prompt-häälestus treenimisaeg",
            shape="",
            fill="",
            size="",
        )
        + pn.scale_x_continuous(
            labels=lambda ticks: [f"{cast('float', t) / 3600:.1f}h" for t in ticks],
        )
        + pn.scale_y_continuous(
            labels=lambda ticks: [f"{cast('float', t) / 3600:.1f}h" for t in ticks],
        )
        + pn.scale_size_continuous(
            range=(2, 7),
            labels=lambda x: [f"{cast('float', v) / 1e9:.0f}B" for v in x],
        )
        + pn.coord_cartesian(xlim=(0, _max_time), ylim=(0, _max_time))
        + pn.geom_point(stroke=0.3, color="white")
        + pn.scale_fill_manual(values=arch_colors, labels=arch_labels)
        + pn.scale_shape_manual(values=shapes, labels=model_labels)
        + theme()
        + pn.theme(figure_size=(5, 4))
        + pn.guides(
            size=pn.guide_legend(ncol=1, order=1, override_aes={"color": "black"}),
            shape=pn.guide_legend(
                nrow=4,
                order=2,
                override_aes={"size": 4, "color": "black"},
            ),
            fill=pn.guide_legend(
                ncol=1,
                order=3,
                override_aes={"size": 4},
            ),
        )
    )

    _p.save(figpath / "runtime-scaling-comparison.png", dpi=300)
    _p
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Compute vs performance
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    #### Flan-T5
    """)
    return


@app.cell
def _(colors, df, figpath, method_colors, method_labels, theme):
    _method_labels = method_labels.copy()
    _method_labels["prompt-tune-pretrained"] = "Prompt-häälestus"

    _method_order = ["prompt-tune-pretrained", "fine-tune"]

    _df = df.filter(
        pl.col("method").is_in(["fine-tune", "prompt-tune-pretrained"]),
        pl.col("model_type").is_in(["t5"]),
    ).with_columns(pl.col("method").cast(pl.Enum(_method_order)))

    _p = (
        pn.ggplot(_df)
        + pn.aes(
            x="train_runtime",
            y="test_f1",
            fill="method",
        )
        + pn.labs(
            x="Treenimisaeg",
            y="F1",
            color="",
            shape="",
            fill="",
            size="",
        )
        + pn.scale_x_continuous(
            labels=lambda ticks: [f"{cast('float', t) / 3600:.1f}h" for t in ticks],
        )
        + pn.scale_y_continuous(
            breaks=[0, 0.25, 0.5, 0.75, 1.0],
            labels=["0%", "25%", "50%", "75%", "100%"],
            limits=(0, 1),
        )
        + pn.scale_size_continuous(
            range=(3, 6),
            labels=lambda x: [f"{cast('float', v) / 1e9:.0f}B" for v in x],
        )
        + pn.geom_line(
            pn.aes(group="base_model"),
            linetype="dashed",
            alpha=0.75,
            color=colors[3],
        )
        + pn.geom_point(
            pn.aes(size="total_parameters"),
            stroke=0.3,
            color="white",
            shape="D",
        )
        + pn.scale_fill_manual(values=method_colors, labels=_method_labels)
        + theme()
        + pn.theme(figure_size=(5, 4))
        + pn.guides(
            size=pn.guide_legend(ncol=2, order=1, override_aes={"color": "black"}),
            fill=pn.guide_legend(
                ncol=1,
                order=2,
                override_aes={"size": 4},
            ),
        )
    )

    _p.save(figpath / "compute-vs-performance-t5.png", dpi=300)
    _p
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    #### Qwen 3.5
    """)
    return


@app.cell
def _(colors, df, figpath, method_colors, method_labels, theme):
    _method_labels = method_labels.copy()
    _method_labels["prompt-tune-pretrained"] = "Prompt-häälestus"

    _method_order = ["prompt-tune-pretrained", "fine-tune"]

    _df = df.filter(
        pl.col("method").is_in(["fine-tune", "prompt-tune-pretrained"]),
        pl.col("model_type").is_in(["qwen3_5_text"]),
    ).with_columns(pl.col("method").cast(pl.Enum(_method_order)))

    _p = (
        pn.ggplot(_df)
        + pn.aes(
            x="train_runtime",
            y="test_f1",
            fill="method",
        )
        + pn.labs(
            x="Treenimisaeg",
            y="F1",
            color="",
            shape="",
            fill="",
            size="",
        )
        + pn.scale_x_continuous(
            labels=lambda ticks: [f"{cast('float', t) / 3600:.1f}h" for t in ticks],
        )
        + pn.scale_y_continuous(
            breaks=[0.75, 0.8, 0.85, 0.9],
            labels=["75%", "80%", "85%", "90%"],
            limits=(0.75, 0.9),
        )
        + pn.scale_size_continuous(
            range=(2, 6),
            labels=lambda x: [f"{cast('float', v) / 1e9:.0f}B" for v in x],
            guide=None,
        )
        + pn.geom_line(
            pn.aes(group="base_model"),
            linetype="dashed",
            alpha=0.75,
            color=colors[3],
        )
        + pn.geom_point(
            pn.aes(size="total_parameters"),
            stroke=0.3,
            color="white",
            shape="s",
        )
        + pn.scale_fill_manual(values=method_colors, labels=_method_labels)
        + theme()
        + pn.theme(figure_size=(5, 4))
        + pn.guides(
            fill=pn.guide_legend(order=2, override_aes={"size": 4}),
        )
    )

    _p.save(figpath / "compute-vs-performance-qwen.png", dpi=300)
    _p
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    #### mmBERT
    """)
    return


@app.cell
def _(colors, df, figpath, method_colors, method_labels, theme):
    _method_labels = method_labels.copy()
    _method_labels["prompt-tune-pretrained"] = "Prompt-häälestus"

    _method_order = ["prompt-tune-pretrained", "fine-tune"]

    _df = df.filter(
        pl.col("method").is_in(["fine-tune", "prompt-tune-pretrained"]),
        pl.col("model_type").is_in(["modernbert"]),
    ).with_columns(pl.col("method").cast(pl.Enum(_method_order)))

    _p = (
        pn.ggplot(_df)
        + pn.aes(
            x="train_runtime",
            y="test_f1",
            fill="method",
        )
        + pn.labs(
            x="Treenimisaeg",
            y="F1",
            color="",
            shape="",
            fill="",
            size="",
        )
        + pn.scale_x_continuous(
            labels=lambda ticks: [f"{cast('float', t) / 60:.1f}m" for t in ticks],
        )
        + pn.scale_y_continuous(
            breaks=[0.75, 0.8, 0.85, 0.9, 0.95],
            labels=["75%", "80%", "85%", "90%", "95%"],
            limits=(0.75, 0.95),
        )
        + pn.scale_size_continuous(
            range=(4, 6),
            labels=lambda x: [f"{cast('float', v) / 1e6:.0f}M" for v in x],
            guide=None,
        )
        + pn.geom_line(
            pn.aes(group="base_model"),
            linetype="dashed",
            alpha=0.75,
            color=colors[3],
        )
        + pn.geom_point(
            pn.aes(size="total_parameters"),
            stroke=0.3,
            color="white",
            shape="s",
        )
        + pn.scale_fill_manual(values=method_colors, labels=_method_labels)
        + theme()
        + pn.theme(figure_size=(5, 4))
        + pn.guides(
            fill=pn.guide_legend(order=2, override_aes={"size": 4}),
        )
    )

    _p.save(figpath / "compute-vs-performance-mmbert.png", dpi=300)
    _p
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    #### Three together
    """)
    return


@app.cell
def _(
    arch_labels,
    colors,
    df,
    figpath,
    method_colors,
    method_labels,
    model_labels,
    shapes,
    theme,
):
    _seconds_per_minute = 300
    _seconds_per_hour = 3600
    _method_labels = method_labels.copy()
    _method_labels["prompt-tune-pretrained"] = "Prompt-häälestus"
    _method_order = ["prompt-tune-pretrained", "fine-tune"]

    _df = df.filter(
        pl.col("method").is_in(["fine-tune", "prompt-tune-pretrained"]),
        pl.col("model_type").is_in(["modernbert", "gpt_neox", "qwen3_5_text"]),
    ).with_columns(pl.col("method").cast(pl.Enum(_method_order)))

    _p = (
        pn.ggplot(_df)
        + pn.aes(
            x="train_runtime",
            y="test_f1",
            fill="method",
            shape="architecture",
        )
        + pn.labs(
            x="Treenimisaeg",
            y="F1",
            color="",
            shape="",
            fill="",
            size="",
        )
        + pn.scale_x_continuous(
            labels=lambda ticks: [
                (f"{t:.0f}s" if t < _seconds_per_minute else f"{t / 60:.0f}m")
                if t < _seconds_per_hour
                else f"{t / 3600:.1f}h"
                for t in cast("Sequence[float]", ticks)
            ],
        )
        + pn.scale_y_continuous(
            breaks=[0, 0.25, 0.5, 0.75, 1.0],
            labels=["0%", "25%", "50%", "75%", "100%"],
            limits=(0, 1),
        )
        + pn.scale_size_continuous(
            range=(3, 6),
            labels=lambda x: [f"{cast('float', v) / 1e9:.0f}B" for v in x],
        )
        + pn.facet_wrap(
            "model_type",
            scales="free_x",
            labeller=lambda s: model_labels.get(s, s),  # ty: ignore[invalid-argument-type]
        )
        + pn.geom_line(
            pn.aes(group="base_model"),
            linetype="dashed",
            alpha=0.75,
            color=colors[3],
        )
        + pn.geom_point(pn.aes(size="total_parameters"), stroke=0.3, color="white")
        + pn.scale_fill_manual(values=method_colors, labels=_method_labels)
        + pn.scale_shape_manual(values=shapes, labels=arch_labels)
        + theme()
        + pn.theme(
            panel_spacing_x=0.025,
            panel_spacing_y=0.025,
            figure_size=(8, 3.5),
        )
        + pn.guides(
            size=pn.guide_legend(ncol=2, order=1, override_aes={"color": "black"}),
            shape=pn.guide_legend(
                ncol=2,
                order=2,
                override_aes={"color": "black", "size": 4},
            ),
            fill=pn.guide_legend(
                ncol=1,
                order=3,
                override_aes={"size": 4},
            ),
        )
    )

    _p.save(figpath / "compute-vs-performance-other.png", dpi=300)
    _p
    return


@app.cell
def _(
    arch_labels,
    colors,
    df,
    figpath,
    method_colors,
    method_labels,
    model_labels,
    shapes,
    theme,
):
    _method_labels = method_labels.copy()
    _method_labels["prompt-tune-pretrained"] = "Prompt-häälestus"

    _method_order = ["prompt-tune-pretrained", "fine-tune"]

    _df = df.filter(
        pl.col("method").is_in(["fine-tune", "prompt-tune-pretrained"]),
    ).with_columns(pl.col("method").cast(pl.Enum(_method_order)))

    _p = (
        pn.ggplot(_df)
        + pn.aes(
            x="train_runtime",
            y="test_f1",
            fill="method",
            shape="architecture",
        )
        + pn.labs(
            x="Treenimisaeg",
            y="F1",
            color="",
            shape="",
            fill="",
            size="",
        )
        + pn.scale_x_continuous(
            labels=lambda ticks: [f"{cast('float', t) / 3600:.1f}h" for t in ticks],
        )
        + pn.scale_y_continuous(
            breaks=[0, 0.25, 0.5, 0.75, 1.0],
            labels=["0%", "25%", "50%", "75%", "100%"],
            limits=(0, 1),
        )
        + pn.scale_size_continuous(
            range=(3, 6),
            labels=lambda x: [f"{cast('float', v) / 1e9:.0f}B" for v in x],
        )
        + pn.facet_wrap(
            "model_type",
            labeller=lambda s: model_labels.get(s, s),  # ty: ignore[invalid-argument-type]
        )
        + pn.geom_line(
            pn.aes(group="base_model"),
            linetype="dashed",
            alpha=0.75,
            color=colors[3],
        )
        + pn.geom_point(pn.aes(size="total_parameters"), stroke=0.3, color="white")
        + pn.scale_fill_manual(values=method_colors, labels=_method_labels)
        + pn.scale_shape_manual(values=shapes, labels=arch_labels)
        + theme()
        + pn.theme(figure_size=(8, 7))
        + pn.guides(
            size=pn.guide_legend(ncol=1, order=1, override_aes={"color": "black"}),
            shape=pn.guide_legend(
                ncol=1,
                order=2,
                override_aes={"color": "black", "size": 4},
            ),
            fill=pn.guide_legend(
                ncol=1,
                order=3,
                override_aes={"size": 4},
            ),
        )
    )

    _p.save(figpath / "compute-vs-performance-all.png", dpi=300)
    _p
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Low-resource
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Model scaling
    """)
    return


@app.cell
def _(
    arch_labels,
    df_raw,
    figpath,
    method_colors,
    method_labels,
    model_labels,
    shapes,
    theme,
):
    _method_labels = method_labels.copy()
    _method_labels["prompt-tune-pretrained"] = "Prompt-häälestus"

    _df_fewshot = pl.concat(
        [
            df_raw.filter(
                pl.col("method") == "5-shot",
            ).with_columns(pl.lit(s).alias("train_samples"))
            for s in [10.0, 100.0, 1000.0, 20000.0]
        ],
    )

    _df = pl.concat(
        [
            df_raw.filter(
                pl.col("method").is_in(
                    ["cls-head", "fine-tune", "prompt-tune-pretrained"],
                ),
            ),
            _df_fewshot,
        ],
    )

    _p = (
        pn.ggplot(_df)
        + pn.aes(
            x="total_parameters",
            y="test_f1",
            fill="method",
            shape="architecture",
        )
        + pn.labs(x="Parameetrid", y="F1", fill="", color="", shape="")
        + pn.scale_x_log10(
            expand=(0.2, 0),
            breaks=[1e6, 1e7, 1e8, 1e9, 1e10],
            labels=["1M", "10M", "100M", "1B", "10B"],
        )
        + pn.scale_y_continuous(
            breaks=[0, 0.25, 0.5, 0.75, 1.0],
            labels=["0%", "25%", "50%", "75%", "100%"],
            limits=(0, 1),
        )
        + pn.geom_line(pn.aes(color="method"))
        + pn.geom_point(stroke=0.3, size=3.5, color="white")
        + pn.scale_color_manual(values=method_colors, labels=_method_labels)
        + pn.scale_fill_manual(values=method_colors, labels=_method_labels)
        + pn.scale_shape_manual(values=shapes, labels=arch_labels)
        + pn.facet_grid(
            "model_type ~ train_samples",
            labeller=lambda s: model_labels.get(s, f"{s[:-2]} lauset"),  # ty: ignore[invalid-argument-type]
        )
        + theme()
        + pn.theme(figure_size=(8, 15))
        + pn.guides(
            color=pn.guide_legend(nrow=2),
            shape=pn.guide_legend(
                ncol=2,
                override_aes={"color": "black"},
            ),
        )
    )

    _p.save(figpath / "low-resource-params.png", dpi=300)
    _p
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Trainset scaling
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    #### All
    """)
    return


@app.cell
def _(
    arch_labels,
    df_raw,
    figpath,
    method_colors,
    method_labels,
    shapes,
    theme,
):
    _method_labels = method_labels.copy()
    _method_labels["prompt-tune-pretrained"] = "Prompt-häälestus"

    _model_labels = {
        "distilbert/distilbert-base-cased": "DistilBERT (65M)",
        "jhu-clsp/mmBERT-base": "mmBERT (307M)",
        "microsoft/deberta-v3-large": "DeBERTa (435M)",
        "google/gemma-3-4b-it": "Gemma 3 (4.3B)",
        "EleutherAI/pythia-6.9b": "GPT-NeoX (6.9B)",
        "meta-llama/Llama-3.1-8B-Instruct": "Llama (8B)",
        "Qwen/Qwen3.5-9B": "Qwen 3.5 (9B)",
        "google/flan-t5-xxl": "Flan-T5 (11B)",
    }

    _model_order = [
        "distilbert/distilbert-base-cased",
        "jhu-clsp/mmBERT-base",
        "microsoft/deberta-v3-large",
        "google/gemma-3-4b-it",
        "EleutherAI/pythia-6.9b",
        "meta-llama/Llama-3.1-8B-Instruct",
        "Qwen/Qwen3.5-9B",
        "google/flan-t5-xxl",
    ]

    _sizes = [10.0, 100.0, 1000.0, 20000.0]

    _df_fewshot = pl.concat(
        [
            df_raw.filter(
                pl.col("method") == "5-shot",
                pl.col("base_model").is_in(_model_order),
            ).with_columns(pl.lit(s).alias("train_samples"))
            for s in _sizes
        ],
    )

    _df = pl.concat(
        [
            df_raw.filter(
                pl.col("method").is_in(
                    [
                        "cls-head",
                        "fine-tune",
                        "prompt-tune-pretrained",
                    ],
                ),
                pl.col("base_model").is_in(_model_order),
            ),
            _df_fewshot,
        ],
    ).with_columns(pl.col("base_model").cast(pl.Enum(_model_order)))

    _p = (
        pn.ggplot(_df)
        + pn.aes(
            x="train_samples",
            y="test_f1",
            fill="method",
            shape="architecture",
        )
        + pn.labs(
            x="Treening laused",
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
        + pn.geom_line(pn.aes(color="method"))
        + pn.geom_point(stroke=0.3, size=3.5, color="white")
        + pn.scale_color_manual(values=method_colors, labels=_method_labels)
        + pn.scale_fill_manual(values=method_colors, labels=_method_labels)
        + pn.scale_shape_manual(values=shapes, labels=arch_labels)
        + pn.facet_wrap("base_model", labeller=lambda s: _model_labels.get(s, s))  # ty: ignore[invalid-argument-type]
        + theme()
        + pn.theme(figure_size=(7, 6))
        + pn.guides(
            color=pn.guide_legend(nrow=2),
            shape=pn.guide_legend(
                ncol=2,
                override_aes={"color": "black"},
            ),
        )
    )

    _p.save(figpath / "low-resource-trainset.png", dpi=300)
    _p
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    #### Llama
    """)
    return


@app.cell
def _(df_raw, figpath, method_colors, method_labels, theme):
    _method_labels = method_labels.copy()
    _method_labels["prompt-tune-pretrained"] = "Prompt-häälestus"

    _model_labels = {
        "distilbert/distilbert-base-cased": "DistilBERT (65M)",
        "jhu-clsp/mmBERT-base": "mmBERT (307M)",
        "microsoft/deberta-v3-large": "DeBERTa (435M)",
        "google/gemma-3-4b-it": "Gemma 3 (4.3B)",
        "EleutherAI/pythia-6.9b": "GPT-NeoX (6.9B)",
        "meta-llama/Llama-3.1-8B-Instruct": "Llama (8B)",
        "Qwen/Qwen3.5-9B": "Qwen 3.5 (9B)",
        "google/flan-t5-xxl": "Flan-T5 (11B)",
    }

    _model_order = [
        "distilbert/distilbert-base-cased",
        "jhu-clsp/mmBERT-base",
        "microsoft/deberta-v3-large",
        "google/gemma-3-4b-it",
        "EleutherAI/pythia-6.9b",
        "meta-llama/Llama-3.1-8B-Instruct",
        "Qwen/Qwen3.5-9B",
        "google/flan-t5-xxl",
    ]

    _sizes = [10.0, 100.0, 1000.0, 20000.0]

    _df_fewshot = pl.concat(
        [
            df_raw.filter(
                pl.col("method") == "5-shot",
                pl.col("base_model").is_in(_model_order),
            ).with_columns(pl.lit(s).alias("train_samples"))
            for s in _sizes
        ],
    )

    _df = (
        pl.concat(
            [
                df_raw.filter(
                    pl.col("method").is_in(
                        [
                            "cls-head",
                            "fine-tune",
                            "prompt-tune-pretrained",
                        ],
                    ),
                    pl.col("base_model").is_in(_model_order),
                ),
                _df_fewshot,
            ],
        )
        .with_columns(pl.col("base_model").cast(pl.Enum(_model_order)))
        .filter(pl.col("base_model").eq("meta-llama/Llama-3.1-8B-Instruct"))
    )

    _p = (
        pn.ggplot(_df)
        + pn.aes(
            x="train_samples",
            y="test_f1",
            fill="method",
            shape="architecture",
        )
        + pn.labs(
            x="Treening laused",
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
        + pn.geom_line(pn.aes(color="method"))
        + pn.geom_point(stroke=0.3, size=3.5, color="white", shape="s")
        + pn.scale_color_manual(values=method_colors, labels=_method_labels)
        + pn.scale_fill_manual(values=method_colors, labels=_method_labels)
        + theme()
        + pn.theme(figure_size=(5, 4))
        + pn.guides(color=pn.guide_legend(nrow=1))
    )

    _p.save(figpath / "low-resource-trainset-llama.png", dpi=300)
    _p
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    #### mmBERT
    """)
    return


@app.cell
def _(df_raw, figpath, method_colors, method_labels, theme):
    _method_labels = method_labels.copy()
    _method_labels["prompt-tune-pretrained"] = "Prompt-häälestus"

    _model_labels = {
        "distilbert/distilbert-base-cased": "DistilBERT (65M)",
        "jhu-clsp/mmBERT-base": "mmBERT (307M)",
        "microsoft/deberta-v3-large": "DeBERTa (435M)",
        "google/gemma-3-4b-it": "Gemma 3 (4.3B)",
        "EleutherAI/pythia-6.9b": "GPT-NeoX (6.9B)",
        "meta-llama/Llama-3.1-8B-Instruct": "Llama (8B)",
        "Qwen/Qwen3.5-9B": "Qwen 3.5 (9B)",
        "google/flan-t5-xxl": "Flan-T5 (11B)",
    }

    _model_order = [
        "distilbert/distilbert-base-cased",
        "jhu-clsp/mmBERT-base",
        "microsoft/deberta-v3-large",
        "google/gemma-3-4b-it",
        "EleutherAI/pythia-6.9b",
        "meta-llama/Llama-3.1-8B-Instruct",
        "Qwen/Qwen3.5-9B",
        "google/flan-t5-xxl",
    ]

    _sizes = [10.0, 100.0, 1000.0, 20000.0]

    _df_fewshot = pl.concat(
        [
            df_raw.filter(
                pl.col("method") == "5-shot",
                pl.col("base_model").is_in(_model_order),
            ).with_columns(pl.lit(s).alias("train_samples"))
            for s in _sizes
        ],
    )

    _df = (
        pl.concat(
            [
                df_raw.filter(
                    pl.col("method").is_in(
                        [
                            "cls-head",
                            "fine-tune",
                            "prompt-tune-pretrained",
                        ],
                    ),
                    pl.col("base_model").is_in(_model_order),
                ),
                _df_fewshot,
            ],
        )
        .with_columns(pl.col("base_model").cast(pl.Enum(_model_order)))
        .filter(pl.col("base_model").eq("jhu-clsp/mmBERT-base"))
    )

    _p = (
        pn.ggplot(_df)
        + pn.aes(
            x="train_samples",
            y="test_f1",
            fill="method",
            shape="architecture",
        )
        + pn.labs(
            x="Treening laused",
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
        + pn.geom_line(pn.aes(color="method"))
        + pn.geom_point(stroke=0.3, size=3.5, color="white", shape="o")
        + pn.scale_color_manual(values=method_colors, labels=_method_labels)
        + pn.scale_fill_manual(values=method_colors, labels=_method_labels)
        + theme()
        + pn.theme(figure_size=(5, 4))
        + pn.guides(color=pn.guide_legend(nrow=1))
    )

    _p.save(figpath / "low-resource-trainset-mmbert.png", dpi=300)
    _p
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Loss
    """)
    return


@app.cell
def _(collected, dataset, figpath, method_colors, method_labels, theme):
    split_labels = {
        "eval": "Testandmed",
        "train": "Treeningandmed",
    }

    method_order = [
        "prompt-tune-pretrained",
        "prompt-tune-random",
        "fine-tune",
    ]

    _df = collected.history.filter(
        pl.col("status") == "FINISHED",
        pl.col("dataset") == dataset,
        pl.col("base_model") == "google/flan-t5-xxl",
        pl.col("method").is_in(method_order),
    ).with_columns(
        pl.col("method").cast(pl.Enum(method_order)),
        pl.when(pl.col("metric") == "loss")
        .then(pl.lit("train"))
        .otherwise(pl.lit("eval"))
        .cast(pl.Enum(["train", "eval"]))
        .alias("split"),
    )

    _p = (
        pn.ggplot(_df)
        + pn.aes(x="step", y="value", color="method", fill="method")
        + pn.labs(x="Treeningsamm", y="Kadu", color="", fill="")
        + pn.facet_wrap("split", labeller=lambda s: split_labels.get(s, s))  # ty: ignore[invalid-argument-type]
        + pn.geom_line()
        + pn.geom_point(shape="D", stroke=0.3, size=2, color="white")
        + pn.scale_color_manual(values=method_colors, labels=method_labels)
        + pn.scale_fill_manual(values=method_colors, labels=method_labels)
        + theme()
        + pn.theme(figure_size=(6, 4))
        + pn.guides(color=pn.guide_legend(nrow=2))
    )

    _p.save(figpath / "flan-t5-loss.png", dpi=300)
    _p
    return


if __name__ == "__main__":
    app.run()

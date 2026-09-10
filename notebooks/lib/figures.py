"""Shared plotnine figure style."""

import plotnine as pn

SHAPES: list[str] = ["o", "s", "D", "^", "v", "<", ">", "*", "p", "h", "8", "+", "x"]
COLORS: list[str] = ["#4878CF", "#6ACC65", "#D65F5F", "#B47CC7", "#C4AD66", "#77BEDB"]

METHOD_LABELS: dict[str, str] = {
    "5-shot": "Few-shot (5)",
    "fine-tune": "Fine-tune",
    "cls-head": "Cls-head",
    "prompt-tune-random": "Prompt-tune (random)",
    "prompt-tune-pretrained": "Prompt-tune (pretrained)",
}

METHOD_COLORS: dict[str, str] = {
    "5-shot": "#C4AD66",
    "cls-head": "#8C7530",
    "fine-tune": "#4878CF",
    "prompt-tune-random": "#3E8A3A",
    "prompt-tune-pretrained": "#6ACC65",
}

METRIC_LABELS: dict[str, str] = {
    "test_f1": "F1",
    "test_precision": "Precision",
    "test_recall": "Recall",
}

MODEL_ORDER: list[str] = [
    "distilbert",
    "modernbert",
    "deberta-v2",
    "gemma3_text",
    "gpt_neox",
    "llama",
    "qwen3_5_text",
    "t5",
]

MODEL_LABELS: dict[str, str] = {
    "distilbert": "DistilBERT",
    "modernbert": "mmBERT",
    "deberta-v2": "DeBERTa",
    "gemma3_text": "Gemma 3",
    "gpt_neox": "GPT-NeoX",
    "llama": "Llama 3",
    "qwen3_5_text": "Qwen 3.5",
    "t5": "Flan-T5",
}

ARCH_ORDER: list[str] = ["encoder", "decoder", "encoder-decoder"]

ARCH_LABELS: dict[str, str] = {
    "encoder": "Encoder",
    "decoder": "Decoder",
    "encoder-decoder": "Encoder-decoder",
}

ARCH_COLORS: dict[str, str] = {
    "encoder": "#4878CF",
    "decoder": "#6ACC65",
    "encoder-decoder": "#B47CC7",
}


def theme(base_size: int = 11, base_family: str = "DejaVu Sans") -> pn.theme:
    """Build the shared minimal base theme for figures."""
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

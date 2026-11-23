"""Kev-style decision model: a frozen backbone, a LoRA adapter and a pointer head."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import TYPE_CHECKING, cast

import torch
from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
    get_peft_model_state_dict,
    load_peft_weights,
    set_peft_model_state_dict,
)
from torch import nn
from transformers import AutoModel

from eh30pt.logging import logger
from eh30pt.modeling.lora import _lora_kwargs

if TYPE_CHECKING:
    from typing import Any

    from peft import PeftModel
    from transformers import PreTrainedTokenizerFast

    from eh30pt.settings import LoRASettings

_HUB_PATTERNS = ["*.json", "*.safetensors", "*.pt", "*.txt", "*.jinja"]


class PointerHead(nn.Module):
    """Score option closing tokens against a question's final token."""

    def __init__(self, hidden_size: int, head_dim: int) -> None:
        """Build independent query and key projections of `head_dim`."""
        super().__init__()
        self.q = nn.Linear(hidden_size, head_dim)
        self.k = nn.Linear(hidden_size, head_dim)
        self.scale = 1 / math.sqrt(head_dim)
        self.temperature = 1.0

    def forward(self, decide: torch.Tensor, options: torch.Tensor) -> torch.Tensor:
        """Return one logit per option for each question."""
        return (self.k(options) @ self.q(decide).unsqueeze(-1)).squeeze(-1) * self.scale


class KevModel(nn.Module):
    """A causal backbone with a LoRA adapter and a pointer head."""

    def __init__(
        self, backbone: PeftModel, *, hidden_size: int, head_dim: int, pad_id: int
    ) -> None:
        """Attach a pointer head to a trainable backbone wrapper."""
        super().__init__()
        self.backbone = backbone
        self.head = PointerHead(hidden_size, head_dim)
        self.pad_id = pad_id

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        decide_index: torch.Tensor,
        option_index: torch.Tensor,
    ) -> torch.Tensor:
        """Return `[N, K]` option logits for a padded batch of decision rows."""
        output = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )
        hidden = output.last_hidden_state.float()
        width = hidden.shape[-1]
        decide = hidden.gather(
            1,
            decide_index.view(-1, 1, 1).expand(-1, 1, width),
        ).squeeze(1)
        options = hidden.gather(
            1,
            option_index.unsqueeze(-1).expand(-1, option_index.shape[1], width),
        )
        return self.head(decide, options)

    def trainable_parameters(self) -> list[nn.Parameter]:
        """Return the LoRA and pointer-head parameters."""
        return [parameter for parameter in self.parameters() if parameter.requires_grad]

    def warm_start(self, checkpoint: str) -> None:
        """Load a checkpoint's adapter and pointer head into this model."""
        path = _resolve_checkpoint(checkpoint)
        raw = torch.load(path / "head.pt", map_location="cpu", weights_only=True)
        meta = cast("dict[str, Any]", raw)
        head_dim = self.head.q.out_features
        if int(meta.get("head_dim", 0)) != head_dim:
            msg = (
                f"checkpoint head_dim is {meta.get('head_dim')}, "
                f"but model head_dim is {head_dim}"
            )
            raise ValueError(msg)

        weights = load_peft_weights(str(path), device="cpu")
        have = set(get_peft_model_state_dict(self.backbone))
        unexpected = sorted(set(weights) - have)
        missing = sorted(have - set(weights))
        if unexpected or missing:
            msg = (
                "adapter layout differs from the checkpoint: "
                f"{len(unexpected)} unexpected and {len(missing)} missing tensors"
            )
            raise ValueError(msg)

        set_peft_model_state_dict(self.backbone, weights)
        self.head.load_state_dict(meta["head"])
        logger.info("warm-started decision model from '%s'", path)

    def save_checkpoint(
        self,
        tokenizer: PreTrainedTokenizerFast,
        output_dir: Path,
        *,
        base_model: str,
        base_revision: str | None,
        lora_rank: int,
        temperature: float,
        training_args: dict[str, object],
    ) -> Path:
        """Write a checkpoint the llama.cpp decision converter recognises."""
        output_dir.mkdir(parents=True, exist_ok=True)
        self.backbone.save_pretrained(str(output_dir))
        head_dim = self.head.q.out_features
        meta: dict[str, object] = {
            "base": base_model,
            "base_revision": base_revision,
            "head": {
                name: tensor.detach().cpu()
                for name, tensor in self.head.state_dict().items()
            },
            "lora": lora_rank,
            "head_dim": head_dim,
            "option_isolation": False,
            "special_embeddings": False,
            "weights_dtype": "fp32",
            "weights": "lora",
            "temperature": temperature,
            "holdout": [],
        }
        torch.save(meta, output_dir / "head.pt")

        training_config = {
            "args": {**training_args, "head_dim": head_dim},
            "base_revision": base_revision,
        }
        (output_dir / "training_config.json").write_text(
            json.dumps(training_config, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        tokenizer.save_pretrained(output_dir)
        logger.info("saved decision checkpoint to '%s'", output_dir)

        return output_dir


def resolve_device(name: str) -> torch.device:
    """Resolve an accelerator name, failing when it was asked for explicitly."""
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        msg = "cuda was requested but is not available"
        raise ValueError(msg)
    if device.type == "mps" and not torch.mps.is_available():
        msg = "mps was requested but is not available"
        raise ValueError(msg)

    return device


def load_kev_model(
    model_path: str,
    tokenizer: PreTrainedTokenizerFast,
    device: torch.device,
    *,
    lora: LoRASettings,
    head_dim: int,
    init_from: str | None = None,
) -> KevModel:
    """Load a frozen backbone, wrap it in LoRA, and attach a pointer head."""
    logger.info("loading decision backbone '%s'", model_path)
    attn = "sdpa" if device.type == "cuda" else "eager"
    backbone = AutoModel.from_pretrained(
        model_path,
        dtype=torch.float32,
        attn_implementation=attn,
    )
    config = backbone.config.get_text_config()
    config.use_cache = False

    targets = _lora_kwargs(model_path)
    adapter = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,
        r=lora.rank,
        lora_alpha=lora.alpha,
        lora_dropout=lora.dropout,
        target_modules=targets.get("target_modules"),
        target_parameters=targets.get("target_parameters"),
    )
    wrapped = cast("PeftModel", get_peft_model(backbone, adapter))
    model = KevModel(
        wrapped,
        hidden_size=int(config.hidden_size),
        head_dim=head_dim,
        pad_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0,
    )
    model.to(device)

    if init_from is not None:
        model.warm_start(init_from)

    return model


def _resolve_checkpoint(run: str) -> Path:
    """Return a local checkpoint directory, downloading a Hub repo when needed."""
    local = Path(run).expanduser()
    if local.is_dir():
        return local

    from huggingface_hub import snapshot_download

    repo, _, revision = run.partition("@")
    return Path(
        snapshot_download(
            repo, revision=revision or None, allow_patterns=_HUB_PATTERNS
        ),
    )

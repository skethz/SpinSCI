"""Paper architecture and strict loading of the four reference checkpoints."""
import json
from copy import deepcopy
from pathlib import Path

import torch

from cacti.models import build_model


def normalize_config(config):
    """Accept the previous model type name in saved training recipes."""
    config = deepcopy(config)
    if config.get("model", {}).get("type") == "SpinSCIv4":
        config["model"]["type"] = "SpinSCI"
    return config


def read_config(path):
    with open(path, encoding="utf-8") as stream:
        return normalize_config(json.load(stream))


def create_model(config, ffn_impl=None):
    model_config = dict(config["model"])
    if ffn_impl is not None:
        model_config["ffn_impl"] = ffn_impl
    return build_model(model_config)


def load_reference(model, path):
    """Load ordinary weights for CR8; real CR10/50 files already contain EMA.

    Never fall back to partial loading or silently select a different weight set.
    Only load checkpoints obtained from a trusted source.
    """
    checkpoint = torch.load(Path(path), map_location="cpu", weights_only=True)
    state = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state, strict=True)
    return {k: checkpoint[k] for k in ("epoch", "ema_decay", "interrupted")
            if k in checkpoint}


def mask_normalizer(phi):
    """Paper convention: mask sum with unexposed pixels replaced by one."""
    total = phi.sum(dim=1, keepdim=True)
    return torch.where(total == 0, torch.ones_like(total), total)

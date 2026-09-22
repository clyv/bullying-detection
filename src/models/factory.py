"""Build a model from a config's ``model.name``.

Keeps ``stgcn`` reachable by name so every checkpoint trained before the adaptive
backbone existed still loads and still reproduces — the architecture comparison is
itself a result worth keeping, not a migration to finish.
"""

from __future__ import annotations

import os

MODELS = ("stgcn", "agcn")


def build_model(config, num_classes=None, in_channels=None):
    """Instantiate the model named in ``config['model']['name']`` (default stgcn)."""
    from src.models.agcn import AGCN
    from src.models.stgcn import STGCNBaseline

    model_cfg = config["model"]
    data_cfg = config["data"]
    name = str(model_cfg.get("name", "stgcn")).lower()

    kwargs = {
        "in_channels": in_channels if in_channels is not None else model_cfg["in_channels"],
        "num_classes": num_classes if num_classes is not None else model_cfg["num_classes"],
        "num_persons": data_cfg["max_persons"],
        "graph_strategy": "spatial",
        "dropout": model_cfg.get("dropout", 0.3),
    }

    if name == "stgcn":
        return STGCNBaseline(**kwargs)
    if name == "agcn":
        return AGCN(
            **kwargs,
            base_channels=model_cfg.get("base_channels", 64),
            adaptive=model_cfg.get("adaptive", True),
        )
    raise ValueError(f"unknown model {name!r}; expected one of {MODELS}")


def checkpoint_name(config, stream="joint"):
    """Per-(experiment, model, stream) checkpoint filename.

    Multi-stream training writes four models per experiment; without the stream in
    the name they overwrite each other — the same collision that once silently
    replaced a good checkpoint with a broken epoch-1 one.
    """
    model = str(config["model"].get("name", "stgcn")).lower()
    suffix = "" if stream == "joint" else f"_{stream}"
    return f"{model}{suffix}_best.pt"


def checkpoint_meta(config, stream="joint"):
    """What a checkpoint must record to be loaded back exactly as it was trained.

    The presence of ``normalize`` is itself meaningful: only checkpoints written
    after the per-axis normalization fix carry it (see ``resolve_normalize``).
    """
    return {
        "model": str(config["model"].get("name", "stgcn")).lower(),
        "normalize": config["data"].get("normalize", False),
        "stream": stream,
    }


def infer_architecture(state_dict):
    """Architecture of a state dict from its parameter names (for pre-metadata files)."""
    if any(key.startswith("blocks.") for key in state_dict):
        return "agcn"
    if any(key.startswith("layer1.") for key in state_dict):
        return "stgcn"
    raise ValueError("unrecognised checkpoint: neither stgcn nor agcn parameter names")


def resolve_normalize(state, config_normalize):
    """Normalization mode a checkpoint was trained with.

    Checkpoints that record it are taken at their word. Older ones predate the
    per-axis normalization fix, so if they were trained normalized at all it was
    with the pooled-std scale — ``"legacy"`` — and feeding them the corrected scale
    would silently shift every score they produce.
    """
    if isinstance(state, dict) and "normalize" in state:
        return state["normalize"]
    return "legacy" if config_normalize else False


def load_for_inference(checkpoint_path, config, device, num_classes=2):
    """Build whichever model a checkpoint holds and load it for inference.

    Returns ``(model, normalize)``. Architecture comes from the checkpoint rather
    than the config, so ``--checkpoint stgcn_best.pt`` keeps working after the
    config's default moved to ``agcn`` — and a fresh ``agcn_best.pt`` loads in the
    operator tools that used to hardcode ST-GCN. ``normalize`` is the mode the
    caller must feed this model, which can differ from the config's (see
    ``resolve_normalize``).
    """
    import torch

    state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = state.get("model_state_dict", state) if isinstance(state, dict) else state
    arch = state.get("model") if isinstance(state, dict) and "model" in state else None
    arch = arch or infer_architecture(state_dict)

    model_config = {**config, "model": {**config["model"], "name": arch}}
    model = build_model(model_config, num_classes=num_classes).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    normalize = resolve_normalize(state, config["data"].get("normalize", False))
    if normalize != config["data"].get("normalize", False):
        print(
            f"[checkpoint] {os.path.basename(checkpoint_path)} was trained with "
            f"normalize={normalize!r}; using that instead of the config's "
            f"{config['data'].get('normalize', False)!r}."
        )
    return model, normalize

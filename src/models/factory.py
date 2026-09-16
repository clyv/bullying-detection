"""Build a model from a config's ``model.name``.

Keeps ``stgcn`` reachable by name so every checkpoint trained before the adaptive
backbone existed still loads and still reproduces — the architecture comparison is
itself a result worth keeping, not a migration to finish.
"""

from __future__ import annotations

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

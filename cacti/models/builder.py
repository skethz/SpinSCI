from cacti.utils.registry import Registry,build_from_cfg

MODELS = Registry("models")

def build_model(cfg,default_args=None):
    cfg = dict(cfg)
    if cfg.get("type") == "SpinSCIv4":
        cfg["type"] = "SpinSCI"
    model = build_from_cfg(cfg, MODELS, default_args)
    return model

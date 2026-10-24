from collections.abc import Iterable
import inspect
import math

import torch

from cacti.utils.registry import Registry, build_from_cfg


OPTIMIZERS = Registry("optimizer")


def register_torch_optimizers():
    torch_optimizers = []
    for module_name in dir(torch.optim):
        if module_name.startswith("__"):
            continue
        optimizer = getattr(torch.optim, module_name)
        if inspect.isclass(optimizer) and issubclass(
            optimizer, torch.optim.Optimizer
        ):
            OPTIMIZERS.register_module(optimizer)
            torch_optimizers.append(module_name)
    return torch_optimizers


register_torch_optimizers()


def _muon_zeropower_newton_schulz(
    gradient,
    ns_coefficients=(3.4445, -4.7750, 2.0315),
    ns_steps=5,
    eps=1e-7,
):
    """PyTorch Muon's Newton--Schulz orthogonalization for older PyTorch."""
    if gradient.ndim != 2:
        raise ValueError("Muon only supports 2-D gradients")
    if int(ns_steps) >= 100:
        raise ValueError("Muon ns_steps must be less than 100")
    a, b, c = tuple(ns_coefficients)
    update = gradient.bfloat16()
    transposed = gradient.size(0) > gradient.size(1)
    if transposed:
        update = update.T
    update.div_(update.norm().clamp(min=float(eps)))
    for _ in range(int(ns_steps)):
        gram = update @ update.T
        gram_update = torch.addmm(gram, gram, gram, beta=b, alpha=c)
        update = torch.addmm(update, gram_update, update, beta=a)
    return update.T if transposed else update


class MuonCompat(torch.optim.Optimizer):
    """API-compatible fallback for environments without native Muon."""

    def __init__(
        self,
        params,
        lr=1e-3,
        weight_decay=0.1,
        momentum=0.95,
        nesterov=True,
        ns_coefficients=(3.4445, -4.7750, 2.0315),
        eps=1e-7,
        ns_steps=5,
        adjust_lr_fn=None,
    ):
        if float(lr) < 0 or float(momentum) < 0 or float(weight_decay) < 0:
            raise ValueError("Muon lr, momentum, and weight_decay must be non-negative")
        if adjust_lr_fn not in (None, "original", "match_rms_adamw"):
            raise ValueError("Unsupported Muon learning-rate adjustment")
        defaults = dict(
            lr=float(lr),
            weight_decay=float(weight_decay),
            momentum=float(momentum),
            nesterov=bool(nesterov),
            ns_coefficients=tuple(ns_coefficients),
            eps=float(eps),
            ns_steps=int(ns_steps),
            adjust_lr_fn=adjust_lr_fn,
        )
        super().__init__(params, defaults)
        for group in self.param_groups:
            for parameter in group["params"]:
                if parameter.ndim != 2:
                    raise ValueError("MuonCompat received a non-2-D parameter")

    @staticmethod
    def _adjusted_lr(lr, adjust_lr_fn, shape):
        rows, columns = shape[:2]
        if adjust_lr_fn is None or adjust_lr_fn == "original":
            ratio = math.sqrt(max(1.0, rows / columns))
        elif adjust_lr_fn == "match_rms_adamw":
            ratio = 0.2 * math.sqrt(max(rows, columns))
        else:
            ratio = 1.0
        return float(lr) * ratio

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr = float(group["lr"])
            momentum = float(group["momentum"])
            for parameter in group["params"]:
                gradient = parameter.grad
                if gradient is None:
                    continue
                if gradient.is_sparse or torch.is_complex(parameter):
                    raise RuntimeError("Muon does not support sparse or complex gradients")
                state = self.state[parameter]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(
                        gradient, memory_format=torch.preserve_format
                    )
                buffer = state["momentum_buffer"]
                buffer.lerp_(gradient, 1.0 - momentum)
                update = (
                    gradient.lerp(buffer, momentum)
                    if group["nesterov"]
                    else buffer
                )
                update = _muon_zeropower_newton_schulz(
                    update,
                    group["ns_coefficients"],
                    group["ns_steps"],
                    group["eps"],
                )
                adjusted_lr = self._adjusted_lr(
                    lr, group["adjust_lr_fn"], parameter.shape
                )
                parameter.mul_(1.0 - lr * float(group["weight_decay"]))
                parameter.add_(update, alpha=-adjusted_lr)
        return loss


class HybridMuonAdamW(torch.optim.Optimizer):
    """Muon for eligible 2-D weights and AdamW for every other parameter."""

    def __init__(
        self,
        named_params: Iterable,
        lr=1e-4,
        adamw_lr=None,
        weight_decay=0.0,
        adamw_weight_decay=0.0,
        momentum=0.95,
        nesterov=True,
        ns_steps=5,
        adjust_lr_fn="match_rms_adamw",
        betas=(0.9, 0.999),
        eps=1e-8,
        min_muon_elements=2,
        min_muon_dim=1,
        no_muon_keywords=("embed", "embedding", "pos_embed"),
        lr_groups=None,
    ):
        """Optionally assign initial rates to parameter-name groups.

        Each lr_groups entry contains keywords and optional lr/adamw_lr values.
        The release configs use no extra groups; tools.train applies the same
        cosine learning rate to the Muon and AdamW branches.
        """
        named_params = list(named_params)
        trainable = [
            (name, parameter)
            for name, parameter in named_params
            if parameter.requires_grad
        ]
        if not trainable:
            raise ValueError("HybridMuonAdamW received no trainable parameters")

        excluded = tuple(str(keyword).lower() for keyword in no_muon_keywords)
        min_muon_dim = int(min_muon_dim)
        if min_muon_dim < 1:
            raise ValueError("HybridMuonAdamW min_muon_dim must be at least 1")
        muon_params = []
        adamw_params = []
        lr_groups = [dict(g) for g in (lr_groups or [])]
        extra_muon = [[] for _ in lr_groups]
        extra_adamw = [[] for _ in lr_groups]
        for name, parameter in trainable:
            use_muon = (
                parameter.ndim == 2
                and parameter.numel() >= int(min_muon_elements)
                and min(parameter.shape) >= min_muon_dim
                and not any(keyword in name.lower() for keyword in excluded)
            )
            gi = next((i for i, g in enumerate(lr_groups)
                       if any(str(k).lower() in name.lower() for k in g.get("keywords", ()))), None)
            if gi is None:
                (muon_params if use_muon else adamw_params).append(parameter)
            else:
                (extra_muon[gi] if use_muon else extra_adamw[gi]).append(parameter)
        self.lr_group_counts = [sum(p.numel() for p in extra_muon[i]) + sum(p.numel() for p in extra_adamw[i]) for i in range(len(lr_groups))]

        if not muon_params:
            raise ValueError(
                "HybridMuonAdamW found no eligible 2-D hidden weights; "
                "use AdamW for this model instead"
            )

        super().__init__([parameter for _, parameter in trainable], dict(lr=lr))
        muon_class = getattr(torch.optim, "Muon", MuonCompat)
        muon_groups = [{"params": muon_params, "lr": float(lr)}]
        muon_groups += [{"params": extra_muon[i], "lr": float(g.get("lr", lr))} for i, g in enumerate(lr_groups) if extra_muon[i]]
        self.muon = muon_class(
            muon_groups,
            lr=float(lr),
            weight_decay=float(weight_decay),
            momentum=float(momentum),
            nesterov=bool(nesterov),
            ns_steps=int(ns_steps),
            adjust_lr_fn=adjust_lr_fn,
        )
        effective_adamw_lr = float(lr if adamw_lr is None else adamw_lr)
        adamw_groups = [{"params": adamw_params, "lr": effective_adamw_lr}] if adamw_params else []
        adamw_groups += [{"params": extra_adamw[i], "lr": float(g.get("adamw_lr", g.get("lr", effective_adamw_lr)))} for i, g in enumerate(lr_groups) if extra_adamw[i]]
        adamw_params = [p for grp in adamw_groups for p in grp["params"]]
        self.adamw = (
            torch.optim.AdamW(
                adamw_groups,
                lr=effective_adamw_lr,
                betas=tuple(betas),
                eps=float(eps),
                weight_decay=float(adamw_weight_decay),
            )
            if adamw_params
            else None
        )
        self.param_groups = list(self.muon.param_groups)
        if self.adamw is not None:
            self.param_groups.extend(self.adamw.param_groups)
        self.group_base_lrs = [float(pg["lr"]) for pg in self.param_groups]
        self.muon_param_count = sum(parameter.numel() for parameter in muon_params)
        self.adamw_param_count = sum(
            parameter.numel() for parameter in adamw_params
        )

    @torch.no_grad()
    def step(self, closure=None):
        loss = self.muon.step(closure)
        if self.adamw is not None:
            adamw_loss = self.adamw.step()
            if loss is None:
                loss = adamw_loss
        return loss

    def zero_grad(self, set_to_none=True):
        self.muon.zero_grad(set_to_none=set_to_none)
        if self.adamw is not None:
            self.adamw.zero_grad(set_to_none=set_to_none)

    def state_dict(self):
        return {
            "hybrid_optimizer": "HybridMuonAdamW",
            "muon": self.muon.state_dict(),
            "adamw": self.adamw.state_dict() if self.adamw is not None else None,
        }

    def load_state_dict(self, state_dict):
        if state_dict.get("hybrid_optimizer") != "HybridMuonAdamW":
            raise ValueError(
                "HybridMuonAdamW cannot load a checkpoint from a different "
                "optimizer; use --warm-start for model-only initialization"
            )
        self.muon.load_state_dict(state_dict["muon"])
        adamw_state = state_dict.get("adamw")
        if self.adamw is not None:
            if adamw_state is None:
                raise ValueError("Hybrid checkpoint is missing its AdamW state")
            self.adamw.load_state_dict(adamw_state)
        self.param_groups = list(self.muon.param_groups)
        if self.adamw is not None:
            self.param_groups.extend(self.adamw.param_groups)


def _is_no_decay_param(name, parameter, no_decay_keywords):
    if not parameter.requires_grad:
        return None
    if parameter.ndim <= 1:
        return True
    if name.endswith(".bias"):
        return True
    if any(keyword in name for keyword in no_decay_keywords):
        return True
    return False


def _build_param_groups(model, weight_decay, no_decay_keywords):
    decay_params = []
    no_decay_params = []

    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if _is_no_decay_param(name, parameter, no_decay_keywords):
            no_decay_params.append(parameter)
        else:
            decay_params.append(parameter)

    param_groups = []
    if decay_params:
        param_groups.append(
            {"params": decay_params, "weight_decay": weight_decay}
        )
    if no_decay_params:
        param_groups.append({"params": no_decay_params, "weight_decay": 0.0})
    return param_groups


def build_optimizer(cfg, default_args=None):
    cfg = dict(cfg)
    default_args = dict(default_args or {})
    model = default_args.pop("model", None)

    if cfg.get("type") == "HybridMuonAdamW":
        if model is None:
            raise ValueError("HybridMuonAdamW requires default_args['model']")
        cfg.pop("type")
        return HybridMuonAdamW(model.named_parameters(), **cfg)

    paramwise_cfg = cfg.pop("paramwise_cfg", None)
    paramwise_cfg = dict(paramwise_cfg) if paramwise_cfg is not None else None

    if paramwise_cfg is not None:
        if model is None:
            raise ValueError("paramwise_cfg requires default_args['model']")
        weight_decay = float(cfg.get("weight_decay", 0.0) or 0.0)
        no_decay_keywords = tuple(
            paramwise_cfg.get(
                "no_decay_keywords",
                (
                    "relative_position_bias_table",
                    "absolute_pos_embed",
                    "pos_embed",
                ),
            )
        )
        default_args["params"] = _build_param_groups(
            model=model,
            weight_decay=weight_decay,
            no_decay_keywords=no_decay_keywords,
        )
    elif "params" not in default_args:
        if model is None:
            raise ValueError(
                "build_optimizer requires default_args['params'] or "
                "default_args['model']"
            )
        default_args["params"] = model.parameters()

    return build_from_cfg(cfg, OPTIMIZERS, default_args)

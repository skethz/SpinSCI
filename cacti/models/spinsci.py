"""SpinSCI: mask-conditioned spatial-temporal reconstruction for video SCI.

The prior combines parallel spatial and temporal attention, MC-IRB refinement,
and an inverted residual FFN. Initialization and final data consistency operate
in the sensing domain. The optional auxiliary head is disabled by the released
final-reconstruction MSE trainer.
"""
import contextlib
from functools import lru_cache, reduce
from operator import mul

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch.nn.init import trunc_normal_

from .builder import MODELS
from cacti.utils.utils import A, At


def window_partition(x, window_size):
    """
    Args:
        x: (B, D, H, W, C)
        window_size (tuple[int]): window size (Wd, Wh, Ww)
    Returns:
        windows: (B * num_windows, Wd*Wh*Ww, C)
    """
    B, D, H, W, C = x.shape
    x = x.view(
        B,
        D // window_size[0], window_size[0],
        H // window_size[1], window_size[1],
        W // window_size[2], window_size[2],
        C,
    )
    windows = (
        x.permute(0, 1, 3, 5, 2, 4, 6, 7)
        .contiguous()
        .view(-1, reduce(mul, window_size), C)
    )
    return windows


def window_reverse(windows, window_size, B, D, H, W):
    """
    Args:
        windows: (B * num_windows, Wd*Wh*Ww, C)
        window_size: (Wd, Wh, Ww)
        B, D, H, W: target dimensions (after padding)
    Returns:
        x: (B, D, H, W, C)
    """
    x = windows.view(
        B,
        D // window_size[0],
        H // window_size[1],
        W // window_size[2],
        window_size[0], window_size[1], window_size[2],
        -1,
    )
    x = (
        x.permute(0, 1, 4, 2, 5, 3, 6, 7)
        .contiguous()
        .view(B, D, H, W, -1)
    )
    return x


@lru_cache()
def compute_mask(D, H, W, window_size, shift_size, device):
    """
    Compute attention mask for shifted windows.
    Returns: (nW, N, N) where N = Wd*Wh*Ww.
    """
    img_mask = torch.zeros((1, D, H, W, 1), device=device)
    cnt = 0
    for d in slice(-window_size[0]), slice(-window_size[0], -shift_size[0]), slice(-shift_size[0], None):
        for h in slice(-window_size[1]), slice(-window_size[1], -shift_size[1]), slice(-shift_size[1], None):
            for w in slice(-window_size[2]), slice(-window_size[2], -shift_size[2]), slice(-shift_size[2], None):
                img_mask[:, d, h, w, :] = cnt
                cnt += 1
    mask_windows = window_partition(img_mask, window_size)  # (nW, N, 1)
    mask_windows = mask_windows.squeeze(-1)                 # (nW, N)
    attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)  # (nW, N, N)
    attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
    return attn_mask


class TemporalAttention(nn.Module):
    def __init__(self, dim, num_heads, qkv_bias=False, qk_scale=None, frames=8, temporal_bias="table"):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.inner_dim = dim // 2
        assert self.inner_dim % num_heads == 0, "dim//2 must be divisible by num_heads"
        self.head_dim = self.inner_dim // num_heads
        self.scale = qk_scale or self.head_dim ** -0.5
        self.max_frames = int(frames)
        if temporal_bias != "table":
            raise ValueError("Temporal attention uses a learned relative-position table")
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros(max(2 * self.max_frames - 1, 1), num_heads)
        )
        trunc_normal_(self.relative_position_bias_table, std=0.02)

        self.qkv = nn.Linear(dim, self.inner_dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(self.inner_dim, dim)
        self.softmax = nn.Softmax(dim=-1)

    def _get_rel_pos_bias(self, n, device, dtype):
        table = self.relative_position_bias_table
        target_len = max(2 * n - 1, 1)
        if table.shape[0] != target_len:
            table_t = table.transpose(0, 1).unsqueeze(0)  # (1, nH, L)
            table_t = F.interpolate(table_t, size=target_len, mode="linear", align_corners=True)
            table = table_t.squeeze(0).transpose(0, 1)  # (target_len, nH)

        idx = torch.arange(n, device=device)
        rel_index = idx[:, None] - idx[None, :] + (n - 1)  # (n, n)
        bias = table[rel_index.reshape(-1)].reshape(n, n, self.num_heads)
        bias = bias.permute(2, 0, 1).contiguous()  # (nH, n, n)
        return bias.to(dtype=dtype)

    def forward(self, x, return_logits=False):
        """
        x: (Bpix, T, C)
        """
        Bpix, T, _ = x.shape
        qkv = (
            self.qkv(x)
            .reshape(Bpix, T, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)  # (Bpix, nH, T, T)
        attn = attn + self._get_rel_pos_bias(T, x.device, attn.dtype).unsqueeze(0)

        logits = attn
        attn = self.softmax(attn)
        out = (attn @ v).transpose(1, 2).reshape(Bpix, T, self.inner_dim)
        out = self.proj(out)
        if return_logits:
            return out, logits
        return out


class MCIRB(nn.Module):
    """
    Mask-Conditioned Ising Refinement Block (MC-IRB).

    Conditions the spin interactions on sensing information:
      - fields h and initial magnetization m0 on [x, cond];
      - couplings J with additional pairwise penalties from cond differences.

    Expected cond token features (default order):
      cond[...,0] = mask value (Phi) at that token/time
      cond[...,1] = normalized initial estimate
      cond[...,2] = Phi_s (mask sum, zero replaced by one) at that location
    """

    def __init__(self, dim, cfg):
        super().__init__()
        # Keep configuration in a plain mapping for independent module copies.
        self.cfg = dict(cfg or {})
        self.mode = str(self.cfg.get("mode", "ising")).lower()
        self.K = int(self.cfg.get("K", 3))
        self.signed = bool(self.cfg.get("signed", True))
        self.symmetric = bool(self.cfg.get("symmetric", True))
        self.row_center = bool(self.cfg.get("row_center", True))
        self.topk = int(self.cfg.get("topk", 0))  # 0 => dense

        self.cond_dim = int(self.cfg.get("cond_dim", 3))
        self.cond_in_h = bool(self.cfg.get("cond_in_h", True))
        self.cond_in_J = bool(self.cfg.get("cond_in_J", True))

        beta0 = float(self.cfg.get("beta", 1.0))
        gamma0 = float(self.cfg.get("gamma", 1.0))
        gate0 = float(self.cfg.get("gate_scale", 1.0))
        refine_ratio = float(self.cfg.get("refine_ratio", 1.0))

        # Pairwise conditioning weights
        lm0 = float(self.cfg.get("lambda_mask", 1.0))
        lr0 = float(self.cfg.get("lambda_res", 1.0))
        ls0 = float(self.cfg.get("lambda_phis", 0.0))
        learn_l = bool(self.cfg.get("learn_lambda", False))

        if bool(self.cfg.get("learn_beta", False)):
            self.beta = nn.Parameter(torch.tensor(beta0))
        else:
            self.register_buffer("beta", torch.tensor(beta0))

        if bool(self.cfg.get("learn_gamma", False)):
            self.gamma = nn.Parameter(torch.tensor(gamma0))
        else:
            self.register_buffer("gamma", torch.tensor(gamma0))

        if learn_l:
            self.lambda_mask = nn.Parameter(torch.tensor(lm0))
            self.lambda_res = nn.Parameter(torch.tensor(lr0))
            self.lambda_phis = nn.Parameter(torch.tensor(ls0))
        else:
            self.register_buffer("lambda_mask", torch.tensor(lm0))
            self.register_buffer("lambda_res", torch.tensor(lr0))
            self.register_buffer("lambda_phis", torch.tensor(ls0))

        if bool(self.cfg.get("learn_gate_scale", False)):
            self.gate_scale = nn.Parameter(torch.tensor(gate0))
        else:
            self.register_buffer("gate_scale", torch.tensor(gate0))

        in_dim = dim + (self.cond_dim if self.cond_in_h else 0)
        self.to_h = nn.Linear(in_dim, 1)
        self.to_m0 = nn.Linear(in_dim, 1)

        hidden = int(self.cfg.get("refine_hidden", 0)) or max(dim, int(dim * refine_ratio))
        self.refine = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim),
        )
        # identity init
        nn.init.zeros_(self.refine[-1].weight)
        nn.init.zeros_(self.refine[-1].bias)

    def _apply_topk(self, J):
        k = self.topk
        if k <= 0 or k >= J.size(-1):
            return J
        absJ = J.abs()
        _, idx = torch.topk(absJ, k=k, dim=-1)
        mask = torch.zeros_like(J, dtype=torch.bool)
        mask.scatter_(-1, idx, True)
        return J.masked_fill(~mask, 0.0)

    @staticmethod
    def _pairwise_l1(v):
        """v: (B, N) -> (B, N, N) with |v_i - v_j|"""
        return (v.unsqueeze(-1) - v.unsqueeze(-2)).abs()

    def forward(self, x, logits, pair_mask=None, cond=None):
        """
        Args:
            x:      (Bwin, N, C)
            logits: (Bwin, nH, N, N) pre-softmax attention logits
            pair_mask: optional (Bwin, N, N), negative entries = invalid pairs (shifted windows)
            cond: optional (Bwin, N, cond_dim)
        """
        Bwin, N, C = x.shape

        # fields & init m
        if self.cond_in_h and (cond is not None):
            xc = torch.cat([x, cond], dim=-1)
        else:
            xc = x
        h = self.to_h(xc)               # (Bwin, N, 1)
        m = torch.tanh(self.to_m0(xc))  # (Bwin, N, 1)

        if self.mode in ("standard_gate", "sigmoid_gate"):
            gate = torch.sigmoid(self.gate_scale.clamp(min=0.0) * h)
            return x + gate * self.refine(x)

        if self.mode in ("generic_gate", "direct_gate", "gate", "no_mf_gate", "conditioned_gate"):
            gate_state = torch.tanh(h + m)
            gate = torch.sigmoid(self.gate_scale.clamp(min=0.0) * gate_state)
            return x + gate * self.refine(x)

        # couplings from logits (avg over heads)
        J = logits.mean(dim=1)  # (Bwin, N, N)

        # mask invalid pairs (shifted windows)
        invalid = None
        if pair_mask is not None:
            invalid = pair_mask < 0
            J = J.masked_fill(invalid, 0.0)

        # Apply the sensing-conditioned pairwise penalty before row-centering.
        if self.cond_in_J and (cond is not None) and cond.shape[-1] >= 2:
            mask_v = cond[..., 0]  # (Bwin, N)
            estimate_v = cond[..., 1]   # (Bwin, N)
            phis_v = cond[..., 2] if cond.shape[-1] >= 3 else None

            lambda_mask = self.lambda_mask.clamp(min=0.0)
            lambda_res = self.lambda_res.clamp(min=0.0)
            lambda_phis = self.lambda_phis.clamp(min=0.0)
            J = J - lambda_mask * self._pairwise_l1(mask_v)
            J = J - lambda_res * self._pairwise_l1(estimate_v)
            if phis_v is not None:
                J = J - lambda_phis * self._pairwise_l1(phis_v)

            if invalid is not None:
                J = J.masked_fill(invalid, 0.0)

        # row-centering so couplings can be positive/negative
        if self.row_center:
            if invalid is not None:
                valid_count = (~invalid).sum(dim=-1, keepdim=True).clamp(min=1)
                row_mean = J.sum(dim=-1, keepdim=True) / valid_count
            else:
                row_mean = J.mean(dim=-1, keepdim=True)
            J = J - row_mean

        if invalid is not None:
            J = J.masked_fill(invalid, 0.0)

        if not self.signed:
            J = torch.relu(J)

        if self.symmetric:
            J = 0.5 * (J + J.transpose(-1, -2))

        # remove self-coupling
        J = J.clone()
        J.diagonal(dim1=-2, dim2=-1).zero_()

        if invalid is not None:
            J = J.masked_fill(invalid, 0.0)

        J = self._apply_topk(J)
        J = self.gamma.clamp(min=0.0) * J

        beta = self.beta.clamp(min=0.0)
        for _ in range(max(self.K, 0)):
            m = torch.tanh(beta * (h + torch.bmm(J, m)))

        gate = torch.sigmoid(self.gate_scale.clamp(min=0.0) * m)  # (Bwin, N, 1)
        return x + gate * self.refine(x)


def _rggb_channel_mask(x_rgb):
    """CFA selectors without strided in-place scatter (Inductor-safe)."""
    H, W = x_rgb.shape[-2:]
    row = torch.arange(H, device=x_rgb.device)[:, None] % 2
    col = torch.arange(W, device=x_rgb.device)[None, :] % 2
    selectors = torch.stack(((row == 0) & (col == 0), row != col,
                             (row == 1) & (col == 1)))
    return selectors.to(x_rgb.dtype).unsqueeze(0).unsqueeze(2)


def rgb_to_bayer_mosaic(x_rgb):
    """RGGB sampling: (B,3,T,H,W) -> (B,T,H,W)."""
    assert x_rgb.dim() == 5 and x_rgb.size(1) == 3
    return (x_rgb * _rggb_channel_mask(x_rgb)).sum(dim=1)


def add_bayer_mosaic_delta(x_rgb, delta):
    """Add a correction at observed RGGB sites, without in-place scatter."""
    return (x_rgb + delta.unsqueeze(1) * _rggb_channel_mask(x_rgb)).to(x_rgb.dtype)


def init_mosaic_single_step(y, phi, phi_s, eps=1e-8):
    """
    y: (B,1,H,W), phi: (B,T,H,W), phi_s: (B,1,H,W)
    Returns x0: (B,T,H,W)
    """
    x = At(y, phi)
    yb = A(x, phi)
    x = x + At((y - yb) / (phi_s + eps), phi)
    return x


def bayer_init_mosaic_single_step(y, phi, phi_s, eps=1e-8):
    """
    Apply initialization separately on the four RGGB sub-grids.
    Returns mosaic-domain x0: (B,T,H,W)
    """
    bayer = [[0, 0], [0, 1], [1, 0], [1, 1]]
    B, T, H, W = phi.shape

    y_bayer = torch.zeros(B, 1, H // 2, W // 2, 4, device=y.device, dtype=y.dtype)
    phi_bayer = torch.zeros(B, T, H // 2, W // 2, 4, device=y.device, dtype=y.dtype)
    phi_s_bayer = torch.zeros(B, 1, H // 2, W // 2, 4, device=y.device, dtype=y.dtype)

    for ib, (r, c) in enumerate(bayer):
        y_bayer[..., ib] = y[:, :, r::2, c::2]
        phi_bayer[..., ib] = phi[:, :, r::2, c::2]
        phi_s_bayer[..., ib] = phi_s[:, :, r::2, c::2]

    y_bayer = rearrange(y_bayer, "b f h w ba -> (b ba) f h w")
    phi_bayer = rearrange(phi_bayer, "b f h w ba -> (b ba) f h w")
    phi_s_bayer = rearrange(phi_s_bayer, "b f h w ba -> (b ba) f h w")

    x = At(y_bayer, phi_bayer)
    yb = A(x, phi_bayer)
    x = x + At((y_bayer - yb) / (phi_s_bayer + eps), phi_bayer)
    x = rearrange(x, "(b ba) f h w -> b f h w ba", b=B)

    x_mosaic = torch.zeros((B, T, H, W), device=y.device, dtype=y.dtype)
    for ib, (r, c) in enumerate(bayer):
        x_mosaic[:, :, r::2, c::2] = x[..., ib]
    return x_mosaic


class LowRankConv3d(nn.Module):
    """Conv3d(cin, cout, k) factorized as Conv3d(cin, rank, k) + Conv3d(rank, cout, 1).

    The k-sized conv carries no bias; the pointwise conv holds the original bias.
    """

    def __init__(self, in_channels, out_channels, kernel_size, rank,
                 stride=1, padding=0):
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_volume = kernel_size ** 3
        else:
            kernel_volume = kernel_size[0] * kernel_size[1] * kernel_size[2]
        # rank cannot exceed the matrix rank of the dense conv it replaces
        rank = min(rank, out_channels, in_channels * kernel_volume)
        self.down = nn.Conv3d(
            in_channels, rank, kernel_size,
            stride=stride, padding=padding, bias=False,
        )
        self.up = nn.Conv3d(rank, out_channels, 1, bias=True)

    def forward(self, x):
        return self.up(self.down(x))


def _grid(h, w):
    return torch.stack(
        torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij"), -1
    ).reshape(-1, 2)


def _rel_index(q_coords, k_coords, span):
    """(Nq, Nk) index into a (2*span-1)**2 relative-position table."""
    rel = q_coords[:, None, :] - k_coords[None, :, :] + (span - 1)
    return rel[..., 0] * (2 * span - 1) + rel[..., 1]


def _pad_hw(t, mult):
    _, _, h, w, _ = t.shape
    pad_b = (mult - h % mult) % mult
    pad_r = (mult - w % mult) % mult
    return F.pad(t, (0, 0, 0, pad_r, 0, pad_b)), h + pad_b, w + pad_r


class SpatialAttention(nn.Module):
    """Windowed spatial attention whose heads attend to K/V pooled by 1, 2, 4.

    Heads with pool 1 use the shifted ``window`` (7x7) window and return
    their pre-softmax logits for the Ising refiner.  A head with pool ``p``
    uses a ``pooled_window``-sized window on the p-times pooled K/V map and a
    query window of ``pooled_window * p`` native tokens, so every query still
    attends to a small fixed number of keys.
    """

    def __init__(self, dim, num_heads, window=7, pooled_window=8,
                 head_pools=(1, 1, 2, 4)):
        super().__init__()
        if len(head_pools) != num_heads or dim % num_heads:
            raise ValueError("head_pools must have num_heads entries and dim % heads == 0")
        self.dim = dim
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.window = int(window)
        self.pooled_window = int(pooled_window)
        self.native_heads = sum(1 for p in head_pools if p == 1)
        self.pooled = [int(p) for p in head_pools if p != 1]
        if self.native_heads == 0:
            raise ValueError("at least one head must use pool 1 (Ising logits)")
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)

        n = self.window
        self.native_bias = nn.Parameter(torch.zeros((2 * n - 1) ** 2, self.native_heads))
        trunc_normal_(self.native_bias, std=0.02)
        self.register_buffer("native_index", _rel_index(_grid(n, n), _grid(n, n), n))

        w = self.pooled_window
        self.pooled_bias = nn.ParameterList()
        for i, p in enumerate(self.pooled):
            bias = nn.Parameter(torch.zeros((2 * w - 1) ** 2))
            trunc_normal_(bias, std=0.02)
            self.pooled_bias.append(bias)
            self.register_buffer(
                f"pooled_index_{i}",
                _rel_index(_grid(w * p, w * p) // p, _grid(w, w), w),
            )

    def _native(self, q, k, v, shift):
        n = self.window
        ws = (1, n, n)
        B, D = q.shape[:2]
        q, Hp, Wp = _pad_hw(q, n)
        k, _, _ = _pad_hw(k, n)
        v, _, _ = _pad_hw(v, n)
        mask = None
        if shift:
            q, k, v = (torch.roll(t, shifts=(-shift, -shift), dims=(2, 3)) for t in (q, k, v))
            mask = compute_mask(D, Hp, Wp, ws, (0, shift, shift), q.device)
        nh, hd = self.native_heads, self.head_dim

        def win(t):
            t = window_partition(t, ws)  # (Bw, N, nh*hd)
            return t.view(t.shape[0], t.shape[1], nh, hd).transpose(1, 2)

        qw, kw, vw = win(q), win(k), win(v)
        Bw, _, N, _ = qw.shape
        attn = (qw * self.scale) @ kw.transpose(-2, -1)
        bias = self.native_bias[self.native_index.reshape(-1)].reshape(N, N, nh)
        attn = attn + bias.permute(2, 0, 1).unsqueeze(0)
        if mask is not None:
            nW = mask.shape[0]
            attn = (attn.view(Bw // nW, nW, nh, N, N) + mask.unsqueeze(1).unsqueeze(0)).view(Bw, nh, N, N)
        logits = attn
        out = (attn.softmax(-1) @ vw).transpose(1, 2).reshape(Bw, N, nh * hd)
        out = window_reverse(out, ws, B, D, Hp, Wp)
        if shift:
            out = torch.roll(out, shifts=(shift, shift), dims=(2, 3))
        return out, logits, mask

    def _pooled(self, q, k, v, p, bias, index):
        w = self.pooled_window
        wq = w * p
        B, D, H, W, c = q.shape
        q, Hp, Wp = _pad_hw(q, wq)
        k, _, _ = _pad_hw(k, wq)
        v, _, _ = _pad_hw(v, wq)

        def pool(t):
            t = rearrange(t, "b d h w c -> (b d) c h w")
            t = F.avg_pool2d(t, p)
            return rearrange(t, "(b d) c h w -> b d h w c", b=B)

        qw = window_partition(q, (1, wq, wq))          # (Bw, wq*wq, c)
        kw = window_partition(pool(k), (1, w, w))      # (Bw, w*w, c)
        vw = window_partition(pool(v), (1, w, w))
        attn = (qw * self.scale) @ kw.transpose(-2, -1)
        attn = attn + bias[index.reshape(-1)].reshape(wq * wq, w * w).unsqueeze(0)
        out = attn.softmax(-1) @ vw
        out = window_reverse(out, (1, wq, wq), B, D, Hp, Wp)
        return out[:, :, :H, :W, :]

    def forward(self, x, shift):
        """x: (B, D, H, W, C) normalized tokens. Returns (out, native logits, mask)."""
        H, W = x.shape[2:4]
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        cn = self.native_heads * self.head_dim
        out, logits, mask = self._native(q[..., :cn], k[..., :cn], v[..., :cn], shift)
        outs = [out[:, :, :H, :W, :]]
        for i, p in enumerate(self.pooled):
            s = slice(cn + i * self.head_dim, cn + (i + 1) * self.head_dim)
            outs.append(self._pooled(
                q[..., s], k[..., s], v[..., s], p,
                self.pooled_bias[i], getattr(self, f"pooled_index_{i}"),
            ))
        return self.proj(torch.cat(outs, dim=-1)), logits, mask


class InvertedResidualFFN(nn.Module):
    """1x1 expand -> GELU -> depthwise 3x3x3 -> GELU -> 1x1 project.

    ``impl="cudnn"`` runs the three Conv3d modules as-is.  ``impl="triton"``
    evaluates the same parameters as two Linear layers on the (B,D,H,W,C)
    token tensor plus the Triton depthwise op (cacti/models/dwconv3d_triton.py):
    identical math, no cuDNN layout transforms, memory-bound depthwise kernel.
    The state dict is the same for both.
    """

    def __init__(self, dim, ratio=2, impl="cudnn"):
        super().__init__()
        hidden = dim * ratio
        self.net = nn.Sequential(
            nn.Conv3d(dim, hidden, 1),
            nn.GELU(),
            nn.Conv3d(hidden, hidden, 3, padding=1, groups=hidden),
            nn.GELU(),
            nn.Conv3d(hidden, dim, 1),
        )
        self.impl = str(impl)
        if self.impl not in ("cudnn", "triton"):
            raise ValueError(f"impl must be 'cudnn' or 'triton', got {impl!r}")

    def forward(self, x):  # (B, D, H, W, C)
        if self.impl == "triton":
            from .dwconv3d_triton import depthwise_conv3d
            pw1, dw, pw2 = self.net[0], self.net[2], self.net[4]
            h = F.gelu(F.linear(x, pw1.weight.flatten(1), pw1.bias))
            h = rearrange(h, "b d h w c -> b c d h w").contiguous()
            h = depthwise_conv3d(h, dw.weight, dw.bias)
            h = F.gelu(rearrange(h, "b c d h w -> b d h w c"))
            return F.linear(h, pw2.weight.flatten(1), pw2.bias)
        x = rearrange(x, "b d h w c -> b c d h w")
        x = self.net(x)
        return rearrange(x, "b c d h w -> b d h w c")


class STSFormerBlock(nn.Module):
    def __init__(self, dim, num_heads, shift, window, pooled_window, head_pools,
                 ffn_ratio, frames, ising_cfg, cond_dim, ffn_impl="cudnn", temporal_bias="table"):
        super().__init__()
        self.shift = int(shift)
        self.window = int(window)
        self.norm1 = nn.LayerNorm(dim)
        self.attn = SpatialAttention(dim, num_heads, window, pooled_window, head_pools)
        self.time_attn = TemporalAttention(dim, num_heads, qkv_bias=True, frames=frames, temporal_bias=temporal_bias)
        self.norm2 = nn.LayerNorm(dim)
        self.ff = InvertedResidualFFN(dim, ffn_ratio, ffn_impl)
        cfg = dict(ising_cfg or {})
        cfg["cond_dim"] = cond_dim
        enable = bool(cfg.get("enable", True))
        self.space_ising = MCIRB(dim, cfg) if enable and cfg.get("space_enable", True) else None
        self.time_ising = MCIRB(dim, cfg) if enable and cfg.get("time_enable", True) else None

    def _refine_spatial_tokens(self, out, logits, mask, cond):
        n = self.window
        ws = (1, n, n)
        B, D, H, W, C = out.shape
        out, Hp, Wp = _pad_hw(out, n)
        cond, _, _ = _pad_hw(cond, n)
        if self.shift:
            out = torch.roll(out, shifts=(-self.shift, -self.shift), dims=(2, 3))
            cond = torch.roll(cond, shifts=(-self.shift, -self.shift), dims=(2, 3))
        windows = window_partition(out, ws)
        cond_windows = window_partition(cond, ws)
        pair_mask = None
        if mask is not None:
            pair_mask = mask.unsqueeze(0).expand(B, -1, -1, -1).reshape(-1, *mask.shape[-2:])
        windows = self.space_ising(windows, logits, pair_mask=pair_mask, cond=cond_windows)
        out = window_reverse(windows, ws, B, D, Hp, Wp)
        if self.shift:
            out = torch.roll(out, shifts=(self.shift, self.shift), dims=(2, 3))
        return out[:, :, :H, :W, :]

    def forward(self, x, cond):
        B, D, H, W, C = x.shape
        h = self.norm1(x)
        space, logits, mask = self.attn(h, self.shift)
        if self.space_ising is not None:
            space = self._refine_spatial_tokens(space, logits, mask, cond)

        h_t = rearrange(h, "b d h w c -> (b h w) d c")
        if self.time_ising is not None:
            h_t, t_logits = self.time_attn(h_t, return_logits=True)
            h_t = self.time_ising(h_t, t_logits, cond=rearrange(cond, "b d h w c -> (b h w) d c"))
        else:
            h_t = self.time_attn(h_t)
        time = rearrange(h_t, "(b h w) d c -> b d h w c", h=H, w=W)

        x = x + space + time
        return x + self.ff(self.norm2(x))


class SpinSCIPrior(nn.Module):
    def __init__(self, color_channels, dim, frames, blocks, num_heads, window,
                 pooled_window, head_pools, ffn_ratio, stem_rank, tail_rank,
                 ising_cfg, cond_dim, conditioning_mode, deep_supervision,
                 aux_after, eps, ffn_impl="cudnn", fp32_io=False,
                 mask_cond_irb=True, mask_input_stem=True, temporal_bias="table"):
        super().__init__()
        self.fp32_io = bool(fp32_io)
        # Optional mask-conditioning controls; both inputs are enabled by the release configs.
        self.mask_cond_irb = bool(mask_cond_irb)
        self.mask_input_stem = bool(mask_input_stem)
        self.color_channels = int(color_channels)
        self.cond_dim = int(cond_dim)
        if conditioning_mode != "backprojection":
            raise ValueError("MC-IRB conditioning uses the normalized initial estimate")
        self.deep_supervision = bool(deep_supervision)
        self.aux_after = tuple(int(i) for i in aux_after)
        self.eps = float(eps)
        d2, d4 = dim * 2, dim * 4

        self.stem = nn.Sequential(
            nn.Conv3d(3, dim, kernel_size=5, padding=2),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(dim, d2, kernel_size=1),
            nn.LeakyReLU(inplace=True),
            LowRankConv3d(d2, d2, 3, stem_rank, padding=1),
            nn.LeakyReLU(inplace=True),
        )
        self.down = nn.Sequential(
            LowRankConv3d(d2, d4, 3, stem_rank, stride=(1, 2, 2), padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(d4, d4, kernel_size=1),
            nn.LeakyReLU(inplace=True),
        )
        self.blocks = nn.ModuleList([
            STSFormerBlock(d4, num_heads, (window // 2) if i % 2 else 0, window,
                    pooled_window, head_pools, ffn_ratio, frames, ising_cfg, cond_dim,
                    ffn_impl, temporal_bias)
            for i in range(blocks)
        ])
        self.up = nn.Sequential(
            nn.ConvTranspose3d(d4, d2, kernel_size=(1, 3, 3), stride=(1, 2, 2),
                               padding=(0, 1, 1), output_padding=(0, 1, 1)),
            nn.LeakyReLU(inplace=True),
        )
        self.fuse = nn.Sequential(nn.Conv3d(d4, d2, kernel_size=1), nn.LeakyReLU(inplace=True))
        self.tail = nn.Sequential(
            LowRankConv3d(d2, d2, 3, tail_rank, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(d2, dim, kernel_size=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(dim, self.color_channels, kernel_size=3, padding=1),
        )
        self.aux_head = nn.Conv3d(d4, self.color_channels, kernel_size=1) if self.deep_supervision else None
        if self.aux_head is not None:
            nn.init.zeros_(self.aux_head.weight)
            nn.init.zeros_(self.aux_head.bias)
    def _build_cond(self, phi, phi_s, initial_estimate, h_out, w_out):
        """
        Build conditioning tokens at feature resolution (h_out, w_out).

        phi:      (B, T, H, W)
        phi_s:    (B, 1, H, W)
        initial_estimate: (B, T, H, W)
        returns:  (B, T, h_out, w_out, cond_dim)
        """
        B, T, H, W = phi.shape

        # Normalize each sample's initial estimate before token-grid resizing.
        scale = initial_estimate.abs().mean(dim=(1, 2, 3), keepdim=True).clamp_min(self.eps)
        cond_signal = initial_estimate / scale

        # Resize to feature map resolution
        # Preserve mask values with nearest resizing; resize continuous signals bilinearly.
        phi_ = phi.reshape(B * T, 1, H, W)
        phi_ds = F.interpolate(phi_, size=(h_out, w_out), mode="nearest").reshape(B, T, h_out, w_out)

        cond_ = cond_signal.reshape(B * T, 1, H, W)
        signal_ds = F.interpolate(
            cond_, size=(h_out, w_out), mode="bilinear", align_corners=False
        ).reshape(B, T, h_out, w_out)
        phis_ds = F.interpolate(phi_s, size=(h_out, w_out), mode="bilinear", align_corners=False)    # (B,1,h,w)

        phis_rep = phis_ds.repeat(1, T, 1, 1) # (B,T,h,w)

        if self.cond_dim == 1:
            cond = phi_ds.unsqueeze(-1)
        elif self.cond_dim == 2:
            cond = torch.stack([phi_ds, signal_ds], dim=-1)
        else:
            # default (phi, frame-specific signal, phi_s)
            cond = torch.stack([phi_ds, signal_ds, phis_rep], dim=-1)
        return cond

    def _decode_aux(self, feat, output_size):
        aux = self.aux_head(feat)
        aux = F.interpolate(
            aux, size=output_size, mode="trilinear", align_corners=False
        )
        if self.color_channels != 3:
            aux = aux.squeeze(1)
        return aux

    def forward(self, x_var, phi, residual, phi_s, return_aux=False):
        B, T, H, W = x_var.shape
        y = A(x_var, phi) + residual
        y_norm = (y / (phi_s + self.eps)).expand(-1, T, -1, -1)
        inp = torch.stack([x_var, phi if self.mask_input_stem else torch.zeros_like(phi), y_norm], dim=1)  # (B, 3, T, H, W)

        # fp32_io: the full-resolution conv stem/decoder and aux heads run in fp32
        # (autocast off); the transformer blocks keep the surrounding autocast.
        io_ctx = torch.autocast("cuda", enabled=False) if self.fp32_io else contextlib.nullcontext()
        with io_ctx:
            skip = self.stem(inp.float() if self.fp32_io else inp)
            feat = self.down(skip)
        _, _, _, Hf, Wf = feat.shape
        cond = self._build_cond(phi, phi_s, x_var, Hf, Wf)
        if not self.mask_cond_irb:  # Optional ablation: remove mask information from IRB conditioning.
            keep = torch.ones(cond.shape[-1], device=cond.device, dtype=cond.dtype)
            keep[0] = 0
            if cond.shape[-1] >= 3: keep[2] = 0
            cond = cond * keep

        out = rearrange(feat, "b c d h w -> b d h w c")
        aux_outs = []
        for i, blk in enumerate(self.blocks):
            out = blk(out, cond)
            want_aux = return_aux and self.aux_head is not None and (i + 1) in self.aux_after
            if want_aux:
                with io_ctx:
                    features = rearrange(out, "b d h w c -> b c d h w")
                    estimate = self._decode_aux(
                        features.float() if self.fp32_io else features, (T, H, W)
                    )
                    aux_outs.append(estimate)
        out = rearrange(out, "b d h w c -> b c d h w")

        with io_ctx:
            if self.fp32_io:
                out = out.float()
            out = self.up(out)
            out = self.fuse(torch.cat([out, skip], dim=1))
            out = self.tail(out)
        if self.color_channels != 3:
            out = out.squeeze(1)
        if return_aux:
            return out, aux_outs
        return out


def _preserve_initialization_rng(dim, frames, num_heads, color_channels, deep_supervision):
    """Preserve the established random stream for fixed-seed initialization.

    These initialization draws do not register additional network layers.
    """
    d2, d4 = dim * 2, dim * 4
    nn.Conv3d(1, dim, 5, padding=2)
    nn.Conv3d(dim, d2, 1)
    nn.Conv3d(d2, d2, 3, padding=1)
    nn.Conv3d(d2, d4, 3, stride=(1, 2, 2), padding=1)
    nn.Conv3d(d4, d4, 1)
    nn.ConvTranspose3d(d4, d2, (1, 3, 3), stride=(1, 2, 2),
                       padding=(0, 1, 1), output_padding=(0, 1, 1))
    nn.Conv3d(d2, d2, 3, padding=1)
    nn.Conv3d(d2, dim, 1)
    nn.Conv3d(dim, color_channels, 3, padding=1)
    nn.Linear(d4, d4 * 3)
    nn.Linear(d4, d4)
    trunc_normal_(torch.zeros(169, num_heads), std=0.02)
    for _ in range(4):
        nn.Conv3d(d2, d2, 3, padding=1)
    TemporalAttention(d4, num_heads, qkv_bias=True, frames=frames)
    if deep_supervision:
        nn.Conv3d(d4, color_channels, 1)


@MODELS.register_module
class SpinSCI(nn.Module):
    """SpinSCI-B defaults with sensing-guided initialization and correction."""
    def __init__(self, color_channels, dim=64, frames=8, blocks=9, num_heads=4,
                 window=7, pooled_window=8, head_pools=(1, 1, 2, 4), ffn_ratio=2,
                 stem_rank=32, tail_rank=32, ising_cfg=None, cond_dim=3, eps=1e-8,
                 alpha_init=1.0, learn_alpha=False, use_bayer_init=True,
                 conditioning_mode="backprojection", dc_position="post",
                 deep_supervision=True, aux_after=(3, 6), return_intermediate=False,
                 ffn_impl="cudnn", fp32_io=True, mask_cond_irb=True, mask_input_stem=True,
                 temporal_bias="table"):
        super().__init__()
        self.color_channels = int(color_channels)
        self.frames = int(frames)
        self.eps = float(eps)
        self.return_intermediate = bool(return_intermediate)
        self.use_bayer_init = bool(use_bayer_init)
        self.learn_alpha = bool(learn_alpha)
        self.dc_position = str(dc_position).lower()
        if self.dc_position not in ("pre", "post", "both"):
            raise ValueError("dc_position must be one of: pre, post, both")
        self.deep_supervision = bool(deep_supervision)
        _preserve_initialization_rng(dim, frames, num_heads, self.color_channels,
                                     self.deep_supervision)
        alpha0 = float(alpha_init)
        if self.learn_alpha:
            alpha0_t = torch.tensor(max(alpha0, 1e-6), dtype=torch.float32)
            self.alpha_raw = nn.Parameter(torch.log(torch.expm1(alpha0_t)).repeat(1))
        else:
            self.register_buffer("alpha", torch.ones(1) * alpha0)
        mc_irb_config = {
            "K": 3, "refine_hidden": 64, "signed": True, "symmetric": True,
            "row_center": True, "cond_in_h": True, "cond_in_J": True,
            "learn_lambda": True, "learn_beta": True, "learn_gamma": True,
            "learn_gate_scale": True,
        }
        mc_irb_config.update(ising_cfg or {})
        self.prior = SpinSCIPrior(
            color_channels, dim, frames, blocks, num_heads, window, pooled_window,
            head_pools, ffn_ratio, stem_rank, tail_rank, mc_irb_config, cond_dim,
            conditioning_mode, deep_supervision, aux_after, eps, ffn_impl, fp32_io,
            mask_cond_irb, mask_input_stem, temporal_bias,
        )

    def _data_consistency(self, x_var, y, phi, phi_s, alpha):
        """
        x_var: (B,T,H,W)
        y:     (B,1,H,W)
        """
        yb = A(x_var, phi)                        # (B,1,H,W)
        res = y - yb                              # (B,1,H,W)
        x_var = x_var + alpha * At(res / (phi_s + self.eps), phi)
        return x_var, res

    def forward(self, y, phi, phi_s):
        """Reconstruct one snapshot and return its final video prediction.

        Shapes are y=(B,1,H,W), phi=(B,T,H,W), and phi_s=(B,1,H,W).
        Optional auxiliary predictions precede the final output during training.
        """
        if self.color_channels == 3 and self.use_bayer_init:
            initial_estimate = bayer_init_mosaic_single_step(y, phi, phi_s, eps=self.eps)
        else:
            initial_estimate = init_mosaic_single_step(y, phi, phi_s, eps=self.eps)

        alpha = F.softplus(self.alpha_raw[0]) if self.learn_alpha else self.alpha[0]
        if self.dc_position in ("pre", "both"):
            initial_estimate, residual = self._data_consistency(
                initial_estimate, y, phi, phi_s, alpha
            )
        else:
            residual = y - A(initial_estimate, phi)

        use_aux = self.training and self.deep_supervision
        decoded = self.prior(initial_estimate, phi, residual, phi_s, return_aux=use_aux)
        if use_aux:
            output, auxiliary = decoded
        else:
            output, auxiliary = decoded, []

        if self.dc_position in ("post", "both"):
            mosaic = rgb_to_bayer_mosaic(output) if self.color_channels == 3 else output
            corrected, _ = self._data_consistency(mosaic, y, phi, phi_s, alpha)
            output = (add_bayer_mosaic_delta(output, corrected - mosaic)
                      if self.color_channels == 3 else corrected)
        return auxiliary + [output]

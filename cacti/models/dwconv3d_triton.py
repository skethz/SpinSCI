"""Depthwise 3x3x3 Conv3d (padding 1, groups=C) as a Triton custom op.

Same math as ``F.conv3d(x, w, b, padding=1, groups=C)`` for weight shape
(C, 1, 3, 3, 3): fp32 accumulation over the 27 taps, output in the input
dtype. Registered through
``torch.library`` so torch.compile treats it as an opaque op (no graph break).
"""
import torch
import triton
import triton.language as tl

_BLOCK = 256


@triton.jit
def _stencil_kernel(x_ptr, w_ptr, b_ptr, y_ptr, C, D, H, W,
                    HAS_BIAS: tl.constexpr, FLIP: tl.constexpr, BLOCK: tl.constexpr):
    """One program per ((b, c, d) plane, BLOCK-wide slice of the H*W plane).

    y[n] = sum_k w[c, k] * x[n + off(k)]  (FLIP: use w[c, 26-k]).
    """
    plane = tl.program_id(0)            # index over B*C*D
    d = plane % D
    c = (plane // D) % C
    HW = H * W
    offs = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < HW
    hi = offs // W
    wi = offs % W
    vol = (plane // D) * (D * HW)       # start of the (b, c) volume
    acc = tl.zeros([BLOCK], dtype=tl.float32)
    for k in tl.static_range(27):
        kd = k // 9
        kh = (k // 3) % 3
        kw = k % 3
        dd = d + (kd - 1)
        hh = hi + (kh - 1)
        ww = wi + (kw - 1)
        valid = mask & (dd >= 0) & (dd < D) & (hh >= 0) & (hh < H) & (ww >= 0) & (ww < W)
        xv = tl.load(x_ptr + vol + dd * HW + hh * W + ww, mask=valid, other=0.0)
        if FLIP:
            wv = tl.load(w_ptr + c * 27 + (26 - k))
        else:
            wv = tl.load(w_ptr + c * 27 + k)
        acc += xv.to(tl.float32) * wv.to(tl.float32)
    if HAS_BIAS:
        acc += tl.load(b_ptr + c).to(tl.float32)
    tl.store(y_ptr + vol + d * HW + offs, acc.to(y_ptr.dtype.element_ty), mask=mask)


@triton.jit
def _wgrad_kernel(x_ptr, dy_ptr, part_ptr, C, D, H, W, NBLK, BLOCK: tl.constexpr):
    """Partial weight gradients, one program per ((b, c, d) plane, BLOCK slice):
    part[program, k] = sum over the slice of dy[p] * x[p + off(k)].  Reduced over
    (b, d, slices) per channel by the caller."""
    plane = tl.program_id(0)
    blk = tl.program_id(1)
    d = plane % D
    HW = H * W
    offs = blk * BLOCK + tl.arange(0, BLOCK)
    mask = offs < HW
    hi = offs // W
    wi = offs % W
    vol = (plane // D) * (D * HW)
    dyv = tl.load(dy_ptr + vol + d * HW + offs, mask=mask, other=0.0).to(tl.float32)
    base = part_ptr + (plane * NBLK + blk) * 27
    for k in tl.static_range(27):
        kd = k // 9
        kh = (k // 3) % 3
        kw = k % 3
        dd = d + (kd - 1)
        hh = hi + (kh - 1)
        ww = wi + (kw - 1)
        valid = mask & (dd >= 0) & (dd < D) & (hh >= 0) & (hh < H) & (ww >= 0) & (ww < W)
        xv = tl.load(x_ptr + vol + dd * HW + hh * W + ww, mask=valid, other=0.0).to(tl.float32)
        tl.store(base + k, tl.sum(xv * dyv, axis=0))


def _stencil(x, w, b, flip):
    x = x.contiguous()
    y = torch.empty_like(x)
    B, C, D, H, W = x.shape
    grid = (B * C * D, triton.cdiv(H * W, _BLOCK))
    has_bias = b is not None
    with torch.cuda.device(x.device):
        _stencil_kernel[grid](x, w.contiguous(), b if has_bias else w, y, C, D, H, W,
                              HAS_BIAS=has_bias, FLIP=flip, BLOCK=_BLOCK)
    return y


@torch.library.custom_op("spinsci::dwconv3d", mutates_args=())
def dwconv3d(x: torch.Tensor, w: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return _stencil(x, w, b, flip=False)


@dwconv3d.register_fake
def _(x, w, b):
    return torch.empty_like(x)


@torch.library.custom_op("spinsci::dwconv3d_backward", mutates_args=())
def dwconv3d_backward(dy: torch.Tensor, x: torch.Tensor, w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    dy = dy.contiguous()
    x = x.contiguous()
    B, C, D, H, W = x.shape
    dx = _stencil(dy, w, None, flip=True)
    nblk = triton.cdiv(H * W, _BLOCK)
    part = torch.empty(B * C * D * nblk, 27, dtype=torch.float32, device=x.device)
    with torch.cuda.device(x.device):
        _wgrad_kernel[(B * C * D, nblk)](x, dy, part, C, D, H, W, nblk, BLOCK=_BLOCK)
    dw = part.view(B, C, D * nblk, 27).sum(dim=(0, 2))
    db = dy.sum(dim=(0, 2, 3, 4), dtype=torch.float32)
    return dx, dw.view_as(w).to(w.dtype), db.to(w.dtype)


@dwconv3d_backward.register_fake
def _(dy, x, w):
    return torch.empty_like(x), torch.empty_like(w), torch.empty(w.shape[0], dtype=w.dtype, device=w.device)


def _setup_context(ctx, inputs, output):
    x, w, b = inputs
    ctx.save_for_backward(x, w)


def _backward(ctx, dy):
    x, w = ctx.saved_tensors
    dx, dw, db = torch.ops.spinsci.dwconv3d_backward(dy, x, w)
    return dx, dw, db


torch.library.register_autograd("spinsci::dwconv3d", _backward, setup_context=_setup_context)


def depthwise_conv3d(x, weight, bias):
    """Drop-in for F.conv3d(x, weight, bias, padding=1, groups=C) with weight (C,1,3,3,3).

    Mirrors autocast conv semantics: under bf16 autocast the weight and bias are
    cast to the input dtype before the op. Gradients pass through these casts
    to the FP32 parameters.
    """
    w = weight.reshape(weight.shape[0], 27).to(x.dtype)
    b = bias.to(x.dtype)
    return torch.ops.spinsci.dwconv3d(x, w, b)

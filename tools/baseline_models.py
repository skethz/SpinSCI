"""Adapters for the official baseline implementations profiled in Tables 4-6.

The upstream code stays in its own checkout (see `UPSTREAM`). Several upstream
repositories ship a package named `cacti`, so one process loads exactly one
baseline and never imports this repository's `cacti` package.
"""
import hashlib
import importlib
import subprocess
import sys
from pathlib import Path

import torch

UPSTREAM = {
    'pnp': ('https://github.com/ucaswangls/cacti', '4085c9923a9aa3e3448daaf555ce67af94dcba39'),
    'efficientsci': ('https://github.com/ucaswangls/EfficientSCI', 'f123d46bbdcbb6b63c055f44fc339118a1855014'),
    'stformer': ('https://github.com/ucaswangls/STFormer', '59630253f779a3eaba38dd43201b3a82705bf99e'),
    'mambasci': ('https://github.com/PAN083/MambaSCI', '98266e1f627e234fd5ca85afd42f18360abe476a'),
    'hisvit': ('https://github.com/pwangcs/HiSViT', 'a9e6f06e4c91f3e8f0f149a42480cea1954f8eca'),
}
CONFIGS = {  # official test configurations, relative to each checkout
    ('pnp', 1): 'configs/PnP-FastDVD/fastdvd.py',
    ('pnp', 3): 'configs/PnP-FastDVD/fastdvd_color_mid_color.py',
    ('efficientsci', 1): 'configs/EfficientSCI/efficientsci_base.py',
    ('efficientsci', 3): 'configs/EfficientSCI/efficientsci_base_mid_color.py',
    ('stformer', 1): 'configs/STFormer/stformer_base.py',
    ('stformer', 3): 'configs/STFormer/stformer_base_mid_color.py',
    ('mambasci', 3): 'configs/Mamba/mamba_dp.py',
}
BAYER = [[0, 0], [0, 1], [1, 0], [1, 1]]
HISVIT_PATCH = 'patches/hisvit_pooled_padding.patch'


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def revision(root):
    try:
        head = subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip()
        dirty = subprocess.check_output(['git', '-C', str(root), 'status', '--porcelain', '--untracked-files=no'],
                                        text=True).strip()
        return {'commit': head, 'modified_tracked_files': dirty.splitlines()}
    except (OSError, subprocess.CalledProcessError):
        return {'commit': None, 'modified_tracked_files': None}


def _import_upstream(root):
    # The upstream `cacti` directories have no __init__.py, so this repository's regular `cacti`
    # package would shadow them. Drop this repository (and the working directory) from the path.
    root = str(Path(root).resolve())
    here = str(Path(__file__).resolve().parents[1])
    for name in [n for n in sys.modules if n == 'cacti' or n.startswith('cacti.') or n == 'model' or n.startswith('model.')]:
        del sys.modules[name]
    sys.path[:] = [root] + [p for p in sys.path if p not in ('', '.', here) and str(Path(p).resolve()) != here]
    importlib.invalidate_caches()


def _state_dict(path):
    state = torch.load(path, map_location='cpu', weights_only=False)
    for key in ('state_dict', 'model_state_dict', 'model'):
        if isinstance(state, dict) and key in state and isinstance(state[key], dict):
            state = state[key]
            break
    for prefix in ('module.', '_orig_mod.'):
        if state and all(k.startswith(prefix) for k in state):
            state = {k[len(prefix):]: v for k, v in state.items()}
    return state


def _official_model(method, root, channels):
    from cacti.utils.config import Config
    cfg = Config.fromfile(str(Path(root) / CONFIGS[(method, channels)]))
    spec = dict(cfg.get('model') or {'type': 'MambaSCI'})  # the MambaSCI configuration leaves the type implicit
    module = importlib.import_module('cacti.models')
    cls = getattr(module, spec.pop('type'))
    return cls(**spec), cfg


class Baseline:
    """`forward(*inputs)` maps GPU tensors to the reconstruction. `prepare` turns the
    measurement, masks and mask sum into those inputs (identity except HiSViT9,
    which is timed from its prepared network input as in the paper)."""

    def __init__(self, method, root, checkpoint, channels, size, device):
        if (method, channels) not in CONFIGS and method != 'hisvit':
            raise ValueError(f'{method} has no {channels}-channel configuration')
        self.method, self.channels, self.size = method, channels, tuple(size)
        _import_upstream(root)
        self.meta = {'upstream': UPSTREAM[method][0], 'pinned_commit': UPSTREAM[method][1],
                     'checkout': revision(root), 'checkpoint_sha256': sha256(checkpoint)}
        if method == 'hisvit':
            self._build_hisvit(root, checkpoint, device)
        elif method == 'pnp':
            self._build_pnp(root, checkpoint, device)
        else:
            self.model, _ = _official_model(method, root, channels)
            result = self.model.load_state_dict(_state_dict(checkpoint), strict=True)
            self.meta.update(config=CONFIGS[(method, channels)], load=str(result))
            self.model = self.model.to(device).eval()
        self.meta['parameters_total'] = sum(p.numel() for p in self.model.parameters())

    def _build_hisvit(self, root, checkpoint, device):
        block = (Path(root) / 'model/block.py').read_text()
        self.meta['pooled_padding_patch'] = 'max_q_split_size' in block
        if self.size[0] % 128 or self.size[1] % 128:
            if not self.meta['pooled_padding_patch']:
                raise RuntimeError(f'HiSViT at {self.size} needs {HISVIT_PATCH} applied to the checkout')
        from model.arch import HiSViT
        self.model = HiSViT(dim=[128, 256, 128], frames=8, size=list(self.size), color_ch=self.channels, blocks=9)
        own = self.model.state_dict()
        state = _state_dict(checkpoint)
        # Attention masks are buffers built for the input size; they are not learned. Keep the ones built here.
        kept = {k: v for k, v in state.items() if not ('attn_mask' in k and k in own and own[k].shape != v.shape)}
        result = self.model.load_state_dict(kept, strict=False)
        rebuilt = sorted(set(state) - set(kept))
        if result.unexpected_keys or set(result.missing_keys) - set(rebuilt):
            raise RuntimeError(f'HiSViT checkpoint mismatch: {result}')
        self.meta.update(config='HiSViT(dim=[128, 256, 128], frames=8, blocks=9)', rebuilt_size_buffers=len(rebuilt))
        self.model = self.model.to(device).eval()

    def _build_pnp(self, root, checkpoint, device):
        self.model, cfg = _official_model('pnp', root, self.channels)
        self.model.load_state_dict(_state_dict(checkpoint), strict=True)
        self.model = self.model.to(device).eval()
        from cacti.models.admm_denoise import ADMM_denoise
        from cacti.models.gap_denoise import GAP_denoise
        self.step = ADMM_denoise if cfg.denoise_method == 'ADMM' else GAP_denoise
        self.sigmas = [torch.tensor(s, dtype=torch.float32, device=device) for s in cfg.sigma_list]
        self.iterations = list(cfg.iter_list)
        self.kwargs = dict(color_denoiser=bool(cfg.get('color_denoiser', False)), bayer=BAYER,
                           use_cv2_demosaic=bool(cfg.get('use_cv2_demosaic', False))) if self.channels == 3 else {}
        self.meta.update(config=CONFIGS[('pnp', self.channels)], method=cfg.denoise_method,
                         iterations=self.iterations, options={k: v for k, v in self.kwargs.items() if k != 'bayer'})

    def prepare(self, meas, mask, mask_s):
        if self.method != 'hisvit':
            return [meas, mask, mask_s]
        meas_re = meas / mask_s
        x = meas_re + mask * meas_re
        frames = mask.shape[1]
        return [torch.cat((x.unsqueeze(1), meas_re.repeat(1, frames, 1, 1).unsqueeze(1)), dim=1)]

    def forward(self, *inputs):
        if self.method == 'pnp':
            return self._pnp(*inputs)
        output = self.model(*inputs)
        return output[-1] if isinstance(output, (list, tuple)) else output

    def _pnp(self, meas, mask, mask_s):
        from cacti.utils.utils import At
        if self.channels == 3:  # official colour path: the four CFA phases form a batch of half-size problems
            meas, mask, mask_s = [torch.cat([t[:, :, b[0]::2, b[1]::2] for b in BAYER], 0) for t in (meas, mask, mask_s)]
        x = At(meas, mask)
        aux = torch.zeros_like(x) if self.step.__name__ == 'ADMM_denoise' else torch.zeros_like(meas)
        for sigma, count in zip(self.sigmas, self.iterations):
            for _ in range(count):
                x, aux = self.step(self.model, x, meas, aux, mask, mask_s, sigma, **self.kwargs)
        if self.channels == 1:
            return x
        mosaic = x.new_empty((1, x.shape[1], 2 * x.shape[2], 2 * x.shape[3]))
        for index, b in enumerate(BAYER):
            mosaic[0, :, b[0]::2, b[1]::2] = x[index]
        return mosaic

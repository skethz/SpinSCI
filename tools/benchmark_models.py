"""Shared model loading for SpinSCI/DU benchmarks; upstream DU stays external."""
import hashlib
import importlib.util
from pathlib import Path

import torch

_DU_CLASSES = {}
from cacti.release import create_model, load_reference, read_config


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def build(args, ffn='triton'):
    if args.model == 'spinsci':
        config = read_config(args.config)
        model = create_model(config, ffn)
        metadata = load_reference(model, args.checkpoint)
    else:
        source = Path(args.du_root) / 'cacti/models/DU.py'
        source_key = str(source.resolve())
        if source_key not in _DU_CLASSES:
            spec = importlib.util.spec_from_file_location('benchmark_external_du', source)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            _DU_CLASSES[source_key] = module.DU
        config = {'model': {'dim': 48, 'color_dim': args.channels, 'stage': 9, 'skip': 3}}
        model = _DU_CLASSES[source_key](**config['model'])
        state = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
        key = 'bare_state_dict'
        for candidate in ('model_state_dict', 'state_dict'):
            if candidate in state:
                state, key = state[candidate], candidate
                break
        for prefix in ('module.', '_orig_mod.'):
            if state and all(k.startswith(prefix) for k in state):
                state = {k[len(prefix):]: v for k, v in state.items()}
        model.load_state_dict(state, strict=True)
        metadata = {'selected_state': key, 'du_source_sha256': sha256(source)}
    return model.eval(), config, metadata


def call(model, name, inputs):
    output = model(inputs[0], (inputs[1], inputs[2])) if name == 'du' else model(*inputs)
    return output[-1] if isinstance(output, (list, tuple)) else output

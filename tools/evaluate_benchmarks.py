"""Fixed-checkpoint reconstruction with explicit data layouts and CACTI metrics.

Supports an external official DU checkout. Floating predictions are evaluated
before clipping; only exported PNGs are clipped and rounded. No training occurs.
"""
import argparse
from contextlib import nullcontext
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import sys
import time
import traceback

import h5py
import numpy as np
from PIL import Image
import scipy.io as sio
import torch
try:
    from .benchmark_models import build, call, sha256
    from .snapshot_timing import check_interference
except ImportError:
    from benchmark_models import build, call, sha256
    from snapshot_timing import check_interference
from cacti.utils.metrics import compare_psnr, compare_ssim


def write(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    tmp.replace(path)


def load_sequence(path, channels, layout, real):
    key = 'meas' if real else 'orig'
    try:
        data = sio.loadmat(path, variable_names=[key])[key]
        if real or channels == 1:
            if data.ndim == 2:
                data = data[..., None]
            data = data.transpose(2, 0, 1)
        else:
            data = data.transpose(3, 2, 0, 1)
            if layout == 'mid_rot90':
                # Match CACTI's v7.3 layout before its rotation and vertical flip.
                data = data.transpose(0, 1, 3, 2)
    except NotImplementedError:
        with h5py.File(path) as stream:
            data = np.array(stream[key])
        if channels == 3 and layout == 'large_hdf5':
            data = data.transpose(0, 1, 3, 2)
    if channels == 3 and layout == 'mid_rot90':
        # SpinSCI MatlabBayerData: rotate 90 degrees, then flip the vertical axis.
        data = np.flip(np.rot90(data, axes=(-2, -1)), axis=-2)
    # Gray CACTI inputs start from MATLAB doubles; color loaders normalize float32.
    return data / 255. if channels == 1 else data.astype(np.float32) / 255.


def mosaic(rgb):
    # rgb = T,C,H,W; exactly the RGGB selectors in the SpinSCI evaluator.
    raw = rgb[:, 1].copy()
    raw[:, 0::2, 0::2] = rgb[:, 0, 0::2, 0::2]
    raw[:, 1::2, 1::2] = rgb[:, 2, 1::2, 1::2]
    return raw


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', choices=['spinsci', 'du'], required=True)
    p.add_argument('--du-root')
    p.add_argument('--config')
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--channels', type=int, choices=[1,3], required=True)
    p.add_argument('--data-root', required=True)
    p.add_argument('--mask', required=True)
    p.add_argument('--frames', type=int, default=8)
    p.add_argument('--layout', choices=['gray', 'mid_rot90', 'mid_native', 'large_hdf5'], required=True)
    p.add_argument('--precision', choices=['fp32','bf16'], default='fp32')
    p.add_argument('--compile', action='store_true')
    p.add_argument('--real', action='store_true')
    p.add_argument('--max-snapshots', type=int)
    p.add_argument('--save-raw', action='store_true')
    p.add_argument('--no-images', action='store_true')
    p.add_argument('--expected-gpu-uuid')
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    (a.output / 'png').mkdir(exist_ok=True)
    rec = {'status':'running', 'model':a.model, 'command':sys.argv, 'checkpoint_sha256':sha256(a.checkpoint),
           'mask_sha256':sha256(a.mask), 'evaluator_sha256':sha256(__file__),
           'adapter_sha256':sha256(Path(__file__).with_name('benchmark_models.py')),
           'layout':a.layout, 'orientation':('rot90_then_vertical_flip' if a.layout == 'mid_rot90' else a.layout), 'precision':a.precision, 'compiled':a.compile,
           'frames_per_snapshot':a.frames, 'real':a.real,
           'metric':'CACTI PSNR/SSIM, unclipped and unrounded, RGGB mosaic for color, 0-255 scale',
           'aggregation':'frame mean within sequence; equally weighted sequence mean',
           'export':'round(clip(prediction,0,1)*255), fixed intensity scale',
           'started_utc':datetime.now(timezone.utc).isoformat(), 'scenes':{}}
    if a.real and a.model == 'du':
        rec['checkpoint_scope'] = 'Official simulated-gray CR8 checkpoint, zero-shot real reconstruction; no real-trained DU weights supplied'
    write(a.output / 'metrics.json', rec)
    try:
        assert torch.cuda.device_count() == 1
        props = torch.cuda.get_device_properties(0)
        uuid = 'GPU-' + str(props.uuid).removeprefix('GPU-')
        if a.expected_gpu_uuid:
            assert a.expected_gpu_uuid == uuid
        torch.manual_seed(42)
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = True
        model, config, meta = build(a)
        model = model.cuda()
        rec.update({'model_config':config,'checkpoint_metadata':meta,
                    'parameters':sum(p.numel() for p in model.parameters()),
                    'gpu':props.name,'gpu_uuid':uuid,'torch':str(torch.__version__),'cuda':torch.version.cuda})
        eager_model = model
        checked_compilation = False
        if a.compile:
            model = torch.compile(model, backend='inductor', mode='default')
        mask = sio.loadmat(a.mask)['mask'][..., :a.frames].transpose(2,0,1).astype(np.float32)
        assert mask.shape[0] == a.frames
        frames, height, width = mask.shape
        total = mask.sum(0); total[total == 0] = 1
        phi = torch.from_numpy(mask[None]).cuda()
        phis = torch.from_numpy(total[None,None]).cuda()
        rec['input_hw'] = [height, width]
        rec['mask_array_sha256'] = __import__('hashlib').sha256(mask.astype(np.uint8).tobytes()).hexdigest()
        others = check_interference('before evaluation')
        for path in sorted(Path(a.data_root).glob('*.mat')):
            if path.name.startswith('._'):
                continue
            data = load_sequence(path, a.channels, a.layout, a.real)
            assert data.shape[-2:] == (height, width), (path, data.shape, mask.shape)
            steps = len(data) if a.real else len(data) // frames
            if a.max_snapshots:
                steps = min(steps, a.max_snapshots)
            name = path.stem.replace('_cacti','').replace('_bayer','')
            name = re.sub(r'\d+$','', name) if not a.real else name.split('_')[1]
            metrics, times, extrema = [], [], []
            for step in range(steps):
                if a.real:
                    measurement = data[step] * (frames / 2)
                else:
                    target = data[step*frames:(step+1)*frames]
                    truth = mosaic(target) if a.channels == 3 else target
                    measurement = (truth * mask).sum(0)
                y = torch.from_numpy(np.ascontiguousarray(measurement[None,None], dtype=np.float32)).cuda()
                context = torch.autocast('cuda', dtype=torch.bfloat16) if a.precision == 'bf16' else nullcontext()
                torch.cuda.synchronize()
                started = time.perf_counter()
                with torch.no_grad(), context:
                    if a.compile and not checked_compilation:
                        reference = call(eager_model, a.model, (y,phi,phis)).float().cpu()
                    prediction = call(model, a.model, (y,phi,phis))
                    if a.compile and not checked_compilation:
                        candidate = prediction.float().cpu()
                        error = (candidate-reference).square().mean().sqrt().item()
                        relative = error / max(reference.square().mean().sqrt().item(), 1e-12)
                        rec['compiled_vs_eager_first_snapshot'] = {'rms':error,'relative_rms':relative}
                        if relative >= 0.02:
                            raise RuntimeError(f'Compiled/eager mismatch: relative RMS {relative}')
                        del reference, candidate
                        checked_compilation = True
                torch.cuda.synchronize()
                times.append(time.perf_counter()-started)
                predicted = prediction[0].float().cpu().numpy()
                del prediction, y
                assert np.isfinite(predicted).all()
                if a.channels == 3:
                    predicted = predicted.transpose(1,0,2,3)
                    estimates = mosaic(predicted)
                else:
                    estimates = predicted
                extrema.append([float(predicted.min()),float(predicted.max())])
                if a.save_raw:
                    np.save(a.output/f'{name}_snapshot_{step+1}.npy',predicted)
                for f in range(frames):
                    number = step*frames+f+1
                    if not a.real:
                        metrics.append({'frame':number,
                            'psnr':float(compare_psnr(truth[f]*255,estimates[f]*255)),
                            'ssim':float(compare_ssim(truth[f]*255,estimates[f]*255))})
                    if not a.no_images:
                        picture = predicted[f].transpose(1,2,0) if a.channels == 3 else predicted[f]
                        Image.fromarray(np.rint(np.clip(picture,0,1)*255).astype(np.uint8)).save(
                            a.output/'png'/f'{name}_{"du9" if a.model == "du" else "spinsci"}_{number}.png')
                print('SNAPSHOT', name, step+1, '/', steps, 'seconds', round(times[-1],3), flush=True)
            record = {'data_file':str(path),'data_sha256':sha256(path),'frames':steps*frames,
                      'snapshots':steps,'forward_seconds':times,'prediction_ranges':extrema,
                      'per_frame':metrics}
            if metrics:
                record.update({key:float(np.mean([m[key] for m in metrics])) for key in ['psnr','ssim']})
            rec['scenes'][name] = record
            write(a.output/'metrics.json',rec)
            print('SEQUENCE',name,{k:v for k,v in record.items() if k in ['frames','psnr','ssim']},flush=True)
            del data
        if not a.real:
            rec.update({key+'_mean':float(np.mean([s[key] for s in rec['scenes'].values()])) for key in ['psnr','ssim']})
        rec['other_gpu_processes'] = sorted(set(others + check_interference('after evaluation')))
        rec.update({'status':'complete','finished_utc':datetime.now(timezone.utc).isoformat(),
                    'total_frames':sum(s['frames'] for s in rec['scenes'].values())})
        write(a.output/'metrics.json',rec)
        print('COMPLETE',{k:v for k,v in rec.items() if k in ['psnr_mean','ssim_mean','total_frames']},flush=True)
    except Exception:
        rec.update({'status':'failed','traceback':traceback.format_exc()})
        write(a.output/'metrics.json',rec)
        raise


if __name__ == '__main__':
    main()

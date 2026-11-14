"""Reproduce the baseline Params, FLOPs, latency and peak memory of Tables 4-6.

Each run loads one official baseline from its own checkout (`--root`) and the
official checkpoint, and uses the same synthetic inputs, timing and memory
conventions as `tools.profile_benchmarks`: FP32 eager, batch 1, eight frames,
no_grad, GPU-resident inputs, CUDA-synchronized wall clock, median of the
timed calls after warm-up, peak allocated memory including weights and inputs.
Large-scale latency and memory cover one whole snapshot (24 sequential
272 x 320 tiles of a 1088 x 1920 input, assembled on the GPU).

FLOPs count one THOP module multiply-accumulate as one FLOP. PnP-FastDVDNet is
counted over its complete iterative reconstruction with PyTorch's operator-level
counter (MACs = FLOPs / 2). Large-scale FLOPs use a full 1080 x 1920 forward,
except HiSViT9 and PnP-FastDVDNet, which are counted over the 24 tiles.
Run from the repository root with `python -m tools.profile_baselines`.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback

import torch

try:
    from .baseline_models import Baseline, sha256
    from .snapshot_timing import FRAME, PADDED, TILE, TILES, measure, tiled_snapshot
except ImportError:
    from baseline_models import Baseline, sha256
    from snapshot_timing import FRAME, PADDED, TILE, TILES, measure, tiled_snapshot

SIZES = {'gray256': (256, 256), 'color512': (512, 512)}
DEFAULT_COUNTS = {  # (warm-ups, timed calls) of the paper protocol
    'gray256': (3, 40), 'color512': (3, 20), 'large24': (3, 10)}


def write(path, record):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(record, indent=2) + '\n')
    tmp.replace(path)


def inputs_at(height, width, frames=8):
    phi = (torch.rand(1, frames, height, width, device='cuda') < 0.5).float()
    return [torch.rand(1, 1, height, width, device='cuda') * 4, phi, phi.sum(1, keepdim=True).clamp(min=1)]


def gpu_state():
    query = ['nvidia-smi', '--query-gpu=index,uuid,memory.used,utilization.gpu,temperature.gpu,power.draw,clocks.sm',
             '--format=csv']
    apps = ['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,used_memory', '--format=csv']
    return subprocess.check_output(query, text=True) + subprocess.check_output(apps, text=True)


def operator_macs(fn):
    from torch.utils.flop_counter import FlopCounterMode
    counter = FlopCounterMode(display=False)
    with counter:
        fn()
    return counter.get_total_flops() / 2


def thop_macs(model, inputs):
    from thop import profile
    macs, _ = profile(model, inputs=tuple(inputs), verbose=False)
    return macs


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--method', choices=['pnp', 'efficientsci', 'stformer', 'mambasci', 'hisvit'], required=True)
    p.add_argument('--root', type=Path, required=True, help='Official checkout of the method (see tools/baseline_models.py)')
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--case', choices=['gray256', 'color512', 'large24'], required=True)
    p.add_argument('--mode', choices=['costs', 'runtime'], required=True)
    p.add_argument('--warmups', type=int)
    p.add_argument('--runs', type=int)
    p.add_argument('--expected-gpu-uuid')
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    warmups, runs = DEFAULT_COUNTS[a.case] if a.method != 'pnp' else (1, 3)
    a.warmups = warmups if a.warmups is None else a.warmups
    a.runs = runs if a.runs is None else a.runs
    if a.runs < 1 or a.warmups < 0:
        p.error('Invalid run/warmup counts')
    channels = 1 if a.case == 'gray256' else 3
    here = Path(__file__)
    record = {'status': 'running', 'method': a.method, 'case': a.case, 'mode': a.mode,
              'started_utc': datetime.now(timezone.utc).isoformat(), 'command': sys.argv,
              'profiler_sha256': sha256(here), 'adapter_sha256': sha256(here.with_name('baseline_models.py')),
              'timing_sha256': sha256(here.with_name('snapshot_timing.py'))}
    write(a.output, record)
    try:
        assert torch.cuda.device_count() == 1, 'Expose exactly the requested GPU using CUDA_VISIBLE_DEVICES'
        props = torch.cuda.get_device_properties(0)
        uuid = 'GPU-' + str(props.uuid).removeprefix('GPU-')
        if a.expected_gpu_uuid:
            assert uuid == a.expected_gpu_uuid, (uuid, a.expected_gpu_uuid)
        torch.manual_seed(0)
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = True
        large = a.case == 'large24'
        if a.mode == 'costs' and large and a.method not in ('hisvit', 'pnp'):
            size = FRAME
        else:
            size = TILE if large else SIZES[a.case]
        baseline = Baseline(a.method, a.root, a.checkpoint, channels, size, 'cuda')
        record.update({'gpu': props.name, 'gpu_uuid': uuid, 'torch': str(torch.__version__), 'cuda': torch.version.cuda,
                       'python': sys.version, 'baseline': baseline.meta, 'batch_size': 1, 'frames': 8,
                       'precision': 'fp32', 'compiled': False, 'tf32_matmul': False, 'tf32_cudnn': True,
                       'cpu_affinity': sorted(os.sched_getaffinity(0)), 'load_average': os.getloadavg(),
                       'gpu_before': gpu_state(), 'input_hw': list(size)})
        print('LOADED', a.method, a.case, baseline.meta['parameters_total'], flush=True)
        with torch.no_grad():
            inputs = baseline.prepare(*inputs_at(*size))
            if a.mode == 'costs':
                if a.method == 'pnp':
                    macs = operator_macs(lambda: baseline.forward(*inputs))
                    counter = 'torch.utils.flop_counter over the complete iterative reconstruction, MACs = FLOPs / 2'
                else:
                    macs = thop_macs(baseline.model, inputs)
                    counter = 'THOP module multiply-accumulates'
                scale = TILES if large and size == TILE else 1
                record.update({'gmacs_per_call': macs / 1e9, 'calls': scale, 'table_gflops': scale * macs / 1e9,
                               'count_convention': '1 MAC = 1 table FLOP; ' + counter})
            else:
                record.update({'warmups': a.warmups, 'runs': a.runs,
                               'timing_scope': 'complete reconstruction from GPU-resident inputs; loading and I/O excluded'})
                record['single_forward'] = measure(lambda: baseline.forward(*inputs), a.warmups, a.runs)
                record['table_latency_ms'] = record['single_forward']['latency_ms_median']
                record['table_peak_memory_mib'] = record['single_forward']['peak_allocated_mib']
                record['memory_scope'] = 'Full frame'
                if large:
                    # The table reports the whole snapshot; the single tile above is kept for reference.
                    del inputs
                    torch.cuda.empty_cache()
                    padded = baseline.prepare(*inputs_at(*PADDED))
                    snapshot = tiled_snapshot(baseline.forward, padded)
                    record['whole_snapshot'] = measure(snapshot, a.warmups, a.runs)
                    record['table_latency_ms'] = record['whole_snapshot']['latency_ms_median']
                    record['table_peak_memory_mib'] = record['whole_snapshot']['peak_allocated_mib']
                    record['memory_scope'] = 'Whole snapshot: 24 tile forwards and assembly, padded input resident'
        record.update({'parameters_total': baseline.meta['parameters_total'], 'gpu_after': gpu_state(),
                       'status': 'complete', 'finished_utc': datetime.now(timezone.utc).isoformat()})
        write(a.output, record)
        print('RESULT', json.dumps({k: record.get(k) for k in ['method', 'case', 'mode', 'status', 'parameters_total',
              'table_gflops', 'table_latency_ms', 'table_peak_memory_mib']}), flush=True)
    except Exception:
        record.update({'status': 'failed', 'traceback': traceback.format_exc()})
        write(a.output, record)
        raise


if __name__ == '__main__':
    main()

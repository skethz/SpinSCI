"""Reproduce table operation counts and GPU timings with a shared model adapter.

THOP's module MAC count is reported separately from exact parameter totals.
Runtime profiling uses no_grad, resident inputs, synchronized wall-clock timing,
five warmups and median of 20 runs. Large latency executes and stitches 24 tiles;
its table memory is the standalone tile peak. FLOPs use an actual full-frame
forward, not an area estimate. No dataset is required.
"""
import argparse
from contextlib import nullcontext
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
import traceback

import torch
try:
    from .benchmark_models import build, call, sha256
except ImportError:
    from benchmark_models import build, call, sha256


def write(path, record):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(record, indent=2) + '\n')
    tmp.replace(path)


def measure(fn, warmups, runs):
    for _ in range(warmups):
        output = fn()
        del output
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    samples = []
    for _ in range(runs):
        torch.cuda.synchronize()
        started = time.perf_counter()
        output = fn()
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - started) * 1000)
        del output
    return {'latency_ms_samples': samples, 'latency_ms_median': statistics.median(samples),
            'peak_allocated_mib': torch.cuda.max_memory_allocated() / 2**20,
            'peak_reserved_mib': torch.cuda.max_memory_reserved() / 2**20}


def inputs_at(height, width):
    phi = (torch.rand(1, 8, height, width, device='cuda') < 0.5).float()
    return (torch.rand(1, 1, height, width, device='cuda') * 4,
            phi, phi.sum(1, keepdim=True).clamp(min=1))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', choices=['spinsci', 'du'], required=True)
    p.add_argument('--config')
    p.add_argument('--du-root')
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--case', choices=['gray256', 'color512', 'large24'], required=True)
    p.add_argument('--mode', choices=['costs', 'runtime'], required=True)
    p.add_argument('--precision', choices=['fp32', 'bf16'], default='bf16')
    p.add_argument('--compile', action='store_true')
    p.add_argument('--warmups', type=int, default=5)
    p.add_argument('--runs', type=int, default=20)
    p.add_argument('--expected-gpu-uuid')
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.runs < 1 or a.warmups < 0:
        p.error('Invalid run/warmup counts')
    a.channels = 1 if a.case == 'gray256' else 3
    record = {'status': 'running', 'model': a.model, 'case': a.case, 'mode': a.mode,
              'started_utc': datetime.now(timezone.utc).isoformat(), 'command': sys.argv,
              'profiler_sha256': sha256(__file__),
              'adapter_sha256': sha256(Path(__file__).with_name('benchmark_models.py'))}
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
        model, config, metadata = build(a, 'cudnn' if a.mode == 'costs' else 'triton')
        model = model.cuda()
        record.update({'gpu': props.name, 'gpu_uuid': uuid, 'torch': str(torch.__version__),
                       'cuda': torch.version.cuda, 'python': sys.version, 'config': config,
                       'checkpoint_metadata': metadata, 'checkpoint_sha256': sha256(a.checkpoint),
                       'parameters_total': sum(p.numel() for p in model.parameters()),
                       'batch_size': 1, 'frames': 8, 'tf32_matmul': False, 'tf32_cudnn': True,
                       'gpu_before': subprocess.check_output(['nvidia-smi', '-i', uuid,
                           '--query-gpu=name,memory.used,utilization.gpu,temperature.gpu,power.draw,clocks.sm',
                           '--format=csv'], text=True)})
        height, width = {'gray256': (256, 256), 'color512': (512, 512),
                         'large24': (1080, 1920) if a.mode == 'costs' else (272, 320)}[a.case]
        inputs = inputs_at(height, width)
        record['input_hw'] = [height, width]
        print('LOADED', a.model, a.case, record['parameters_total'], flush=True)
        if a.mode == 'costs':
            from thop import profile
            executed = set()
            def mark(module, _inputs):
                executed.update(id(p) for p in module.parameters(recurse=False))
            handles = [m.register_forward_pre_hook(mark) for m in model.modules()]
            thop_inputs = (inputs[0], (inputs[1], inputs[2])) if a.model == 'du' else inputs
            with torch.no_grad():
                macs, thop_parameters = profile(model, inputs=thop_inputs, verbose=False)
            for handle in handles:
                handle.remove()
            torch.cuda.synchronize()
            unexecuted = {n: p.numel() for n, p in model.named_parameters() if id(p) not in executed}
            record.update({'thop_gmacs': macs / 1e9, 'thop_parameters': int(thop_parameters),
                           'parameters_executed_owners': record['parameters_total'] - sum(unexecuted.values()),
                           'unexecuted_owner_parameters': unexecuted,
                           'count_convention': '1 THOP MAC = 1 table FLOP; module hooks omit unregistered functional operations',
                           'ffn': 'cuDNN depthwise convolution so THOP sees the equivalent SpinSCI operation'})
        else:
            record.update({'precision': a.precision, 'compiled': a.compile, 'warmups': a.warmups, 'runs': a.runs,
                           'timing_scope': 'complete forward, resident GPU inputs; compilation, transfer and I/O excluded'})
            ctx = torch.autocast('cuda', dtype=torch.bfloat16) if a.precision == 'bf16' else nullcontext()
            with torch.no_grad(), ctx:
                if a.compile:
                    reference = call(model, a.model, inputs).float().cpu()
                    model = torch.compile(model, backend='inductor', mode='default')
                    started = time.perf_counter()
                    output = call(model, a.model, inputs)
                    torch.cuda.synchronize()
                    record['first_compiled_call_seconds'] = time.perf_counter() - started
                    candidate = output.float().cpu()
                    assert torch.isfinite(candidate).all()
                    difference = candidate - reference
                    rms = difference.square().mean().sqrt().item()
                    relative = rms / max(reference.square().mean().sqrt().item(), 1e-12)
                    record['compiled_vs_eager'] = {'rms': rms, 'relative_rms': relative,
                                                   'max_abs': difference.abs().max().item()}
                    if relative >= 0.02:
                        raise RuntimeError(f'Compiled/eager mismatch: relative RMS {relative}; do not use timings')
                    del output, candidate, reference, difference
                record['single_forward'] = measure(lambda: call(model, a.model, inputs), a.warmups, a.runs)
                record['table_latency_ms'] = record['single_forward']['latency_ms_median']
                record['table_peak_memory_mib'] = record['single_forward']['peak_allocated_mib']
                if a.case == 'large24':
                    tiles = [inputs_at(272, 320) for _ in range(24)]
                    def snapshot():
                        assembled = inputs[0].new_empty((1, 3, 8, 1088, 1920))
                        for index, tile in enumerate(tiles):
                            row, col = divmod(index, 6)
                            prediction = call(model, a.model, tile)
                            assembled[..., row*272:(row+1)*272, col*320:(col+1)*320] = prediction
                            del prediction
                        return assembled[..., :1080, :1920]
                    record['whole_snapshot'] = measure(snapshot, a.warmups, a.runs)
                    record['table_latency_ms'] = record['whole_snapshot']['latency_ms_median']
                    record['memory_scope'] = 'Standalone tile; full 24-tile assembly peak stored separately'
                else:
                    record['memory_scope'] = 'Full frame'
        record.update({'status': 'complete', 'finished_utc': datetime.now(timezone.utc).isoformat()})
        write(a.output, record)
        print('RESULT', json.dumps({k: v for k, v in record.items() if k in
              ['model','case','mode','status','parameters_total','thop_gmacs','thop_parameters',
               'table_latency_ms','table_peak_memory_mib','compiled_vs_eager']}), flush=True)
    except Exception:
        record.update({'status': 'failed', 'traceback': traceback.format_exc()})
        write(a.output, record)
        raise


if __name__ == '__main__':
    main()

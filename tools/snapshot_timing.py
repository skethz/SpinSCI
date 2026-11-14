"""Timing helpers shared by the SpinSCI/DU and baseline profilers.

This module imports no model code, so either profiler can use it next to an
external checkout. Latency is a CUDA-synchronized wall clock around one complete
call; peak memory is the largest allocation during the timed calls, including
weights and resident inputs.

Other jobs on the same GPU or CPU change both numbers. `check_interference`
prints a warning and flags such jobs before and after every timed run.
"""
import os
import statistics
import subprocess
import sys
import time

import torch

TILE = (272, 320)
PADDED = (1088, 1920)
FRAME = (1080, 1920)
TILES = (PADDED[0] // TILE[0]) * (PADDED[1] // TILE[1])


INTERFERENCE_WARNING = (
    'WARNING: latency and peak memory depend on everything else running on this machine. '
    'If a value differs from the paper, check whether other jobs are using this GPU or '
    'loading the CPU (nvidia-smi, top), then measure again on an idle node.')
_warned = False


def check_interference(stage):
    """Print the warning above once per process, then flag other compute
    processes on the current GPU and a busy CPU. Returns the other processes."""
    global _warned
    if not _warned:
        print(INTERFERENCE_WARNING, file=sys.stderr, flush=True)
        _warned = True
    uuid = ''
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(torch.cuda.current_device())
        uuid = str(getattr(props, 'uuid', '')).removeprefix('GPU-')
    try:
        apps = subprocess.run(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,used_memory',
                               '--format=csv,noheader'], capture_output=True, text=True, timeout=30).stdout
    except (OSError, subprocess.SubprocessError):
        apps = ''
    others = []
    for line in apps.splitlines():
        fields = [field.strip() for field in line.split(',')]
        if len(fields) == 3 and fields[1] != str(os.getpid()) and (
                not uuid or fields[0].removeprefix('GPU-') == uuid):
            others.append(line.strip())
    if others:
        print(f'WARNING ({stage}): other processes are using this GPU: ' + '; '.join(others),
              file=sys.stderr, flush=True)
    try:
        load, cpus = os.getloadavg()[0], os.cpu_count() or 1
        if load > cpus / 2:
            print(f'WARNING ({stage}): CPU load average {load:.1f} on {cpus} logical CPUs',
                  file=sys.stderr, flush=True)
    except (AttributeError, OSError):
        pass
    return others


def measure(fn, warmups, runs):
    others = check_interference('before timing')
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
    result = {'latency_ms_samples': samples, 'latency_ms_median': statistics.median(samples),
              'peak_allocated_mib': torch.cuda.max_memory_allocated() / 2**20,
              'peak_reserved_mib': torch.cuda.max_memory_reserved() / 2**20}
    result['other_gpu_processes'] = sorted(set(others + check_interference('after timing')))
    return result


def tiled_snapshot(forward, inputs):
    """Return a call that reconstructs one large-scale snapshot.

    `inputs` are GPU-resident tensors whose last two dimensions span the
    1088 x 1920 padded frame. The call runs `forward` on the 24 non-overlapping
    272 x 320 tiles in sequence, writes each tile prediction into one GPU
    buffer, and returns the 1080 x 1920 crop.
    """
    height, width = TILE

    def run():
        assembled = None
        for row in range(PADDED[0] // height):
            for col in range(PADDED[1] // width):
                rows, cols = slice(row * height, (row + 1) * height), slice(col * width, (col + 1) * width)
                prediction = forward(*[x[..., rows, cols].contiguous() for x in inputs])
                if assembled is None:
                    assembled = prediction.new_empty(prediction.shape[:-2] + PADDED)
                assembled[..., rows, cols] = prediction
                del prediction
        return assembled[..., :FRAME[0], :FRAME[1]]
    return run

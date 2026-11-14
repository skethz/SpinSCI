"""Timing helpers shared by the SpinSCI/DU and baseline profilers.

This module imports no model code, so either profiler can use it next to an
external checkout. Latency is a CUDA-synchronized wall clock around one complete
call; peak memory is the largest allocation during the timed calls, including
weights and resident inputs.
"""
import statistics
import time

import torch

TILE = (272, 320)
PADDED = (1088, 1920)
FRAME = (1080, 1920)
TILES = (PADDED[0] // TILE[0]) * (PADDED[1] // TILE[1])


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

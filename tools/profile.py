"""Batch-one BF16 compiled latency and memory; no dataset is needed."""
import argparse
import json
import statistics
import time
from pathlib import Path

import torch

from cacti.release import create_model, load_reference, mask_normalizer, read_config
try:
    from .snapshot_timing import check_interference
except ImportError:
    from snapshot_timing import check_interference


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("A CUDA GPU with BF16 support is required")
    if args.runs < 1 or args.warmups < 0:
        raise ValueError("Invalid repetition counts")
    config = read_config(args.config)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = True
    model = create_model(config, "triton").eval()
    load_reference(model, args.checkpoint)
    parameters_total = sum(p.numel() for p in model.parameters())
    inference_parameters = sum(p.numel() for name, p in model.named_parameters()
                               if not name.startswith("prior.aux_head."))
    model = model.cuda()
    torch.manual_seed(0)
    phi = (torch.rand(1, config["model"]["frames"], args.height, args.width, device="cuda") < .5).float()
    y = torch.rand(1, 1, args.height, args.width, device="cuda") * 4
    phis = mask_normalizer(phi)
    samples = []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        reference = model(y, phi, phis)[-1].float().cpu()
        model = torch.compile(model, backend="inductor", mode="default")
        candidate = model(y, phi, phis)[-1].float().cpu()
        difference = candidate - reference
        relative_rms = (difference.square().mean().sqrt().item()
                        / max(reference.square().mean().sqrt().item(), 1e-12))
        if not torch.isfinite(candidate).all() or relative_rms >= .02:
            raise RuntimeError("Compiled/eager mismatch; do not use these timings")
        del reference, candidate, difference
        others = check_interference("before timing")
        for _ in range(args.warmups):
            model(y, phi, phis)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        for _ in range(args.runs):
            torch.cuda.synchronize()
            started = time.perf_counter()
            result = model(y, phi, phis)
            torch.cuda.synchronize()
            samples.append((time.perf_counter()-started)*1000)
            del result
        others = sorted(set(others + check_interference("after timing")))
    result = {"gpu": torch.cuda.get_device_name(), "torch": torch.__version__, "cuda": torch.version.cuda,
              "input_hw": [args.height, args.width], "frames": config["model"]["frames"],
              "batch_size": 1, "precision": "BF16 with FP32 input/output", "compiled": True,
              "warmups": args.warmups, "runs": args.runs, "latency_ms": samples,
              "compiled_vs_eager_relative_rms": relative_rms,
              "median_ms": statistics.median(samples),
              "peak_allocated_mib": torch.cuda.max_memory_allocated()/2**20,
              "other_gpu_processes": others,
              "parameters_total": parameters_total,
              "inference_parameters": inference_parameters,
              "parameter_convention": "Table counts include all parameters; inference_parameters excludes the auxiliary head",
              "scope": "One complete forward with resident inputs; excludes compilation, transfer, and I/O"}
    path = Path(args.output); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

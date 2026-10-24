"""Train the paper recipe on external DAVIS2017 data (no implicit resume)."""
import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader

from cacti.data import Davis2017, load_mask
from cacti.models.spinsci import rgb_to_bayer_mosaic
from cacti.release import create_model, load_reference, mask_normalizer, normalize_config, read_config
from cacti.training import GlobalBatchSampler, epoch_lr, reconstruction_loss
from cacti.utils.optim_builder import build_optimizer
from cacti.utils.utils import A


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config")
    parser.add_argument("--davis-root", required=True)
    parser.add_argument("--mask", required=True)
    parser.add_argument("--mask-key", default="mask")
    parser.add_argument("--mask-layout", choices=["HWT", "THW"], default="HWT")
    parser.add_argument("--resolution", default="480p")
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--ffn-impl", choices=["triton", "cudnn"], default=None)
    parser.add_argument("--batch-scope", choices=["per_gpu", "global"])
    start = parser.add_mutually_exclusive_group()
    start.add_argument("--resume", help="An epoch-boundary checkpoint from this trainer")
    start.add_argument("--warm-start", help="Explicit model-only initialization; starts a new schedule")
    return parser.parse_args()


def main():
    args = parse_args()
    config = read_config(args.config)
    recipe = config["training"]
    if args.batch_scope:
        recipe["batch_scope"] = args.batch_scope
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if args.device == "cuda":
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise RuntimeError("CUDA BF16 is required; use --device cpu only for debugging")
        torch.cuda.set_device(local_rank)
    device = torch.device(args.device, local_rank if args.device == "cuda" else None)
    if world > 1:
        dist.init_process_group(backend="nccl" if device.type == "cuda" else "gloo")
    seed = recipe["seed"]
    random.seed(seed+rank); np.random.seed(seed+rank); torch.manual_seed(seed)
    ffn = args.ffn_impl or ("triton" if device.type == "cuda" else "cudnn")
    model = create_model(config, ffn).to(device)
    # Final-reconstruction MSE does not use the optional auxiliary head.
    # Keep its checkpoint keys, but freeze it so DDP has no unused parameters.
    model.deep_supervision = False
    if model.prior.aux_head is not None:
        model.prior.aux_head.requires_grad_(False)
    optimizer = build_optimizer(config["optimizer"], {"model": model})
    first_epoch = 0
    if args.warm_start:
        load_reference(model, args.warm_start)
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=True)
        if normalize_config(checkpoint.get("recipe") or {}) != config:
            raise ValueError("Resume recipe differs; use --warm-start for an intentional new run")
        if checkpoint.get("world_size") != world:
            raise ValueError("Resume with the original world size to preserve batch semantics")
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        optimizer.load_state_dict(checkpoint["optim_state_dict"])
        first_epoch = int(checkpoint["completed_epoch"])+1
    if world > 1:
        model = DistributedDataParallel(model, device_ids=[local_rank] if device.type == "cuda" else None)
    data = Davis2017(args.davis_root, frames=config["model"]["frames"],
                    crop=recipe["crop_size"], color=config["model"]["color_channels"] == 3,
                    resolution=args.resolution)
    global_batch = recipe["batch_size"] * (world if recipe["batch_scope"] == "per_gpu" else 1)
    sampler = GlobalBatchSampler(len(data), global_batch, rank, world, seed)
    generator = torch.Generator()
    loader = DataLoader(data, batch_sampler=sampler, num_workers=args.workers,
                        pin_memory=device.type == "cuda", generator=generator)
    phi = load_mask(args.mask, config["model"]["frames"], args.mask_key,
                    args.mask_layout, recipe["crop_size"]).unsqueeze(0).to(device)
    normalizer = mask_normalizer(phi)
    output = Path(args.output)
    if rank == 0:
        output.mkdir(parents=True, exist_ok=True)
        (output/"recipe.json").write_text(json.dumps(config, indent=2)+"\n")
        print(json.dumps({"global_batch": global_batch, "world_size": world,
                          "training_windows": len(data), "loss": "final MSE only",
                          "device": str(device), "ffn": ffn}), flush=True)
    for epoch in range(first_epoch, recipe["epochs"]):
        sampler.set_epoch(epoch)
        epoch_seed = seed + epoch*world + rank
        generator.manual_seed(epoch_seed)
        if args.workers == 0:
            # In-process augmentations need the same epoch seed after resume.
            random.seed(epoch_seed)
            np.random.seed(epoch_seed % 2**32)
            torch.manual_seed(epoch_seed)
        lr = epoch_lr(epoch, recipe["epochs"], recipe["lr"], recipe["min_lr"])
        for group in optimizer.param_groups:
            group["lr"] = lr
        model.train()
        total = torch.zeros((), device=device)
        for target, weights in loader:
            target, weights = target.to(device), weights.to(device)
            mask = phi.expand(target.shape[0], -1, -1, -1)
            phis = normalizer.expand(target.shape[0], -1, -1, -1)
            # Sensing, loss, and gradients accumulate in FP32.
            mosaic = rgb_to_bayer_mosaic(target) if target.ndim == 5 else target
            measurement = A(mosaic, mask)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                prediction = model(measurement, mask, phis)[-1]
            loss_sum = reconstruction_loss(prediction, target, weights)
            # DDP averages gradients. This factor yields an exact global mean,
            # including global batches not divisible by the number of GPUs.
            (loss_sum * world/global_batch).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), recipe["grad_clip_norm"], error_if_nonfinite=True)
            optimizer.step()
            total += loss_sum.detach()
        if world > 1:
            dist.all_reduce(total)
        if rank == 0:
            core = model.module if world > 1 else model
            state = {"model_state_dict": core.state_dict(), "optim_state_dict": optimizer.state_dict(),
                     "completed_epoch": epoch, "recipe": config, "world_size": world}
            temporary = output/"latest.tmp"
            torch.save(state, temporary)
            temporary.replace(output/"latest.pth")
            print(json.dumps({"epoch": epoch+1, "lr": lr,
                              "mse": total.item()/(len(sampler)*global_batch)}), flush=True)
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

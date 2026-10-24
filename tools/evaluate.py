"""Evaluate external gray/RGB sequences or reconstruct real measurements."""
import argparse
import json
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch

from cacti.data import load_mask, read_array
from cacti.release import create_model, load_reference, read_config
from cacti.utils.metrics import compare_psnr, compare_ssim


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input", nargs="+", required=True, help="One .mat/.npy/.npz file per sequence")
    parser.add_argument("--key", default="orig")
    parser.add_argument("--layout", choices=["HWT", "HWCT", "THW", "CTHW"], default="HWT")
    parser.add_argument("--value-scale", type=float, default=255.0)
    parser.add_argument("--mask", required=True)
    parser.add_argument("--mask-key", default="mask")
    parser.add_argument("--mask-layout", choices=["HWT", "THW"], default="HWT")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--precision", choices=["fp32", "bf16"], default="fp32")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--ffn-impl", choices=["triton", "cudnn"], default=None)
    parser.add_argument("--real", action="store_true", help="Input contains measurements, not ground truth")
    parser.add_argument("--measurement-scale", type=float, default=1.0,
                        help="For CACTI real data use CR/2 after division by --value-scale")
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def to_array(values, layout):
    if layout == "HWT":
        if values.ndim == 2:
            values = values[..., None]
        values = values.transpose(2, 0, 1)
    elif layout == "HWCT":
        values = values.transpose(2, 3, 0, 1)
    return np.ascontiguousarray(values)


def bayer_mosaic(rgb):
    """Sample C,T,H,W RGB arrays at their observed RGGB sites."""
    mosaic = rgb[1].copy()
    mosaic[:, 0::2, 0::2] = rgb[0, :, 0::2, 0::2]
    mosaic[:, 1::2, 1::2] = rgb[2, :, 1::2, 1::2]
    return mosaic


def main():
    args = parse_args()
    if args.value_scale <= 0:
        raise ValueError("--value-scale must be positive")
    device = torch.device(args.device)
    if args.precision == "bf16" and device.type != "cuda":
        raise ValueError("The released BF16 protocol requires CUDA")
    config = read_config(args.config)
    frames = config["model"]["frames"]
    color = config["model"]["color_channels"] == 3
    torch.manual_seed(42)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = True
    ffn = args.ffn_impl or ("triton" if device.type == "cuda" else "cudnn")
    model = create_model(config, ffn)
    metadata = load_reference(model, args.checkpoint)
    model = model.eval().to(device)
    if args.compile:
        model = torch.compile(model, backend="inductor", mode="default")
    mask = load_mask(args.mask, frames, args.mask_key, args.mask_layout).numpy()
    # Match CACTI's column-major spatial storage, without changing mask values.
    # Tensor strides affect the kernels chosen for BF16 compiled inference.
    mask = np.ascontiguousarray(mask.swapaxes(-1, -2)).swapaxes(-1, -2)
    normalizer = mask.sum(0)
    normalizer[normalizer == 0] = 1
    phi = torch.from_numpy(mask[None]).to(device)
    phis = torch.from_numpy(normalizer[None, None]).to(device)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    records = []
    for path in args.input:
        data = to_array(read_array(path, args.key), args.layout)
        # CACTI retains gray source precision through measurement formation;
        # its color loader normalizes and sums FP32 arrays.
        data = (data.astype(np.float32) if color else data) / args.value_scale
        if tuple(data.shape[-2:]) != tuple(phi.shape[-2:]):
            raise ValueError(f"Sequence and mask sizes differ: {data.shape}, {phi.shape}; crop explicitly first")
        if not args.real and color != (data.ndim == 4 and data.shape[0] == 3):
            raise ValueError("Use gray T,H,W targets or color 3,T,H,W targets with the matching config")
        length = data.shape[-3]
        if args.real:
            if data.ndim != 3:
                raise ValueError("Real measurements must have shape N,H,W")
            steps, dropped = length, 0
        else:
            steps, dropped = divmod(length, frames)
        if not steps:
            raise ValueError(f"No complete {frames}-frame snapshot in {path}")
        outputs, psnrs, ssims = [], [], []
        with torch.no_grad():
            for step in range(steps):
                if args.real:
                    measurement = data[step] * args.measurement_scale
                else:
                    target = data[..., step*frames:(step+1)*frames, :, :]
                    truth = bayer_mosaic(target) if color else target
                    measurement = (truth * mask).sum(0)
                measurement = torch.from_numpy(
                    np.ascontiguousarray(measurement[None, None], dtype=np.float32)
                ).to(device)
                context = torch.autocast("cuda", dtype=torch.bfloat16) if args.precision == "bf16" else nullcontext()
                with context:
                    prediction = model(measurement, phi, phis)[-1]
                outputs.append(prediction.float().cpu().numpy()[0])
                if not args.real:
                    # Paper protocol: unclipped predictions, color metrics on RGGB mosaic.
                    estimate = bayer_mosaic(outputs[-1]) if color else outputs[-1]
                    for estimated_frame, true_frame in zip(estimate, truth):
                        a, b = estimated_frame*255, true_frame*255
                        psnrs.append(compare_psnr(a, b)); ssims.append(compare_ssim(a, b))
        name = Path(path).stem
        np.save(output/f"{name}_reconstruction.npy", np.concatenate(outputs, axis=1 if color else 0))
        record = {"sequence": name, "snapshots": steps, "dropped_tail_frames": dropped}
        if psnrs:
            record.update(psnr=float(np.mean(psnrs)), ssim=float(np.mean(ssims)), frames=len(psnrs))
        records.append(record)
    summary = {"sequences": records, "precision": args.precision, "compiled": args.compile,
               "ffn": ffn, "checkpoint_metadata": metadata, "real": args.real}
    if not args.real:
        summary["psnr"] = float(np.mean([r["psnr"] for r in records]))
        summary["ssim"] = float(np.mean([r["ssim"] for r in records]))
    (output/"metrics.json").write_text(json.dumps(summary, indent=2)+"\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

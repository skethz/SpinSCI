# Inference masks

These are the original sensing-mask files used for SpinSCI benchmark inference.
Each MATLAB file contains only a `mask` array in `H,W,T` order. The arrays and
original file bytes are preserved; sizes, data types, and SHA-256 hashes are in
[`manifest.json`](manifest.json).

| Benchmark | File | Stored shape | Use |
| --- | --- | --- | --- |
| Small-scale gray | `gray_cr8.mat` | 256 × 256 × 8 | All six gray sequences, CR8 |
| Mid-scale color | `mid_color_cr8.mat` | 512 × 512 × 8 | All six mid-color sequences, CR8 |
| Large-scale color | `large_color_1080x1920_cr8.mat` | 1080 × 1920 × 8 | Hummingbird, Messi, Swinger, CR8 |
| Large-scale color | `large_color_822x1920_cr8.mat` | 822 × 1920 × 8 | Football, CR8 |
| Real data | `real_mask.mat` | 512 × 512 × 50 | First 10 frames for CR10; all 50 for CR50 |

The simulated-benchmark masks are binary. The real mask is a calibrated
floating-point mask: retain its values without thresholding or rescaling.
The shared benchmark evaluator (`tools.evaluate_benchmarks`) selects the first
`--frames` mask frames, so the commands below use the same `real_mask.mat` file
for CR10 and CR50. The general evaluator (`tools.evaluate`) requires an exact
frame count; for CR10, supply a separate array containing `mask[:, :, :10]`.

The files total approximately 38.8 MiB. Ground-truth videos and real measurement
files are supplied separately. This folder specifies inference inputs; training
commands accept a separately selected mask through `--mask`.

## Benchmark commands

Run from the repository root with the benchmark dependencies installed and one
GPU exposed, for example `CUDA_VISIBLE_DEVICES=0`. Replace only the data paths.
The [reproduction guide](../docs/reproduction.md) describes layout and metric
conventions.

```bash
python -m tools.evaluate_benchmarks --model spinsci \
  --config configs/gray_cr8.json --checkpoint checkpoints/spinsci_gray_cr8.pth \
  --channels 1 --data-root /path/to/gray --mask masks/gray_cr8.mat \
  --frames 8 --layout gray --precision fp32 --no-images \
  --output outputs/gray_benchmark

python -m tools.evaluate_benchmarks --model spinsci \
  --config configs/color_cr8.json --checkpoint checkpoints/spinsci_color_cr8.pth \
  --channels 3 --data-root /path/to/mid_color --mask masks/mid_color_cr8.mat \
  --frames 8 --layout mid_rot90 --precision fp32 --no-images \
  --output outputs/mid_color_benchmark

python -m tools.evaluate_benchmarks --model spinsci \
  --config configs/color_cr8.json --checkpoint checkpoints/spinsci_color_cr8.pth \
  --channels 3 --data-root /path/to/large_color_1080x1920 \
  --mask masks/large_color_1080x1920_cr8.mat \
  --frames 8 --layout large_hdf5 --precision fp32 --no-images \
  --output outputs/large_color_benchmark

python -m tools.evaluate_benchmarks --model spinsci \
  --config configs/color_cr8.json --checkpoint checkpoints/spinsci_color_cr8.pth \
  --channels 3 --data-root /path/to/football \
  --mask masks/large_color_822x1920_cr8.mat \
  --frames 8 --layout large_hdf5 --precision fp32 --no-images \
  --output outputs/football_benchmark

python -m tools.evaluate_benchmarks --model spinsci \
  --config configs/real_cr10.json --checkpoint checkpoints/spinsci_real_cr10.pth \
  --real --channels 1 --data-root /path/to/real_cr10 --mask masks/real_mask.mat \
  --frames 10 --layout gray --precision fp32 --output outputs/real_cr10

python -m tools.evaluate_benchmarks --model spinsci \
  --config configs/real_cr50.json --checkpoint checkpoints/spinsci_real_cr50.pth \
  --real --channels 1 --data-root /path/to/real_cr50 --mask masks/real_mask.mat \
  --frames 50 --layout gray --precision fp32 --output outputs/real_cr50
```

The real-data commands apply the evaluator's `meas / 255 * (T / 2)` convention.
The large-color directories must contain sequences matching the selected mask's
spatial dimensions. Aggregate large-color quality by weighting all four
sequences equally, rather than equally weighting the two directory means.

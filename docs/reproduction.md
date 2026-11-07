# Reproduction guide

This guide documents the training, data conventions, and efficiency measurements
used by the commands in the [README](../README.md).

## Training

The JSON files in [`configs/`](../configs/) define the architecture, optimizer,
and training recipe. The CR8 recipe uses DAVIS2017, 200 epochs, and four GPUs.
Gray crops are 256 x 256 with global batch 10; color crops are 512 x 512 with
global batch 2. These are totals across all GPUs.

Eligible matrix weights use Muon. The remaining trainable parameters use AdamW
with zero weight decay. Both learning rates follow cosine decay from 2e-4 in the
first epoch to 1e-6 in the last epoch. The objective is final-reconstruction MSE
averaged over frames, channels, and pixels. Input/output convolutions, loss, and
parameter updates remain FP32; transformer computation uses BF16 autocast.
The gradient norm is clipped at 1. The seed is 2447.

The DAVIS loader reads only sequences listed in `ImageSets/2017/train.txt`.
Each contiguous eight-frame window occurs once per epoch before shuffling;
the incomplete final batch is dropped. Frames smaller than the crop are resized
with preserved aspect ratio. The same random crop and horizontal/vertical flips
are applied to every frame of a window. Masks are center-cropped to the training
size. Use `--resolution Full-Resolution` when that is the downloaded directory.

Exact global batches are preserved when the batch is not divisible by the GPU
count: extra replica inputs have zero loss weight. For color batch 2 on four
GPUs, two replicas contribute zero gradient. Two GPUs can avoid that redundant
computation while retaining global batch 2. `--batch-scope per_gpu` explicitly
changes the interpretation of the configured batch size.

The auxiliary head after blocks 3 and 6 is retained for checkpoint compatibility,
but frozen and unused in the final-MSE training recipe. Training starts from
random initialization unless a checkpoint is explicitly supplied:

```bash
# Resume model and optimizer at an epoch boundary, using the original GPU count.
torchrun --standalone --nproc_per_node=4 -m tools.train configs/gray_cr8.json \
  --davis-root /path/to/DAVIS --mask /path/to/gray_mask.mat \
  --resume outputs/gray/latest.pth --output outputs/gray

# Initialize only model weights, with a fresh optimizer and learning-rate schedule.
torchrun --standalone --nproc_per_node=4 -m tools.train configs/color_cr8.json \
  --davis-root /path/to/DAVIS --mask /path/to/color_mask.mat \
  --warm-start checkpoints/spinsci_color_cr8.pth --output outputs/color_finetune
```

## Checkpoint loading

The four supplied checkpoint files are preserved byte for byte. Their sizes,
SHA-256 hashes, and matching configurations are listed in
[`checkpoints/manifest.json`](../checkpoints/manifest.json).
CR8 loading selects `model_state_dict`; the real-data files contain bare EMA
state dictionaries. Loading is strict, so incompatible tensor names or shapes
raise an error.

## Input layouts and scaling

The general evaluator, `python -m tools.evaluate`, accepts MATLAB, NumPy, and
NPZ arrays; MATLAB v7.3 files are supported.

| Input | Default key | Layout |
| --- | --- | --- |
| Gray ground truth | `orig` | `H,W,T` (`--layout HWT`) |
| RGB ground truth | `orig` | `H,W,3,T` (`--layout HWCT`) |
| Real measurements | `meas` | `H,W,N` (`--layout HWT --real`) |
| Mask | `mask` | `H,W,T` (`--mask-layout HWT`) |

Use `--key`, `--layout`, `--mask-key`, and `--mask-layout` for other supported
conventions. Values are divided by 255 by default; use `--value-scale 1` for
normalized arrays. Input sizes must match the mask. This evaluator does not
silently crop, rotate, or flip a benchmark.

The original benchmark inference masks are included in [`masks/`](../masks/).
The general evaluator requires the mask's frame count to match the configuration
exactly. For CR10, supply the first ten frames of `real_mask.mat` as a separate
array, or use the shared benchmark evaluator below, which selects those frames
directly from the included 50-frame file. Preserve the real mask's calibrated
floating-point values without thresholding or rescaling.

Gray normalization and measurement formation retain the source array precision
before conversion to FP32 model inputs. Color arrays use FP32 throughout this
preprocessing, matching the shared CACTI benchmark evaluator.
The mask and normalizer also retain CACTI's column-major spatial storage for
consistent tensor layout across inference modes.

For CACTI real files, the measurement is scaled as `meas / 255 * (T / 2)`:
use `--measurement-scale 5` for CR10 and `--measurement-scale 25` for CR50.
For already-normalized sum measurements, use
`--value-scale 1 --measurement-scale 1`.

Reconstruction arrays retain unclipped predictions. PSNR and SSIM are averaged
over frames within each sequence, then equally over sequences. Color metrics
are computed on the RGGB mosaic, rather than all RGB channels. Incomplete
temporal tails are reported and excluded. Large-scale reconstruction quality
uses full frames; tiled efficiency profiling is a separate protocol.

## Shared benchmark evaluation and frame export

The shared evaluator applies the fixed CACTI layouts used for the manuscript
comparisons and exports PNGs alongside metrics. Install
`requirements-benchmarks.txt` first. Expose exactly one GPU:

```bash
export CUDA_VISIBLE_DEVICES=0

python -m tools.evaluate_benchmarks --model spinsci \
  --config configs/gray_cr8.json --checkpoint checkpoints/spinsci_gray_cr8.pth \
  --channels 1 --data-root /path/to/gray --mask masks/gray_cr8.mat \
  --frames 8 --layout gray --precision fp32 \
  --output outputs/gray_benchmark

python -m tools.evaluate_benchmarks --model spinsci \
  --config configs/color_cr8.json --checkpoint checkpoints/spinsci_color_cr8.pth \
  --channels 3 --data-root /path/to/mid_color --mask masks/mid_color_cr8.mat \
  --frames 8 --layout mid_rot90 --precision fp32 \
  --output outputs/mid_color_benchmark
```

`mid_rot90` follows the CACTI mid-color loader's 90-degree rotation and vertical
flip. Use `--layout large_hdf5` for the large-color files and their corresponding
mask. The [mask guide](../masks/README.md#benchmark-commands) provides commands
for both large-color resolutions. Use these conventions only with the matching
dataset layout.
`--save-raw` also stores floating-point predictions; `--no-images` skips PNGs.
Metrics use unclipped, unrounded values. Display PNGs use a fixed intensity
scale: `round(clip(prediction, 0, 1) * 255)`.

For real SpinSCI frame exports, select the real configuration and checkpoint,
then pass `--real --channels 1 --frames 10 --layout gray --precision bf16`
(or the CR50 configuration, checkpoint, and `--frames 50`). Both use
`--mask masks/real_mask.mat`; the evaluator selects the first 10 or all 50 mask
frames and applies `meas / 255 * (T / 2)` automatically. Real-data
quality is shown qualitatively because ground truth is unavailable. The
manuscript compares only methods with officially released checkpoints for
real-data reconstruction.

Sensor-orientation PNGs can be converted to the manuscript's display orientation
without changing the saved originals:

```bash
python -m tools.orient_real_benchmark --input outputs/real_cr10/png \
  --output outputs/real_cr10/png_display
```

Hand remains unchanged. Domino, Pendulum Ball, and Water Balloon use transpose,
45-degree counterclockwise bicubic rotation without expansion, then a horizontal
mirror. This is a display transform after reconstruction, with no contrast
rescaling.

## Efficiency profiling

`tools.profile_benchmarks` measures exact parameter totals, module operation
counts, latency, and peak allocated memory using synthetic inputs. No dataset
is required. Expose one GPU and use the matching gray/color configuration:

```bash
export CUDA_VISIBLE_DEVICES=0

# Params and FLOPs for a full 256 x 256 gray snapshot.
python -m tools.profile_benchmarks --model spinsci \
  --config configs/gray_cr8.json --checkpoint checkpoints/spinsci_gray_cr8.pth \
  --case gray256 --mode costs --output outputs/gray_costs.json

# Mid-color FP32 eager inference latency and peak memory.
python -m tools.profile_benchmarks --model spinsci \
  --config configs/color_cr8.json --checkpoint checkpoints/spinsci_color_cr8.pth \
  --case color512 --mode runtime --precision fp32 --warmups 3 --runs 10 \
  --output outputs/color_runtime.json

# Complete large-color inference cost and tiled runtime.
python -m tools.profile_benchmarks --model spinsci \
  --config configs/color_cr8.json --checkpoint checkpoints/spinsci_color_cr8.pth \
  --case large24 --mode costs --output outputs/large_costs.json

python -m tools.profile_benchmarks --model spinsci \
  --config configs/color_cr8.json --checkpoint checkpoints/spinsci_color_cr8.pth \
  --case large24 --mode runtime --precision fp32 --warmups 3 --runs 10 \
  --output outputs/large_runtime.json
```

The main-paper runtime comparisons use FP32 eager inference on one NVIDIA GH200 144 GB, batch size 1 and eight frames. SpinSCI-B uses three warm-ups and the median of ten CUDA-synchronized wall-clock measurements. The timed scope is the complete reconstruction from GPU-resident inputs. Peak allocated memory includes the weights and resident inputs. The appendix compares BF16 autocast with default Inductor compilation for all seven mid-color methods, using five warm-ups and twenty timed calls after compilation.

One THOP module MAC is reported as one table FLOP. Functional attention products
and unregistered elementwise operations are not included in that convention.
The equivalent cuDNN depthwise FFN is used for counting; SpinSCI runtime uses the
Triton FFN. Parameters are counted from the complete instantiated model,
including unused auxiliary parameters. The script also records THOP's parameter
diagnostic and parameters whose owning modules did not execute.

| Case | Exact parameters | FLOPs (G) | Latency (ms) | Table peak (MiB) |
| --- | ---: | ---: | ---: | ---: |
| Gray 256 x 256 | 7,465,294 | 1214.393011 | 132.116 | 1850.170 |
| Color 512 x 512 | 7,469,266 | 4855.741374 | 630.086 | 7151.387 |
| Large color | 7,469,266 | 38401.804538 | 4199.640 | 2423.724 |

Large-color SpinSCI-B FLOPs use a full 1080 × 1920 forward. For every method,
the Table 6 latency estimate is 24 times the median time for a standalone 272 × 320 tile. Table memory is the peak allocation for that tile. The profiler also directly times 24 sequential tile forwards and assembly on the GPU, including a crop from 1088 × 1920 to 1080 × 1920. The separate `whole_snapshot` record for SpinSCI-B is 4187.269 ms and 2693.998 MiB. Runtime and
memory can vary slightly between runs. Hardware, software versions, checkpoint
hashes, and timing samples are saved in each generated JSON result.

`tools.profile` is a simpler single-forward utility with the same eager-reference
check before compilation, warmups, and timing. It reports total parameters and
the auxiliary-head-excluded count separately. Use
`tools.profile_benchmarks` for the table's parameter/FLOP convention and complete
large-snapshot timing.

## DU-9stg comparison

The shared tools load the [official DU implementation](https://github.com/Muyuan-Zhang/DU)
from an external checkout, using `dim=48`, `stage=9`, and `skip=3` with strict
checkpoint loading. Supply its official gray or color checkpoint separately.

```bash
# Same profiler, DU mid-color FP32 eager runtime.
python -m tools.profile_benchmarks --model du --du-root /path/to/DU \
  --checkpoint /path/to/DU/checkpoint/color_9stg.pth --case color512 \
  --mode runtime --precision fp32 --output outputs/du_color_runtime.json

# Shared-mask evaluation of the released gray checkpoint.
python -m tools.evaluate_benchmarks --model du --du-root /path/to/DU \
  --checkpoint /path/to/DU/checkpoint/gray_9stg.pth --channels 1 \
  --data-root /path/to/gray --mask masks/gray_cr8.mat \
  --frames 8 --layout gray --precision fp32 --output outputs/du_gray
```

Use `--channels 3 --layout mid_rot90` for mid-color or
`--channels 3 --layout large_hdf5` for large-color quality evaluation, with the
color checkpoint and corresponding masks. Add `--precision bf16 --compile` to
the profiler for a matched compiled-execution comparison. In that setting DU
uses less peak memory than SpinSCI but remains slower; the main-paper runtime comparisons use FP32 eager execution for every method.

| Benchmark | DU reported PSNR | DU measured PSNR | Quality values used in comparisons |
| --- | ---: | ---: | --- |
| Small-scale gray | 37.57 | 37.523 | Reported PSNR/SSIM |
| Mid-scale color | 38.30 | 38.117 | Measured PSNR/SSIM |
| Large-scale color | Unreported | 37.799 | Measured PSNR/SSIM |

The gray mean differs by less than 0.1 dB. The mid-color difference exceeds
0.1 dB and persists with the supplied mask; its cause remains unexplained.

## Functional checks

```bash
python -m unittest discover -s tests -v
```

The tests cover all four checkpoint loads and small reconstruction forwards,
a full-model backward/optimizer step, coupling equations, Bayer selectors and
correction, MSE normalization, cosine endpoints, exact distributed global-batch
gradients, constructor defaults, gray measurement precision, epoch-boundary resume, and exclusion of the
DAVIS validation split. These functional checks
are separate from full training and benchmark evaluation.

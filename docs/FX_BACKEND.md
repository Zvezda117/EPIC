# Explicit-geometry L-band FX backend

This independent candidate backend images complex voltage NPZ arrays using a
persistent native FP32 CUDA/cuBLAS context. It supports noninteger, noncoplanar
antenna positions and explicit per-channel frequencies. It produces engineering
NPZ products; it does not infer UTC, source coordinates, station geometry or
calibration, and does not write FITS.

## Build and minimal Python example

From this repository root (CUDA 12.1/cuBLAS already installed):

```bash
bash src/fx/build.sh
PYTHONPATH=src/python python3 your_fx_script.py
```

The shared library defaults to `src/fx/libepic_fx.so` relative to the wrapper,
independent of the current working directory. `EPIC_FX_LIBRARY` may override it;
an invalid explicit override fails without falling back. Build overrides are
`NVCC`, `CUDA_ARCH` (default `sm_89`) and `CUDA_LIB_DIR` for runtime search.

Copyable `your_fx_script.py` for **already phase-tracked** input:

```python
import json
import numpy as np
from lband_fx import FXImager

with np.load('voltages.npz', allow_pickle=False) as source:
    voltage = source['data']  # complex [T,C,N,2], already tracked to phase_center
with np.load('geometry.npz', allow_pickle=False) as geometry:
    positions = geometry['positions_m']
    frequencies = geometry['frequencies_hz']
    directions = geometry['directions']
    center = geometry['phase_center']
    gains = geometry['gains'] if 'gains' in geometry else None
with FXImager(positions, frequencies, directions, center, gains=gains,
              ntime=voltage.shape[0], device=1, remove_autocorr=False) as imager:
    image = imager.image(voltage)
np.savez('fx_engineering_image.npz', image=image, frequencies_hz=frequencies,
         directions=directions, phase_center=center,
         metadata_json=np.asarray(json.dumps({
             'artifact': 'engineering_FX_products', 'normalization': 'mean',
             'products': ['XX', 'YY', 'Re(XY*)', 'Im(XY*)'],
             'time_basis': 'sequence_indices_no_UTC',
             'sequence_start': 0, 'sequence_end': voltage.shape[0],
             'sequence_count': voltage.shape[0],
             'voltage_convention': 'phase-tracked_to_explicit_phase_center'})))
```

The native context keeps steering, gain arrays, cuBLAS handle and buffers across
`image()` calls. It expects the same `[T,C,N,2]` dimensions each time, accepts
noncontiguous arrays by converting them, returns `float32[C,P,4]`, and closes
idempotently. Geometry/gains are fixed for that context; create another context
when they change. Methods on one Python context serialize with a lock; native
C callers must serialize access to their own context.

## Scientific data contract

- `positions_m[N,3]`: finite Cartesian metres, in antenna order matching input.
  Prefer coordinates relative to an array reference point.
- `frequencies_hz[C]`: explicit positive **RF sky frequencies** in input channel
  order, including correct channel width, FFT order and sideband mapping. The
  backend does not derive frequencies from `chan0`, NPZ metadata or bandwidth.
- `directions[P,3]` and `phase_center[3]`: unit vectors in the **same Cartesian
  frame** as positions. They are not RA/Dec pairs or implicit zenith coordinates.
- `gains[C,N,2]`: complex voltage multipliers, `corrected_voltage = gain * input`.
  Missing H/V inputs use zero voltage or zero gain. No 4-bit quantization or
  gain estimation is performed. Scalar per-polarization gains are not a full
  leakage-correcting Jones matrix.
- Input voltages use the stated complex IQ convention and must be scientifically
  interpretable **voltage samples**, not already averaged powers/correlations.
  The time dimension is counted in sequences, with no invented UTC or duration.

Steering is exactly the supplied far-field 3-D model:

```text
A[c,p,a] = exp(+2*pi*i*frequencies_hz[c] * positions_m[a] dot (s[p]-s0) / c_light)
RXX = mean_t X X^H; RYY = mean_t Y Y^H; RXY = mean_t X Y^H
image products = diagonal(A RXX A^H), diagonal(A RYY A^H),
                 real/imag diagonal(A RXY A^H)
```

The subtraction of `s0` assumes that input has already been phase-tracked to
that direction with a compatible sign convention. Merely setting a source as
`phase_center` is **not calibration**. For untracked raw voltages, pre-multiply
both polarizations by `exp(+2*pi*i*f*r dot s0/c_light)`, so combined steering
is `exp(+2*pi*i*f*r dot s/c_light)`. This positive sign presumes incoming source
voltages have the corresponding negative geometric phase; an opposite IQ/
sideband convention must be corrected explicitly by the caller. The runner's
`--voltage-convention raw` applies this stated tracking gain, not an inferred
source correction.

`remove_autocorr=True` deletes the same **physical antenna** diagonal of all
three covariance matrices, including XY. It removes each antenna's self terms,
not merely the diagonal of an interleaved 2N-input matrix. The four planes are
XX/YY/ReXY/ImXY, not Stokes I/Q/U/V or four complex XX/XY/YX/YY planes.

Scientific use is trustworthy only after RF frequency/sideband, IQ sign,
antenna order, coordinate frame, phase tracking and calibration are verified
from the observation. The wrapper can check shape, finite values and vector
length, but cannot establish those physical facts from an NPZ alone.

## Tiled/windowed NPZ runner

The runner requires `data[T,C,N,2]` and a separate explicit geometry NPZ containing
the fields above (gains optional). For raw input, for example:

```bash
python3 scripts/run_lband_fx.py \
  --input voltages.npz --geometry geometry.npz --output fx_engineering_run \
  --devices 0,1 --channel-tile 16 --window-sequences 1000 \
  --coordinate-frame local_ENU --iq-convention 'I+iQ' \
  --voltage-convention raw
```

Use `--devices 1` for one card, `--voltage-convention phase-tracked` for inputs
already tracked to the geometry center, and optionally `--remove-autocorr`.
The output directory must be new; existing output is never overwritten. Each
window/channel tile has its own NPZ and JSON metadata. A final `manifest.json`
lists all successful chunks and their original channel/sequence ranges. An
absent manifest indicates an incomplete run; partial chunks may remain after
an error. Frequencies are preserved unchanged. The final short window uses its
actual sequence count for the mean; it is never silently zero-padded.

One worker per selected GPU processes bounded channel tiles. Each tile reuses
one context over full windows, closes it before creating a final shorter context,
then advances to its next tile. The runner never allocates an entire time-by-pixel
electric-field cube or retains all window images in RAM. NPZ decompression still
loads **the whole voltage array into CPU memory**; it is not an out-of-core input
reader. Native peak buffer estimates (including temporary geometry) are capped
at 900 MiB per context, with checked arithmetic. Reduce channel tile/window size
if needed. This is a safety bound, not a measured optimal batching choice.

## Verification and limits

```bash
# CPU contract/oracle tests; GPU tests deliberately skip without opt-in
python3 -m unittest discover -s tests -p 'test_lband_fx.py' -v
EPIC_FX_TEST_GPU=1 python3 -m unittest discover -s tests -p 'test_lband_fx.py' -v
EPIC_FX_TEST_GPU=1 python3 -m unittest discover -s tests -p 'test_run_lband_fx.py' -v
# Only after coordinating a short GPU0 window:
EPIC_FX_TEST_GPU=1 EPIC_FX_TEST_TWO_GPU=1 \
  python3 -m unittest discover -s tests -p 'test_lband_fx.py' -v
EPIC_FX_TEST_GPU=1 compute-sanitizer --tool memcheck --leak-check full \
  --error-exitcode 86 python3 -m unittest discover -s tests -p 'test_lband_fx.py' -v
```

The independent double-precision NumPy oracle computes electric fields per
pixel, not covariance products. Covered: unit/complex gains, coherent antennas,
noninteger/noncoplanar positions, two RF frequencies, zero inputs, repeated calls,
all-product single-antenna self-term removal, shape/finite/device errors and
two-card channel reconstruction. Initial tests produced RED before implementation;
logs in `src/fx/TDD_*.log` record each cycle. First native GPU checks had maximum
absolute error `2.73e-6` and maximum scaled error `1.18e-7`; two-card shards matched
whole GPU1 with maximum absolute difference zero. Native GPU1 memcheck reported
zero errors and zero leaked allocations. Tests use `rtol=atol=4e-5`.

cuBLAS uses FP32 complex products with `CUBLAS_PEDANTIC_MATH`, TF32 disabled and
no fast-math build flag. Steering angles/trigonometry use double before storing
complex FP32. A finite input can still overflow FP32 power at extreme gain/voltage
amplitude; nonfinite outputs raise an error. This is a model-specific candidate
backend with no real-observation validation, full Jones correction, automatic
UTC/FITS/WCS or end-to-end fastest-pipeline claim.

The representative **new API** benchmark is reproducible with
`python3 src/fx/benchmark_api.py` on GPU1. Its synthetic noncoplanar case uses
T=1000, C=128, N=16 slots (10 active antennas with H5/V7 availability counts)
and P=4096. One warmup plus five `image()` wall measurements yielded median
16.014 ms (15.901–20.328 ms), including Python/native validation, gain packing,
H2D, covariance/imaging, D2H and finite-output checks. Separate cold setup was
413.27 ms. Two channels × 16 sampled pixels matched the independent CPU-double
oracle with maximum absolute error `4.33e-6` (scaled `2.87e-7`).
The positions and voltages were synthetic, and this measured availability-count
model is not an observation result. Details are saved in
`src/fx/API_BENCHMARK.json`; this API measurement is distinct from the earlier
single-polarization resident-compute microbenchmark and proves no two-GPU speedup.

## Time-dependent astronomical pointing

For raw voltages with a confirmed first-sample epoch and surveyed local ENU
positions, `--sky-config` now generates a source-centered SIN grid and updates
steering each window. See [Crab observation-model repair](CRAB_OBSERVATION_REPAIR.md)
for the JSON contract, Earth-orientation data, validation and unresolved 9178
metadata. A successful numerical run alone does not establish a Crab detection.

For the user-confirmed ITRF coordinate authority, build geometry with
`scripts/build_itrf_geometry.py`; the runner validates/adopts its ENU reference
site. See [ITRF geometry](ITRF_GEOMETRY.md).

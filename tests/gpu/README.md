# Small GPU imager regression

`imager_regression.cu` launches the production `GetImagingKernel<FFT,float2>` /
`BlockEpicImager` with two artificial channels, sixteen compiled slots and one
or two active antennas, plus zero input. It never loads observation files.
Cases cover 64 and 128 grids, support 1/3/5, one/four/32 sequences,
unity/complex gains, a single block with global channel offset 1, enabled and
disabled autocorrelation removal, and two-gulp accumulation. Support-five
footprints reach outer cells and cross both halves of shared memory. Explicit
GPU payload allocations stay below 3 MiB per case, including the per-antenna
self-response grid. Each imager launch has one or two blocks.

The independent CPU oracle explicitly grids sparse voltage points and sums a
direct inverse DFT using double complex arithmetic. It accounts for the
production output transpose, the once-only voltage division by 64 and the
post-product multiplier N; all four products use `X*conj(Y)`. Correction is one.
For enabled self removal, the CPU oracle subtracts each antenna's direct-DFT
power and cross product. Single-antenna output must therefore approach zero,
not the positive unremoved power. The tolerance is
`2e-5 + 0.0075 * per-plane unremoved peak * gulps` for half FFT rounding. It
does not accept arbitrary renormalization, channel permutations or large race
errors. Unlaunched channels must retain an output sentinel.

Build against the current signature ending in `bool remove_autocorr=false,
const float* auto_power=nullptr`:

```sh
bash tests/gpu/build_regression.sh repair-build/gpu_regression
```

Only when compiling against an actual historical header without these two
arguments, select the legacy ABI explicitly:

```sh
EPIC_HAS_AUTOCORR_FLAG=0 bash tests/gpu/build_regression.sh repair-build/gpu_regression_red
```

The current variant explicitly supplies both arguments to `cudaLaunchKernel`;
C++ default arguments do not apply to its `void**` launch interface. This
suite also launches the actual `ComputeAutoCorrPower` for enabled-self-removal
fixtures. Enabled removal uses float spatial FFTs to avoid half-FFT cancellation
residuals; disabled removal preserves the packed half FFT. Both paths retain
the same voltage scaling and use float product accumulation in these tests.

`EPIC_MATHDX_DIR` can override the bundled MathDx include directory. Tests
require CUDA 12.1 and an SM89 device; missing hardware is not a GPU pass.

Run only on the assigned device (GPU0 in this repair session):

```sh
timeout 45s repair-build/gpu_regression --device 0 --repeat 3
timeout 45s /usr/local/cuda-12.1/bin/compute-sanitizer --tool racecheck --error-exitcode 99 --print-limit 8 repair-build/gpu_regression --device 0 --case 64-two-s1-t1
timeout 30s /usr/local/cuda-12.1/bin/compute-sanitizer --tool memcheck --error-exitcode 99 --print-limit 12 repair-build/gpu_regression --device 0 --case boundary-low
timeout 30s /usr/local/cuda-12.1/bin/compute-sanitizer --tool memcheck --error-exitcode 99 --print-limit 12 repair-build/gpu_regression --device 0 --case boundary-high
```

Boundary cases run only when explicitly selected: the legacy kernel can poison
its CUDA context after the invalid write, so each belongs in an isolated
process. Exit codes are 0 (pass), 1 (numerical mismatch), 2 (CLI error), or 3
(CUDA failure); sanitizer errors return 99 and timeout returns 124.

The first legacy RED run is recorded in `repair-build/gpu_regression_red.log`.
All seven ordinary cases had failures over three repetitions. In particular,
`128-two-s3-t4-gain` failed twice (maximum XX errors 0.290 and 1.111) and passed
once (0.0214) with identical input. Offset cases had nearly full-scale errors.
`gpu_boundary_low_red.log` records an invalid atomic at `gridder.cuh:586` and
an ensuing launch failure (sanitizer summary: nine errors).
`gpu_racecheck_red.log` records three shared-memory race errors.
`gpu_boundary_high_red.log` has no memcheck errors but fails the zero-output
oracle; that case alone does not distinguish the missing Y check from the
legacy unconditional autocorrelation subtraction.

The preserved RED executable SHA256 is
`286adbdaf3a6c84118f4419156637bd67db7828531a106bafb817b0f9bd1c761`.

The repaired suite has 22 default fixtures, each repeated three times in
`repair-build/gpu_precision_suite.log`: 66 passes and zero failures. The
standalone Y-boundary fixtures pass with exact zero output and zero memcheck
errors. The complete suite passes memcheck, racecheck and synccheck;
corresponding logs are `gpu_precision_{memcheck,racecheck,synccheck}.log`.
The precision-specific RED and GREEN logs retain the `128-auto-single-s3-gain`
YY residual before (0.00266275) and after (2.08616e-7), without changing the
error budget. The final executable is `repair-build/gpu_regression_precision`.

`gcf_regression.cu` exercises the actual `ComputeGcfElements` using a
constant-one CUDA texture, two channels and sixteen antenna slots. Supports
1/3/5 test more groups than antennas; supports 9/10 test partial second rounds,
including the oversampled support 10 used with support 5 and factor 2. Every
antenna's kernel must sum to one, with equal cell weights `1/support**2`.

```sh
bash tests/gpu/build_gcf.sh
timeout 30s repair-build/gcf_regression --device 0
timeout 45s /usr/local/cuda-12.1/bin/compute-sanitizer --tool synccheck --error-exitcode 99 repair-build/gcf_regression --device 0
```

All five GCF cases pass; maximum sum error is 2.24e-8. Memcheck and synccheck
report zero errors (`repair-build/gcf_{gpu,memcheck,synccheck}.log`).

These tests certify only their artificial fixtures. They do not validate
observation time, physical channelization, array geometry, calibration, the
full multi-stream host-copy path or real-data scientific imaging.

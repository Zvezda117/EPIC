# L-band GPU repair and algorithm benchmark

> Continue the user-authorized staged repair using systematic debugging and TDD. Work in the existing isolated worktree; preserve source snapshots and apply verified changes back with conflict checks.

**Goal:** Correct and GPU-test the MOFF execution path on the two RTX 4090s, and measure relevant algorithm alternatives for sparse L-band arrays.

**Architecture:** Keep the validated channel-major input and existing FFT normalization. Separate per-observation gridding/auto-response precomputation from per-gulp integration. Partition independent frequency channels across GPUs to avoid cross-NUMA GPU traffic. Use independent direct Fourier references to validate numerical results before interpreting throughput.

**References:** EPIC 2017 (arXiv:1510.08318), MOFF (arXiv:0812.3669), the pinned upstream comparisons in docs/review, and batch-1 status.

## Constraints

- The host has two 24GB RTX 4090s; GPU access requires execution outside the restricted sandbox. Do not confuse sandbox visibility with hardware availability.
- Do not change clocks, reset GPUs, or overwrite past observational outputs.
- GPU regression work uses GPU0; independent benchmarking uses GPU1. Keep probes small and bounded.
- Benchmark 10 active antennas versus padded 16 slots distinctly; present synthetic one-polarization experiments honestly.
- Preserve true XX/YY/ReXY/ImXY definitions. Optional auto removal must subtract each antenna's calibrated gridded power/cross-products, not the FFT of raw power with an unrelated normalization.
- Do not invent real UTC, coordinate frame, feed convention, phase center or calibration values.

## Work

- [x] Confirm devices/topology and read the original cost/equivalence arguments.
- [x] Build a small real-CUDA-kernel regression harness with an independent sparse-grid DFT oracle; record pre-fix failures.
- [x] Repair shared-memory stage barriers, two-dimensional bounds, full channel GCF indexing and host upload dependencies.
- [x] Implement explicit optional auto subtraction using precomputed per-antenna gridding power responses with the same voltage FFT scaling; test single antenna, complex gain, cross-products and accumulation.
- [x] Remove conditional full-block barriers from GCF element generation; verify with synccheck and normalized-kernel cases.
- [x] Run numerical tests, memcheck/racecheck/synccheck, and repeatability on GPU0; run small numerical cases on GPU1 too.
- [x] Measure identical synthetic inputs with voltage DFT, FX+image and ideal gridded MOFF where implemented. Report errors, warmup, timing boundaries, transfer/precompute costs and applicability limits.
- [x] Verify actual dual-GPU work partitioning does not duplicate a whole observation.
- [x] Review the diff independently, run CPU and GPU suites, build/link, update status and apply changes back.

## Acceptance

Known deterministic input must match the independent reference within an explicitly reported half-FFT error budget. Zero input stays zero; single-antenna auto removal produces zero within that budget; gain doubling gives fourfold power; channels/streams/devices preserve identities. Sanitizers must report no detected hazards for tested kernels. A throughput claim is limited to the tested workload and timing boundary, not a universal fastest-algorithm claim or a calibrated observation result.

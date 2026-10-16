# Core repair status — 2026-10-02

Batches 1 and 2 are implemented. Batch 2 adds real dual-RTX4090 verification and a native L-band FX backend. See [batch-2 results](BATCH2_GPU_REPAIR.md) and [algorithm choice](LBAND_ALGORITHM_CHOICE.md). This is **not a calibrated astronomical release**; physical observation conventions and calibration still require verification.

## Upstream baseline

Remote refs were checked on 2026-10-02:

| Reference | Commit | Role |
|---|---|---|
| LWA_EPIC `phoenix` | `44bb9b43cf8a13a39e572e1a721b5e9473faf63f` | Current default CUDA branch |
| LWA_EPIC `aiyuan/offline` | `93aab8c88a756e0c0a3c145ff9e8980611e4d400` | Current local committed baseline |
| LWA_EPIC `main` | `0158349f246bd03d78df2f731a3acd21777b9a7a` | Historical Python/Bifrost implementation |
| General EPIC `master` | `c2bd68b792a5269cfffe1d93b4710eae9ba8ca55` | Algorithm reference; different implementation |

The five central CUDA/bridge files in `phoenix` and `aiyuan/offline` are identical. The input layout mismatch, RRII cross-product error, unconditional/incomplete autocorrelation subtraction, shared-memory synchronization gaps and stream GCF indexing problem are inherited. The absolute-epsilon correction flattening is a local regression; the 1024-channel buffer sizing mismatch is an incomplete local expansion.

Historical `main` and generalized EPIC clarify correlation and autocorrelation intent but are not interchangeable numerical reference implementations. In particular, the CUDA FFT's intermediate `1/64` scale has a half-overflow history. Batch 1 does not change that scale or claim absolute flux normalization is verified.

Full pinned-source comparisons are preserved in `docs/review/UPSTREAM_COMPARISON.txt` and `docs/review/GENERAL_EPIC_COMPARISON.txt`.

## Implemented in batch 1

- Convert time-major NPZ to contiguous channel-major voltages before CNib packing; preserve logical shape in metadata.
- Reject missing explicitly selected files, invalid metadata/shape, stand/polarization mismatch, non-finite/fractional/out-of-range nibble components, sequence-count disagreement, insufficient destination capacity and unrepresentable integer frequencies.
- Require signed CNib components in `[-8, 7]`. Floating unquantized experiments must be explicitly quantized before imaging; the bridge no longer silently wraps/truncates them.
- Check the C++ destination and metadata before copying. Correct the reported byte count to one byte per complex CNib.
- Derive output buffer size from actual channel/image dimensions; offline correlator uses two output slots instead of reserving 100 full images. Large input gulps still have a fixed capacity and are safely rejected when too large; this is not arbitrary-size streaming support.
- Correct `Re(X*conj(Y))` for RRII layout and promote half values before multiplying, avoiding premature half overflow.
- Restore the inverse IFFT power response while preserving upstream normalization. Shift spatial axes only, so odd channel counts are not permuted. Mask response below `1e-6` of the per-channel peak to zero rather than inventing a constant gain or infinite correction. Invalid Python kernels are rejected.
- Remove correction function writes to shared CWD debug filenames; tests run without overwriting historical diagnostics.
- Use the actual binned channel width and group centers in FITS. Label the four real planes `CORRPROD` with `POLPROD=XX,YY,RE_XY,IM_XY`, not the misleading standard `XX,YY,XY,YX` Stokes codes.

## Current verification and entrypoints

The final combined CPU/GPU suite has **46 passing tests, no skips** when run with
`EPIC_FX_TEST_GPU=1 EPIC_FX_TEST_TWO_GPU=1`. Without GPU opt-in, the CPU suite has
38 passes and 8 explicit skips. Batch-1 numerical tests are included.

The production CUDA harness has 22 cases repeated three times (66 passes), plus
five GCF normalization cases. memcheck/racecheck/synccheck report no detected
errors for tested kernels. Full NPZ-to-FITS integration confirms exact single-card,
multiple-stream and dual-card channel reconstruction on a small synthetic input.
These tests do not establish sky calibration or real-time full-bandwidth throughput.

The changed `main.cpp`, `MOFF_cu_handler.cu`, and `host_helpers.cu` were rebuilt and
linked using existing dependencies; both compile and device link target sm_89.
This is not a clean-room dependency rebuild. The validated workstation executable
is delivered as `build/epic-repaired++`, launched by `scripts/run_moff_offline.sh`.
The older `build/epic++` is preserved. Rebuilding the normal target requires selecting
the intended Python environment and matching C++ runtime; see the batch-2 guide.

For the new native FX backend use [FX_BACKEND.md](FX_BACKEND.md). Its standalone
CUDA/cuBLAS build avoids the legacy Raft/observatory dependencies. It accepts
explicit geometry/frequencies/gains and emits engineering NPZ products.

Existing historical FITS images are unchanged and are not retroactively corrected.

## Remaining gates

1. Supply and verify actual phase center, coordinate frame, IQ/sideband convention,
   valid polarization masks, station timing, antenna/cable calibration and input
   channelization. The legacy MOFF custom-array gains remain unity and its 2-D
   approximation/WCS are not a validated L-band measurement model.
2. Repair and validate 9197/9178 raw channelization, multi-file sample alignment and
   flag/quality handling against the instrument. The old shell pipelines that emit
   `tstart=0` now fail explicitly in MOFF rather than producing fabricated UTC.
3. NPZ input still decompresses the full array in CPU memory; the new FX runner
   bounds device tiles and output windows but is not an out-of-core voltage reader.
4. Measure real F-engine, I/O, sustained dual-card throughput and full 250 MHz
   operation. At 25 kHz spacing full 250 MHz is 10000 channels, not 1024.
5. Complete a real calibration-source acceptance test and clean dependency rebuild
   before claiming a scientifically usable release. BF16 accumulation has a separate
   precision tradeoff and is not covered by the FP32-accumulation CUDA acceptance.

GPU access in this managed session requires execution outside the filesystem sandbox.
Earlier reports of no CUDA devices described sandbox visibility, not missing hardware.

## 2026-10-16: raw Crab observation model

Packet integrity checks, chunk epoch offsets, explicit UTC/site/ICRS geometry and
per-window raw tracking are now implemented and tested on both GPUs. See
[Crab observation-model repair](CRAB_OBSERVATION_REPAIR.md). Real-data scientific
validation remains blocked by the missing 9178 acquisition record, undocumented
header-clock semantics. The user subsequently selected ITRF as authoritative;
[ITRF-derived ENU geometry](ITRF_GEOMETRY.md) now resolves the conflicting coordinate
columns for runtime use. Earlier
zenith-centered diagnostic images are not a Crab-centered sky map.

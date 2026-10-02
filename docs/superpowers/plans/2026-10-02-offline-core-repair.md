# Offline core repair, batch 1

> Execute with superpowers:executing-plans; use failing behavioral tests before fixes.

**Goal:** Repair independently verifiable input and numerical defects while comparing the same algorithms with upstream LWA_EPIC.

**Architecture:** Preserve the CUDA kernel's channel-major input contract and transpose validated time-major NPZ data at the Python/C++ boundary. Keep pure numerical and input validation helpers free of observatory/network imports so CPU regression tests can exercise production code.

**Tech stack:** C++17/CUDA 12.1, Python 3.10, NumPy, stdlib unittest; existing MathDx headers.

**Spec:** User-approved staged repair following `/mnt/data_ssd/zdj/lwa_epic_review_20261002/REVIEW.txt`.

## Constraints and decisions

- Keep all existing local adaptations; never overwrite raw data or past results.
- Implement and test in `/mnt/data_ssd/zdj/LWA_EPIC_fixwork`; apply verified files back only if original file hashes still match the snapshot.
- Compare pinned `phoenix`, historical `main`, and `aiyuan/offline`; also inspect generalized EPIC as an algorithm reference where useful. Upstream identity is not proof of correctness.
- No GPU device is currently accessible. CUDA compilation/CPU tests cannot certify GPU numerical accuracy, races, or scientific calibration.
- No fabricated UTC, array coordinate system, phase center or calibration coefficients.
- Preserve the existing FFT normalization when removing the absolute-epsilon clamp; use an explicit relative response floor/mask rather than silently rescaling images.
- Do not commit the user's pre-existing changes or push anything.

## Review focus

- Missing explicitly requested input must fail, never fall back to another observation.
- NPZ shape, antenna/polarization width, sequence count, nibble representability and destination capacity must agree before copying.
- Distinct channels must remain distinct after conversion to channel-major layout.
- Complex cross-products must be invariant under a common voltage phase rotation.
- Correction grids must preserve the reference FFT response, and binned frequency metadata must match actual channel centers.

## Tasks

- [x] Pin upstream revisions and record provenance of each repaired behavior, including any corrections to the previous review.
- [x] Add stdlib unittest regression tests and a CUDA-header host math test; record expected failures against the unchanged source.
- [x] Correct `ComputeUU` using float intermediates; test real CUDA half arithmetic against analytic complex products.
- [x] Add `src/python/offline_io.py`: validate a four-dimensional complex NPZ, explicit selection, CNib range, shape and count; return channel-major contiguous data with metadata. Integrate at `get_40ms_gulp` and `Get40msGulp`; check capacity and configured sequence count before writes.
- [x] Derive output buffer elements from image size and output channels, with checked arithmetic; test supported configurations and invalid dimensions using actual C++ helpers.
- [x] Extract a pure correction-grid helper and remove absolute-epsilon flattening, retaining normalization and documented low-response masking. Test delta/nontrivial kernels, channel independence, invalid inputs, 64/128 grids and oversampling.
- [x] Correct FITS frequency centers/spacing using `chan_width`; make cross-product plane labels unambiguous; exercise the actual writer with controlled observation metadata.
- [x] Run the whole first-party test suite; compile changed CUDA and host translation units using the existing toolchain without replacing the production binary.
- [x] Obtain an independent final diff review, resolve important findings, and apply verified source/test/docs files back to the original workspace with snapshot conflict checks.
- [x] Document remaining GPU synchronization/autocorrelation/multistream and physical calibration gates explicitly; do not describe this batch as a fully usable scientific release.

## Verification commands

`/home/zdj/.conda/envs/epic310/bin/python -m unittest discover -s tests -v`

CUDA host math test and translation-unit compilation commands are recorded by the test runner/build log. Hardware-dependent tests must report a skip when no device exists; they must not be counted as GPU passes.


## Execution ledger

- Upstream refs verified over GitHub; generalized EPIC cloned read-only at a pinned revision. Comparison distinguishes inherited defects from local regressions.
- Ruling: execute the already-approved repair roadmap as a bounded first batch; preserve original workspace and apply only hash-checked changes from the isolated worktree.
- Initial seven behavioral regressions failed on the unchanged implementation, including real production half expression 0 versus expected ReXY=1. After repair they passed.
- Output descriptor test initially failed because dimension-derived capacity was absent; new method and actual allocation wiring pass its shape cases.
- Five imaging-output tests reproduced the flattened response, missing invalid-kernel rejection, odd-channel issues and 25kHz binned FITS spacing. Corrected code passes analytic expectations.
- Added capacity/sequence/endpoints tests exercise the implemented input boundary. Later frequency-range regressions failed for chan0=2**40 and for signed-frequency multiplication/tail overflow, then passed after guards.
- Ruling: preserve the CUDA FFT 1/64 and NumPy IFFT normalization. Relative low-response pixels are zero-masked; calibrated absolute flux remains unverified. Cost: old image amplitudes and output product labels are intentionally not backward-compatible.
- Independent final review found one Important input-range omission and one Minor missing zero-mask test. Both addressed: overflow tests RED→GREEN; analytic zero-mask/float-bridge coverage added. Final suite: 20 passing tests.
- Both modified production translation units compile; isolated link and help smoke pass. Existing dependency builds are reused. No GPU test was counted as passing.
- Ruling: do not replace the production executable or rewrite old products before GPU/scientific gates. Cost: user must reconfigure/rebuild to pair the new bridge modules and binary; status document gives explicit instructions.
- Remaining gates: GPU synchronization/bounds/multistream/autocorrelation, real time/phase center/calibration, 9197 scientific conversion and operational failure handling. No claim of a complete usable scientific release.

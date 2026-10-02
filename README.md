# LWA EPIC: L-band offline imaging work

This development branch preserves the local L-band adaptation of
[LWA_EPIC](https://github.com/epic-astronomy/LWA_EPIC), repairs its MOFF execution
path, and adds a native FP32 CUDA/cuBLAS FX imager for small arrays.

The starting point is `aiyuan/offline` at
`93aab8c88a756e0c0a3c145ff9e8980611e4d400`. The upstream default CUDA branch is
`phoenix` (`44bb9b43cf8a13a39e572e1a721b5e9473faf63f`); historical `main` is
`0158349f246bd03d78df2f731a3acd21777b9a7a`. This branch does not merge those
different implementations or modify the upstream baselines.

## Start here

- [Current repair status and remaining scientific gates](docs/CORE_REPAIR_STATUS.md)
- [Dual RTX4090 repairs, tests, and MOFF entrypoint](docs/BATCH2_GPU_REPAIR.md)
- [Native FX build, data contract, and two-GPU NPZ runner](docs/FX_BACKEND.md)
- [Algorithm choice, original papers, and benchmark limits](docs/LBAND_ALGORITHM_CHOICE.md)
- [Original upstream README and build instructions](docs/README.md)

The FX path requires explicit physical antenna positions, RF frequencies,
direction vectors, IQ/phase conventions and voltage gains. It produces engineering
NPZ products. The legacy MOFF path requires known positive UTC and signed CNib
voltages; `scripts/run_moff_offline.sh` selects a separately built executable.
No binary, observation data or generated scientific result is included in Git.
Workstation executable paths in the detailed notes describe the validation host;
build the code and configure those paths on another machine.

## Verification

On the validation host, the combined CPU and two-RTX4090 suite passes 46 tests.
The actual MOFF CUDA regression passes 22 cases repeated three times; tested
memcheck/racecheck/synccheck paths report no errors. Single-card, multiple-stream
and dual-card synthetic FITS reconstruction is identical.

```bash
bash src/fx/build.sh
EPIC_FX_TEST_GPU=1 EPIC_FX_TEST_TWO_GPU=1 \
  python3 -m unittest discover -s tests -v
```

GPU tests require two available CUDA devices. Without opt-in, GPU tests are
explicitly skipped; see `tests/gpu/README.md` for the production-kernel tests.
These results do not establish real-observation calibration, full-bandwidth
throughput or scientific readiness. Older experiment scripts and progress notes
are retained as history; use the current repair status for acceptance criteria.

## Existing dependency build fixes

The workstation has two small uncommitted header fixes inside nested submodules.
Their original submodule commits are retained; the fixes are distributed here as
patches instead of references to unpublished dependency commits. After recursive
submodule initialization, apply them once when building the legacy MOFF target:

```bash
git -C src/extern/RaftLib apply --check ../../../patches/raftlib-cstdint.patch
git -C src/extern/RaftLib apply ../../../patches/raftlib-cstdint.patch
git -C src/extern/RaftLib/git-dep/shm apply --check ../../../../../patches/shm-cstdint.patch
git -C src/extern/RaftLib/git-dep/shm apply ../../../../../patches/shm-cstdint.patch
```

The native FX backend is independent of those legacy Raft dependencies.

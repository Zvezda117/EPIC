"""Validated NPZ input for the channel-major, signed-CNib GPU interface."""
import os
from pathlib import Path

import numpy as np


def load_gulp(nstands=16, npols=2, expected_nseqs=None, capacity_bytes=None,
              channel_start=0, channel_count=None, input_path=None):
    explicit = input_path if input_path is not None else os.environ.get("EPIC_GULP_NPZ")
    if explicit is not None:
        path = Path(explicit).expanduser()
        if not explicit or not path.is_file():
            raise FileNotFoundError(f"EPIC_GULP_NPZ is not a file: {explicit!r}")
    else:
        candidates = [Path(name) for name in (
            "data/40ms_128chan_600offset_gulp_c64_virtransit.npz",
            "build/data/40ms_128chan_600offset_gulp_c64_virtransit.npz",
            "data/epic_ready/9197_1s_calib.npz",
        )]
        path = next((p for p in candidates if p.is_file()), None)
        if path is None:
            raise FileNotFoundError("No offline input; set EPIC_GULP_NPZ")

    with np.load(path, allow_pickle=False) as archive:
        data = archive["data"]
        meta = np.asarray(archive["meta"], dtype=np.float64).ravel()
    if meta.size != 2 or not np.isfinite(meta).all():
        raise ValueError("NPZ meta must contain exactly [unix_start_seconds, chan0]")
    if meta[0] < 0 or meta[1] <= 0 or meta[1] != np.floor(meta[1]):
        raise ValueError("NPZ requires non-negative time and a positive integral chan0")
    if data.ndim != 4 or any(size <= 0 for size in data.shape):
        raise ValueError("NPZ data must have non-empty (time, channel, stand, pol) axes")
    if not isinstance(channel_start, (int, np.integer)) or channel_start < 0:
        raise ValueError("Channel start must be a non-negative integer")
    if channel_count is None:
        channel_count = data.shape[1] - channel_start
    if not isinstance(channel_count, (int, np.integer)) or channel_count <= 0 or channel_start + channel_count > data.shape[1]:
        raise ValueError("Requested channel shard is outside the NPZ frequency axis")
    data = data[:, channel_start:channel_start+channel_count, :, :]
    meta = meta.copy()
    meta[1] += channel_start
    cpp_max = np.iinfo(np.int32).max
    # Existing CUDA/host geometry multiplies an int channel by 25 kHz.
    # Bound the last frequency too, before C++ narrows or multiplies it.
    last_channel = meta[1] + data.shape[1] - 1
    if any(size > cpp_max for size in data.shape) or last_channel > cpp_max // 25000:
        raise ValueError("NPZ dimensions or frequencies exceed the C++ integer range")
    if data.shape[2:] != (nstands, npols):
        raise ValueError(f"NPZ stand/pol shape {data.shape[2:]} != compiled {(nstands, npols)}")
    if expected_nseqs is not None and data.shape[0] != expected_nseqs:
        raise ValueError(f"NPZ has {data.shape[0]} sequences, --nts specifies {expected_nseqs}")
    # Each complex value occupies one byte after signed 4+4 bit packing.
    if capacity_bytes is not None and data.size > capacity_bytes:
        raise ValueError(f"CNib input needs {data.size} bytes, buffer has {capacity_bytes}")
    if not np.issubdtype(data.dtype, np.complexfloating):
        raise ValueError("NPZ voltages must have a complex floating dtype")
    if not np.isfinite(data).all():
        raise ValueError("NPZ contains non-finite voltages")
    for component in (data.real, data.imag):
        if np.any(component < -8) or np.any(component > 7) or np.any(component != np.rint(component)):
            raise ValueError("Quantize voltages to integral signed CNib components [-8, 7] before imaging")
    shape = data.shape
    packed_order = np.ascontiguousarray(data.transpose(1, 0, 2, 3))
    return {
        "meta": np.concatenate((meta, [2 * data.size], shape)).astype(np.float64),
        "data": packed_order.ravel(),
        "data_order": "c_maj",
        "source_path": str(path.resolve()),
    }

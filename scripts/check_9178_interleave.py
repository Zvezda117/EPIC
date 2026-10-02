#!/usr/bin/env python3
"""Validate the fast 9178 deinterleaver against the verified UnpackCustom path."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from convert_9178_to_epic import SAMPLES_PER_PACKET_PER_CHANNEL, read_terminal_channel


DATA_FILES = (
    "HBaseband0Spec0_4453.data",
    "HBaseband1Spec1_4453.data",
    "HBaseband2Spec2_4453.data",
    "VBaseband0Spec0_4453.data",
    "VBaseband1Spec1_4453.data",
    "VBaseband2Spec2_4453.data",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="/mnt/data_ssd/zdj/data/9178")
    parser.add_argument(
        "--unpack-dir",
        default="/mnt/data_ssd/zdj/code/jianli/zdj",
        help="Directory containing the verified UnpackCustom.py and unpack.py",
    )
    parser.add_argument("--iq-order", choices=("IQ", "QI"), default="IQ")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sys.path.insert(0, str(Path(args.unpack_dir).resolve()))
    from UnpackCustom import UnpackCustom

    data_dir = Path(args.data_dir)
    all_ok = True
    for name in DATA_FILES:
        path = data_dir / name
        pack_length, channel_count, data_type, file_size = UnpackCustom.onepack_read(
            str(path), DEBUG=False
        )
        _, unpacked = UnpackCustom.read_and_unpack(
            str(path), 0, 0, pack_length, data_type, file_size, channel_count
        )
        print(f"\n{name}")
        for channel in range(channel_count):
            fast = read_terminal_channel(
                path,
                channel,
                n_samples=SAMPLES_PER_PACKET_PER_CHANNEL,
                start_packet=0,
                iq_order=args.iq_order,
            )
            ref = np.asarray(unpacked[channel], dtype=np.complex64)
            ok = fast.shape == ref.shape and np.array_equal(fast, ref)
            all_ok = all_ok and ok
            max_abs = float(np.max(np.abs(fast - ref))) if fast.shape == ref.shape else np.inf
            print(
                f"  ch{channel}: fast {fast.shape}, ref {ref.shape}, "
                f"equal={ok}, max_abs_diff={max_abs:g}, first={fast[:4].tolist()}"
            )
    if not all_ok:
        raise SystemExit("9178 interleave check failed")
    print("\nAll 9178 first-packet channel streams match UnpackCustom.")


if __name__ == "__main__":
    main()

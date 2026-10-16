#!/usr/bin/env python3
"""
Convert 9178 Crab baseband packets into EPIC offline NPZ input.

This converter intentionally avoids the slow packet-by-packet UnpackCustom path.
9178 packets are common-format datatype 13:

  256-byte header + 262144-byte payload
  payload layout: repeated [32 bytes channel 0][32 bytes channel 1]
  each terminal channel stream: int8 IQ byte pairs

The default output is a 4-stand experimental subset compatible with the
currently compiled EPIC binary in this checkout (LWA_SV_NSTANDS=4,
LWA_SV_NPOLS=2). Each selected physical input is duplicated into the two EPIC
polarization slots, so the first experiment uses four non-zero inputs while
still matching EPIC's dual-pol memory layout.
"""

from __future__ import annotations

import argparse
import math
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src/python"))
from packet9178 import audit_file, audit_group, chunk_epoch


PACKET_BYTES = 262400
HEADER_BYTES = 256
PAYLOAD_BYTES = PACKET_BYTES - HEADER_BYTES
INTERLEAVE_BYTES = 32
TERMINAL_CHANNELS = 2
SAMPLES_PER_PACKET_PER_CHANNEL = PAYLOAD_BYTES // TERMINAL_CHANNELS // 2
SAMPLE_RATE_HZ = 250_000_000.0
SEQ_SECONDS = 40e-6
SEQ_SAMPLES = int(round(SAMPLE_RATE_HZ * SEQ_SECONDS))
EPIC_CHAN_BW_HZ = 25_000.0
CENTER_FREQ_MHZ = 1355.0


@dataclass(frozen=True)
class InputSelection:
    filename: str
    terminal_channel: int
    antenna: str
    polarization: str
    epic_index: int
    epic_pol: int | None = None


EPIC4_INPUTS = (
    InputSelection("HBaseband0Spec0_4453.data", 0, "北4", "V", 0),
    InputSelection("HBaseband0Spec0_4453.data", 1, "北3", "V", 1),
    InputSelection("HBaseband1Spec1_4453.data", 0, "东4", "H", 2),
    InputSelection("HBaseband1Spec1_4453.data", 1, "东3", "V", 3),
)


FULL10_INPUTS = (
    InputSelection("HBaseband0Spec0_4453.data", 0, "北4", "V", 0, 1),
    InputSelection("HBaseband0Spec0_4453.data", 1, "北3", "V", 1, 1),
    InputSelection("HBaseband1Spec1_4453.data", 0, "东4", "H", 2, 0),
    InputSelection("HBaseband1Spec1_4453.data", 1, "东3", "V", 3, 1),
    InputSelection("HBaseband2Spec2_4453.data", 0, "东5", "H", 4, 0),
    InputSelection("HBaseband2Spec2_4453.data", 1, "东6", "H", 5, 0),
    InputSelection("VBaseband0Spec0_4453.data", 0, "东4", "V", 2, 1),
    InputSelection("VBaseband0Spec0_4453.data", 1, "东5", "V", 4, 1),
    InputSelection("VBaseband1Spec1_4453.data", 0, "东9", "H", 6, 0),
    InputSelection("VBaseband1Spec1_4453.data", 1, "东10", "V", 7, 1),
    InputSelection("VBaseband2Spec2_4453.data", 0, "东7", "H", 8, 0),
    InputSelection("VBaseband2Spec2_4453.data", 1, "东8", "V", 9, 1),
)


def validate_packet_header(path: Path) -> tuple[int, int, int]:
    with path.open("rb") as fh:
        header = np.frombuffer(fh.read(32), dtype="<u4")
    if header.size < 8:
        raise ValueError(f"{path} is too small to contain a 9178 header")
    head_tag = int(header[0])
    packet_bytes = int(header[2])
    channel_count = int(header[4] & 0xFFFF)
    data_type = int(header[6] & 0xFF)
    if head_tag != 0x18EFDC01:
        raise ValueError(f"{path.name}: unexpected header tag 0x{head_tag:08x}")
    if packet_bytes != PACKET_BYTES:
        raise ValueError(f"{path.name}: packet length {packet_bytes}, expected {PACKET_BYTES}")
    if channel_count != TERMINAL_CHANNELS:
        raise ValueError(f"{path.name}: channel count {channel_count}, expected 2")
    if data_type != 13:
        raise ValueError(f"{path.name}: datatype {data_type}, expected 13 int8 IQ")
    return packet_bytes, channel_count, data_type


def read_terminal_channel(
    path: Path,
    terminal_channel: int,
    n_samples: int,
    start_packet: int = 0,
    start_sample: int = 0,
    iq_order: str = "IQ",
) -> np.ndarray:
    if terminal_channel not in (0, 1):
        raise ValueError("terminal_channel must be 0 or 1")
    if start_sample < 0:
        raise ValueError("start_sample must be non-negative")
    validate_packet_header(path)

    packet_offset = start_sample // SAMPLES_PER_PACKET_PER_CHANNEL
    sample_offset = start_sample % SAMPLES_PER_PACKET_PER_CHANNEL
    first_packet = start_packet + packet_offset
    packets_needed = math.ceil((sample_offset + n_samples) / SAMPLES_PER_PACKET_PER_CHANNEL)
    file_packets = path.stat().st_size // PACKET_BYTES
    if first_packet + packets_needed > file_packets:
        raise ValueError(
            f"{path.name}: need packets [{first_packet}, {first_packet + packets_needed}), "
            f"but file has {file_packets} packets"
        )

    audit_file(path, first_packet, packets_needed)

    mm = np.memmap(
        path,
        mode="r",
        dtype=np.int8,
        offset=first_packet * PACKET_BYTES,
        shape=(packets_needed, PACKET_BYTES),
    )
    payload = np.asarray(mm[:, HEADER_BYTES:])
    blocks = payload.reshape(packets_needed, -1, TERMINAL_CHANNELS, INTERLEAVE_BYTES)
    channel_bytes = blocks[:, :, terminal_channel, :].reshape(-1)
    if sample_offset:
        channel_bytes = channel_bytes[sample_offset * 2 :]
    pairs = channel_bytes[: n_samples * 2].reshape(-1, 2).astype(np.float32)

    if iq_order.upper() == "IQ":
        real = pairs[:, 0]
        imag = pairs[:, 1]
    elif iq_order.upper() == "QI":
        real = pairs[:, 1]
        imag = pairs[:, 0]
    else:
        raise ValueError("--iq-order must be IQ or QI")

    return (real + 1j * imag).astype(np.complex64, copy=False)


def channelize_40us_sequences(
    samples: np.ndarray,
    nseqs: int,
    nchan: int,
    channel_offset: int,
    dc_mask_chans: int,
) -> np.ndarray:
    needed = nseqs * SEQ_SAMPLES
    if samples.size < needed:
        raise ValueError(f"need {needed} samples, got {samples.size}")
    chunked = samples[:needed].reshape(nseqs, SEQ_SAMPLES)
    spectra = np.fft.fftshift(np.fft.fft(chunked, axis=1), axes=1)

    center = SEQ_SAMPLES // 2
    if dc_mask_chans > 0:
        spectra[:, center - dc_mask_chans : center + dc_mask_chans + 1] = 0

    first_bin = center - nchan // 2 + channel_offset
    last_bin = first_bin + nchan
    if first_bin < 0 or last_bin > SEQ_SAMPLES:
        raise ValueError(
            f"selected bins [{first_bin}, {last_bin}) fall outside FFT size {SEQ_SAMPLES}"
        )
    return spectra[:, first_bin:last_bin].astype(np.complex64, copy=False)


def quantize_for_epic_cnib(
    data: np.ndarray, target: float = 6.0, fixed_scale: float | None = None
) -> np.ndarray:
    """Scale complex spectra into EPIC's signed 4-bit CNib range."""
    max_component = max(float(np.max(np.abs(data.real))), float(np.max(np.abs(data.imag))))
    if max_component <= 0 or not np.isfinite(max_component):
        raise ValueError("cannot quantize all-zero or non-finite data")
    scale = fixed_scale if fixed_scale is not None else target / max_component
    if scale <= 0 or not np.isfinite(scale):
        raise ValueError("--cnib-scale must be positive and finite")
    real = np.rint(np.clip(data.real * scale, -7, 7))
    imag = np.rint(np.clip(data.imag * scale, -7, 7))
    print(
        f"CNib quantization scale: {scale:.6g} "
        f"(max component {max_component:.6g})"
    )
    return (real + 1j * imag).astype(np.complex64, copy=False)


def build_epic_gulp(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray]:
    data_dir = Path(args.data_dir)
    nseqs = int(round(args.integration_ms / (SEQ_SECONDS * 1000.0)))
    if nseqs <= 0:
        raise ValueError("--integration-ms must be positive")
    n_samples = nseqs * SEQ_SAMPLES
    start_sample = args.start_seq * SEQ_SAMPLES

    selections = EPIC4_INPUTS if args.array_mode == "epic4" else FULL10_INPUTS
    args.packet_audit = audit_group(sorted({data_dir / item.filename for item in selections}))
    if args.array_mode == "epic4":
        nstands = 4
    elif args.array_mode == "full10":
        nstands = 10
    else:
        nstands = 16
    epic = np.zeros((nseqs, args.channels, nstands, 2), dtype=np.complex64)

    print(
        f"9178 -> EPIC ({args.array_mode}): "
        f"{args.integration_ms:g} ms = {nseqs} x 40 us sequences"
    )
    print(f"FFT size per sequence: {SEQ_SAMPLES}, output channels: {args.channels}")
    print(f"Input IQ order: {args.iq_order}")
    for selection in selections:
        path = data_dir / selection.filename
        target_pol = (
            "both" if selection.epic_pol is None else ("H" if selection.epic_pol == 0 else "V")
        )
        print(
            f"  read {selection.filename} ch{selection.terminal_channel}: "
            f"{selection.antenna}{selection.polarization} -> "
            f"EPIC stand {selection.epic_index} pol {target_pol}"
        )
        samples = read_terminal_channel(
            path,
            selection.terminal_channel,
            n_samples=n_samples,
            start_packet=args.start_packet,
            start_sample=start_sample,
            iq_order=args.iq_order,
        )
        spectra = channelize_40us_sequences(
            samples,
            nseqs=nseqs,
            nchan=args.channels,
            channel_offset=args.channel_offset,
            dc_mask_chans=args.dc_mask_chans,
        )
        if selection.epic_pol is None:
            epic[:, :, selection.epic_index, 0] = spectra
            if args.fill_missing_pol == "duplicate":
                epic[:, :, selection.epic_index, 1] = spectra
        else:
            epic[:, :, selection.epic_index, selection.epic_pol] = spectra

    if args.quantize_cnib:
        epic = quantize_for_epic_cnib(epic, fixed_scale=args.cnib_scale)

    center_chan = int(round(args.center_freq_mhz * 1e6 / EPIC_CHAN_BW_HZ))
    chan0 = center_chan - args.channels // 2 + args.channel_offset
    meta = np.array([chunk_epoch(args.tstart, args.start_packet, args.start_seq), chan0], dtype=np.float64)
    return epic, meta


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert /mnt/data_ssd/zdj/data/9178 baseband into EPIC NPZ input"
    )
    parser.add_argument("--data-dir", default="/mnt/data_ssd/zdj/data/9178")
    parser.add_argument(
        "--output",
        default="data/epic_ready/9178_crab_epic4_40ms_128chan.npz",
        help="Output path inside the LWA_EPIC project by default",
    )
    parser.add_argument("--channels", type=int, default=128)
    parser.add_argument("--integration-ms", type=float, default=40.0)
    parser.add_argument("--start-packet", type=int, default=0)
    parser.add_argument(
        "--start-seq",
        type=int,
        default=0,
        help="40us sequence offset from --start-packet; enables exact chunking",
    )
    parser.add_argument(
        "--array-mode",
        choices=("epic4", "full10", "full16"),
        default="epic4",
        help=(
            "epic4 is a smoke test; full10 is the physical array data shape; "
            "full16 pads full10 to EPIC's original 32-input layout"
        ),
    )
    parser.add_argument(
        "--channel-offset",
        type=int,
        default=0,
        help="Offset in 25 kHz FFT bins relative to the center-frequency window",
    )
    parser.add_argument("--center-freq-mhz", type=float, default=CENTER_FREQ_MHZ)
    parser.add_argument(
        "--dc-mask-chans",
        type=int,
        default=20,
        help="Mask +/- this many 25 kHz bins around baseband DC",
    )
    parser.add_argument("--iq-order", choices=("IQ", "QI"), default="IQ")
    parser.add_argument(
        "--fill-missing-pol",
        choices=("zero", "duplicate"),
        default="duplicate",
        help="The current EPIC binary is dual-pol; duplicate gives four active stands",
    )
    parser.add_argument(
        "--no-quantize-cnib",
        dest="quantize_cnib",
        action="store_false",
        help="Leave floating spectra unscaled; useful only if EPIC input is changed away from CNib",
    )
    parser.add_argument(
        "--cnib-scale",
        type=float,
        default=None,
        help="Use a fixed linear scale before signed 4-bit CNib clipping",
    )
    parser.set_defaults(quantize_cnib=True)
    parser.add_argument("--tstart", type=float, default=0.0, help="Confirmed Unix UTC of sample 0 in each input file; 0 means unknown. Packet/sequence offsets are added.")
    parser.add_argument(
        "--no-compress",
        dest="compress",
        action="store_false",
        help="Use np.savez instead of np.savez_compressed for faster large chunk output",
    )
    parser.set_defaults(compress=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    epic, meta = build_epic_gulp(args)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    save_npz = np.savez_compressed if args.compress else np.savez
    save_npz(output, data=epic, meta=meta, packet_audit_json=np.asarray(json.dumps(args.packet_audit, ensure_ascii=False)))
    print(f"Saved {output}")
    print(f"  data shape: {epic.shape}, dtype: {epic.dtype}")
    print(f"  meta: tstart={meta[0]}, chan0={int(meta[1])}")
    print(
        "Run EPIC with matching timing, for example: "
        f"--nts {epic.shape[0]} --seq_accum {int(round(args.integration_ms))} "
        "--imageres 0.02"
    )


if __name__ == "__main__":
    main()

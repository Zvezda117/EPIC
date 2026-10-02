#!/usr/bin/env python3
"""Explicit-geometry engineering NPZ imaging; no FITS/UTC inference."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src/python'))
from lband_fx import FXImager


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True, type=Path, help='NPZ complex data[T,C,N,2]')
    parser.add_argument('--geometry', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path, help='new output directory; never overwrite')
    parser.add_argument('--devices', default='1', help='distinct CUDA indices, e.g. 0,1')
    parser.add_argument('--channel-tile', type=int, default=16)
    parser.add_argument('--window-sequences', type=int, default=1000)
    parser.add_argument('--coordinate-frame', required=True, help='caller-declared common Cartesian frame')
    parser.add_argument('--iq-convention', required=True, choices=['I+iQ'])
    parser.add_argument('--voltage-convention', required=True, choices=['phase-tracked', 'raw'],
                        help='raw adds positive phase-center tracking gain before 3D steering')
    parser.add_argument('--remove-autocorr', action='store_true')
    args = parser.parse_args()
    devices = [int(value) for value in args.devices.split(',')]
    if not devices or any(value < 0 for value in devices) or len(set(devices)) != len(devices):
        raise ValueError('--devices must contain distinct nonnegative indices')
    if args.channel_tile < 1 or args.window_sequences < 1:
        raise ValueError('channel tile and window sequences must be positive')
    if not args.coordinate_frame.strip():
        raise ValueError('coordinate frame must be explicit')
    with np.load(args.input, allow_pickle=False) as source:
        data = source['data']  # NPZ decompression is full-array CPU loading.
    if data.ndim != 4 or data.shape[-1] != 2 or min(data.shape) < 1 or not np.iscomplexobj(data):
        raise ValueError('input data must be complex [T,C,N,2] with nonempty dimensions')
    if not np.isfinite(data).all():
        raise ValueError('input data must be finite')
    with np.load(args.geometry, allow_pickle=False) as geometry:
        pos = np.asarray(geometry['positions_m'], dtype=np.float64)
        freq = np.asarray(geometry['frequencies_hz'], dtype=np.float64)
        dirs = np.asarray(geometry['directions'], dtype=np.float64)
        center = np.asarray(geometry['phase_center'], dtype=np.float64)
        gain = np.asarray(geometry['gains'], dtype=np.complex64) if 'gains' in geometry else np.ones((data.shape[1], data.shape[2], 2), np.complex64)
    t, c, n, _ = data.shape
    if pos.shape != (n, 3) or freq.shape != (c,) or gain.shape != (c, n, 2):
        raise ValueError('geometry positions/frequencies/gains do not match input channels/antenna order')
    if (dirs.ndim != 2 or dirs.shape[1] != 3 or dirs.shape[0] < 1 or center.shape != (3,)
            or not all(np.isfinite(a).all() for a in (pos, freq, dirs, center, gain))
            or (freq <= 0).any()
            or not np.allclose(np.linalg.norm(dirs, axis=1), 1., rtol=0., atol=1e-6)
            or not np.isclose(np.linalg.norm(center), 1., rtol=0., atol=1e-6)):
        raise ValueError('geometry requires finite metre positions, positive Hz frequencies and unit directions/center')
    # For raw inputs: exp(+ik r.s0) gain times exp(+ik r.(s-s0)) = exp(+ik r.s).
    if args.voltage_convention == 'raw':
        tracking = np.exp(2j*np.pi*freq[:, None]*(pos@center)[None]/299792458.)
        gain = (gain*tracking[..., None]).astype(np.complex64)
    args.output.mkdir(parents=True, exist_ok=False)
    metadata = dict(artifact='engineering_FX_products', normalization='mean_over_valid_sequences',
                    products=['XX', 'YY', 'Re(XY*)', 'Im(XY*)'],
                    time_basis='sequence_indices_no_UTC', coordinate_frame=args.coordinate_frame,
                    frequency_convention='explicit_positive_RF_Hz_in_input_channel_order',
                    iq_convention=args.iq_convention, voltage_convention=args.voltage_convention,
                    steering='exp(+2*pi*i*f*r_dot_(s-s0)/c)',
                    raw_tracking_gain='exp(+2*pi*i*f*r_dot_s0/c)' if args.voltage_convention == 'raw' else 'none',
                    gain_convention='voltage_multiply', remove_physical_autocorr=args.remove_autocorr)
    tiles = [(lo, min(lo+args.channel_tile, c)) for lo in range(0, c, args.channel_tile)]

    def worker(device, assigned):
        records = []
        for lo, hi in assigned:
            context = None
            length = None
            try:
                for window, begin in enumerate(range(0, t, args.window_sequences)):
                    end = min(begin+args.window_sequences, t)
                    if end-begin != length:
                        if context is not None:
                            context.close()  # do not retain full + tail contexts together
                        length = end-begin
                        context = FXImager(pos, freq[lo:hi], dirs, center,
                                           gains=gain[lo:hi], ntime=length, device=device,
                                           remove_autocorr=args.remove_autocorr)
                    image = context.image(data[begin:end, lo:hi])
                    filename = f'window_{window:06d}_channels_{lo:06d}_{hi:06d}.npz'
                    chunk = dict(file=filename, window_index=window, sequence_start=begin,
                                 sequence_end=end, sequence_count=end-begin,
                                 channel_start=lo, channel_end=hi, device=device)
                    np.savez(args.output/filename, image=image, frequencies_hz=freq[lo:hi],
                             directions=dirs, phase_center=center,
                             metadata_json=np.asarray(json.dumps(metadata | chunk)))
                    records.append(chunk)
            finally:
                if context is not None:
                    context.close()
        return records

    with ThreadPoolExecutor(max_workers=len(devices)) as executor:
        jobs = [executor.submit(worker, device, tiles[i::len(devices)])
                for i, device in enumerate(devices)]
        chunks = [record for job in jobs for record in job.result()]
    chunks.sort(key=lambda value: (value['window_index'], value['channel_start']))
    manifest = metadata | dict(input=str(args.input.resolve()), geometry=str(args.geometry.resolve()),
                               input_shape=list(data.shape), channel_tile=args.channel_tile,
                               window_sequences=args.window_sequences, devices=devices, chunks=chunks)
    (args.output/'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({'output_directory': str(args.output), 'chunks': len(chunks),
                      'status': 'engineering_products_no_UTC_or_FITS'}))


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        print(f'FX runner failed: {exc}', file=sys.stderr)
        raise SystemExit(1)

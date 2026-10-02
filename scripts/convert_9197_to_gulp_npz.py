#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
9197 数据转换：仅做「原始 .data → 校准后 1s 帧 npz」，不成像。

- 每个 .data 文件完整读取（~4.28s），不截断
- 多文件按时间顺序输出：frame_00000.npz, frame_00001.npz, ...
- 每帧 = 1s = 25×40ms，shape (25, 1024, 4, 2)，供 EPIC 单次调用成图
- 校准：CasA 几何时延 + DC mask ±0.5 MHz；1024 通道，全 250 MHz
"""
import numpy as np
import os
import sys
from pathlib import Path
from tqdm import tqdm

DATA_DIR = "/mnt/DATA/存储阵列/9197_Base2_ADC16_26-01-18_19-16-51/"
ANTENNA_FILES = {
    "N3": "HBaseband2Spec2", "N4": "VBaseband0Spec0",
    "N5": "VBaseband1Spec1", "N6": "VBaseband2Spec2",
}
SAMPLE_RATE = 250e6
EPIC_CHAN_BW = 25.0e3
HEADER_SIZE = 256
DATA_SIZE = 65536
PACK_SIZE = HEADER_SIZE + DATA_SIZE
GULP_MS = 40
GULP_SAMPLES = int(SAMPLE_RATE * GULP_MS / 1000)
SEQ_PER_1S = 25   # 1s = 25 × 40ms
OUTPUT_CHANNELS = 1024
C_LIGHT = 299792458.0
CASA_RA_DEG, CASA_DEC_DEG = 350.85, 58.815
STATION_LAT_DEG = 36.6578
CENTER_FREQ_MHZ = 1355.0
ANT_ENU = {'N3': (0.0, 0.0), 'N4': (21.06, 0.035), 'N5': (-2.661, -49.898), 'N6': (23.804, -49.902)}


def geometric_delays_casa():
    dec = np.deg2rad(CASA_DEC_DEG)
    lat = np.deg2rad(STATION_LAT_DEG)
    ha = 0
    sin_alt = np.sin(dec) * np.sin(lat) + np.cos(dec) * np.cos(lat) * np.cos(ha)
    alt = np.arcsin(sin_alt)
    cos_az = np.clip((np.sin(dec) - np.sin(alt) * np.sin(lat)) / (np.cos(alt) * np.cos(lat) + 1e-12), -1, 1)
    az = np.arccos(cos_az)
    s_enu = np.array([np.cos(alt) * np.sin(az), np.cos(alt) * np.cos(az), np.sin(alt)])
    return np.array([np.dot(np.array([e, n, 0.0]), s_enu) / C_LIGHT for (e, n) in ANT_ENU.values()])


def load_one_antenna_full(filepath):
    size = os.path.getsize(filepath)
    num_packs = size // PACK_SIZE
    out = []
    with open(filepath, 'rb') as f:
        for _ in tqdm(range(num_packs), desc=Path(filepath).name, leave=False):
            f.seek(HEADER_SIZE, 1)
            raw = np.frombuffer(f.read(DATA_SIZE), dtype=np.int8)
            I, Q = raw[0::2].astype(np.float32), raw[1::2].astype(np.float32)
            out.append(I + 1j * Q)
    return np.concatenate(out)


def apply_dc_mask(fft_data, dc_mhz=0.5):
    nfft = len(fft_data)
    nbins = int(dc_mhz * 1e6 / (SAMPLE_RATE / nfft))
    fft_data[0:nbins] = 0
    fft_data[-nbins:] = 0
    return fft_data


def process_one_file_to_gulps(file_index, data_dir):
    """读单个文件全部数据，校准后得到 (ngulps, 1024, 4, 2)。"""
    ant_names = ['N3', 'N4', 'N5', 'N6']
    delays = geometric_delays_casa()
    freq_bins = GULP_SAMPLES
    bins_per_ch = freq_bins // OUTPUT_CHANNELS
    chan_freqs = (np.arange(OUTPUT_CHANNELS) + 0.5) * (SAMPLE_RATE / GULP_SAMPLES) * bins_per_ch

    ant_data = {}
    for name in ant_names:
        path = os.path.join(data_dir, f"{ANTENNA_FILES[name]}_{file_index}.data")
        if not os.path.isfile(path):
            raise FileNotFoundError(path)
        ant_data[name] = load_one_antenna_full(path)
    nsamples = min(len(ant_data[n]) for n in ant_names)
    nsamples = (nsamples // GULP_SAMPLES) * GULP_SAMPLES
    ngulps = nsamples // GULP_SAMPLES

    voltages = np.zeros((nsamples, 4), dtype=np.complex64)
    for i, n in enumerate(ant_names):
        voltages[:, i] = ant_data[n][:nsamples]

    epic = np.zeros((ngulps, OUTPUT_CHANNELS, 4, 2), dtype=np.complex64)
    for ig in tqdm(range(ngulps), desc=f"  file{file_index}", leave=False):
        start = ig * GULP_SAMPLES
        end = start + GULP_SAMPLES
        for iant in range(4):
            fft = np.fft.fft(voltages[start:end, iant])
            apply_dc_mask(fft, 0.5)
            pos = fft[:freq_bins]
            for ich in range(OUTPUT_CHANNELS):
                b0, b1 = ich * bins_per_ch, (ich + 1) * bins_per_ch
                v = np.mean(pos[b0:b1]) * np.exp(-2j * np.pi * chan_freqs[ich] * delays[iant])
                epic[ig, ich, iant, 0] = epic[ig, ich, iant, 1] = v
    return epic


def main():
    file_indices = [int(x) for x in sys.argv[1:] if x.isdigit()]
    if not file_indices:
        file_indices = [0]
    out_dir = os.environ.get("GULP_NPZ_DIR", "data/epic_ready/gulps_9197")
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 60)
    print("  9197 → 1s 帧 npz（仅转换+校准，不成图）")
    print("=" * 60)
    print(f"  文件索引: {file_indices}  输出目录: {out_dir}")
    print(f"  通道: {OUTPUT_CHANNELS}  带宽: 250 MHz  每帧: 1s (25×40ms)")
    print("=" * 60)

    global_idx = 0
    for fi in file_indices:
        g = process_one_file_to_gulps(fi, DATA_DIR)
        # 每 25 个 40ms gulp 合并为 1 帧 (1s)
        ngulps = g.shape[0]
        n_frames = ngulps // SEQ_PER_1S
        for iframe in range(n_frames):
            start = iframe * SEQ_PER_1S
            end = start + SEQ_PER_1S
            one = g[start:end, :, :, :].copy()  # (25, 1024, 4, 2)
            tstart = 0.0
            chan0 = int(round(CENTER_FREQ_MHZ * 1e6 / EPIC_CHAN_BW))
            meta = np.array([tstart, chan0], dtype=np.float64)
            path = os.path.join(out_dir, f"frame_{global_idx:05d}.npz")
            np.savez_compressed(path, data=one, meta=meta)
            global_idx += 1
        print(f"  文件 {fi}: {ngulps} gulps → {n_frames} 帧 (1s/帧) → frame_{global_idx - n_frames:05d}.npz .. frame_{global_idx - 1:05d}.npz")

    print(f"\n  共 {global_idx} 个 1s 帧 npz，供 EPIC 并行多次调用成图后拼接视频。")
    print("  下一步: 运行 run_epic_video.sh 或 run_epic_video_parallel.sh")
    return global_idx


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
9197 → EPIC 输入：单帧 1s 积分 + CasA 几何时延校准

- 每 gulp = 25 × 40ms = 1s（25 个 sequence）
- 输出 shape: (25, 1024, 4, 2) → EPIC 按 nseqs=25、1024 通道处理
- 校准：CasA 几何时延，频域相位校正
- 全 250 MHz 带宽、中心 1355 MHz
"""
import numpy as np
import os
from tqdm import tqdm

C_LIGHT = 299792458.0  # m/s
CENTER_FREQ = 1355.0  # MHz
EPIC_CHAN_BW = 25.0e3
SAMPLE_RATE = 250e6
HEADER_SIZE = 256
DATA_SIZE = 65536
PACK_SIZE = HEADER_SIZE + DATA_SIZE
GULP_MS = 40
GULP_SAMPLES = int(SAMPLE_RATE * GULP_MS / 1000)
OUTPUT_CHANNELS = 1024
SEQ_PER_1S = 25  # 1s = 25 × 40ms

DATA_DIR = "/mnt/DATA/存储阵列/9197_Base2_ADC16_26-01-18_19-16-51/"
ANTENNA_FILES = {
    "N3": "HBaseband2Spec2", "N4": "VBaseband0Spec0",
    "N5": "VBaseband1Spec1", "N6": "VBaseband2Spec2",
}
# CasA J2000
CASA_RA_DEG = 350.85
CASA_DEC_DEG = 58.815
# 站点近似 (北区)
STATION_LAT_DEG = 36.6578
STATION_LON_DEG = 117.4903

def load_antenna_raw(filepath, num_packs):
    out = []
    with open(filepath, 'rb') as f:
        for _ in range(num_packs):
            f.seek(HEADER_SIZE, 1)
            raw = np.frombuffer(f.read(DATA_SIZE), dtype=np.int8)
            I = raw[0::2].astype(np.float32)
            Q = raw[1::2].astype(np.float32)
            out.append(I + 1j * Q)
    return np.concatenate(out)

def geometric_delays_casa_enu():
    """北3/北4/北5/北6 相对北3 的 CasA 几何时延（秒）。ENU 近似。"""
    # ENU 米 (东, 北)，相对北3
    enu = {
        'N3': (0.0, 0.0),
        'N4': (21.06, 0.035),
        'N5': (-2.661, -49.898),
        'N6': (23.804, -49.902),
    }
    dec = np.deg2rad(CASA_DEC_DEG)
    ra = np.deg2rad(CASA_RA_DEG)
    lat = np.deg2rad(STATION_LAT_DEG)
    # 源方向 ENU (简化: 子午圈)
    ha = 0
    sin_alt = np.sin(dec) * np.sin(lat) + np.cos(dec) * np.cos(lat) * np.cos(ha)
    alt = np.arcsin(sin_alt)
    cos_az = np.clip((np.sin(dec) - np.sin(alt) * np.sin(lat)) / (np.cos(alt) * np.cos(lat) + 1e-12), -1, 1)
    az = np.arccos(cos_az)
    s_E = np.cos(alt) * np.sin(az)
    s_N = np.cos(alt) * np.cos(az)
    s_enu = np.array([s_E, s_N, np.sin(alt)])
    delays = []
    for name in ['N3', 'N4', 'N5', 'N6']:
        e, n = enu[name]
        delta_enu = np.array([e, n, 0.0])  # U=0 近似
        delays.append(np.dot(delta_enu, s_enu) / C_LIGHT)
    return np.array(delays)

def apply_dc_mask(fft_data, dc_mask_mhz=0.5, sample_rate=SAMPLE_RATE):
    nfft = len(fft_data)
    res = sample_rate / nfft
    nbins = int(dc_mask_mhz * 1e6 / res)
    fft_data[0:nbins] = 0
    fft_data[-nbins:] = 0
    return fft_data

def run(out_npz_path, data_dir=DATA_DIR, file_index=0):
    ant_names = ['N3', 'N4', 'N5', 'N6']
    # 需要 1s = 25 * 40ms → 25 * GULP_SAMPLES 采样
    need_samples = SEQ_PER_1S * GULP_SAMPLES
    need_packs = (need_samples + DATA_SIZE // 2 - 1) // (DATA_SIZE // 2)

    print("【1】几何时延 (CasA)")
    delays = geometric_delays_casa_enu()
    for i, n in enumerate(ant_names):
        print(f"  {n}: τ = {delays[i]*1e9:+.2f} ns")

    print("\n【2】读取 1s 原始数据")
    ant_data = {}
    for name in ant_names:
        path = os.path.join(data_dir, f"{ANTENNA_FILES[name]}_{file_index}.data")
        size = os.path.getsize(path)
        n_pack = size // PACK_SIZE
        n_pack = min(n_pack, need_packs)
        ant_data[name] = load_antenna_raw(path, n_pack)
        print(f"  {name}: {len(ant_data[name]):,} samples")
    nsamples = min(len(ant_data[n]) for n in ant_names)
    nsamples = (nsamples // GULP_SAMPLES) * GULP_SAMPLES
    if nsamples < SEQ_PER_1S * GULP_SAMPLES:
        raise SystemExit(f"数据不足 1s: 仅有 {nsamples/SAMPLE_RATE:.2f} s")

    voltages = np.zeros((nsamples, 4), dtype=np.complex64)
    for i, n in enumerate(ant_names):
        voltages[:, i] = ant_data[n][:nsamples]

    print("\n【3】频域时延校准 + DC mask + 通道合并（全 250 MHz，1024 通道）")
    fft_size = GULP_SAMPLES
    freq_bins = fft_size  # I/Q 复数 FFT → 全 0~250 MHz
    bins_per_ch = freq_bins // OUTPUT_CHANNELS
    chan_freqs = (np.arange(OUTPUT_CHANNELS) + 0.5) * (SAMPLE_RATE / fft_size) * bins_per_ch

    nseqs = min(SEQ_PER_1S, nsamples // GULP_SAMPLES)
    epic = np.zeros((nseqs, OUTPUT_CHANNELS, 4, 2), dtype=np.complex64)

    for seq in tqdm(range(nseqs), desc="  Sequence"):
        start = seq * GULP_SAMPLES
        end = start + GULP_SAMPLES
        for iant in range(4):
            g = voltages[start:end, iant]
            fft = np.fft.fft(g)
            apply_dc_mask(fft, 0.5, SAMPLE_RATE)
            pos = fft[:freq_bins]  # 全 0~250 MHz
            for ich in range(OUTPUT_CHANNELS):
                b0 = ich * bins_per_ch
                b1 = b0 + bins_per_ch
                v = np.mean(pos[b0:b1])
                # 相位校正: V *= exp(-2π i f τ)
                phase_corr = np.exp(-2j * np.pi * chan_freqs[ich] * delays[iant])
                epic[seq, ich, iant, 0] = v * phase_corr
                epic[seq, ich, iant, 1] = v * phase_corr  # 单极化复制

    print(f"\n【4】输出 EPIC npz: shape {epic.shape}")
    # EPIC 期望 meta: 至少 [tstart, chan0]，get_40ms_gulp 会拼上 size 和 shape
    tstart = 0.0
    chan0 = int(round(CENTER_FREQ * 1e6 / EPIC_CHAN_BW))
    meta = np.array([tstart, chan0], dtype=np.float64)
    os.makedirs(os.path.dirname(out_npz_path) or '.', exist_ok=True)
    np.savez_compressed(out_npz_path, data=epic, meta=meta)

    # 复制到 EPIC 默认读取路径（供 get_40ms_gulp 使用）
    epic_default = "build/data/40ms_128chan_600offset_gulp_c64_virtransit.npz"  # 文件名固定，内容为 1024ch
    if os.path.exists(os.path.dirname(epic_default)):
        import shutil
        shutil.copy(out_npz_path, epic_default)
        print(f"  已复制到: {epic_default}")
    print("  完成。")
    return out_npz_path

if __name__ == "__main__":
    import sys
    out = sys.argv[1] if len(sys.argv) > 1 else "data/epic_ready/9197_1s_calib.npz"
    run(out)

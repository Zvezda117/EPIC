#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
9197 多文件、全数据、校准、全带宽 → 视频

- 数据：250 MHz 带宽，中心频率 1355 MHz，I/Q 复数 250 Msps
- 每个 .data 文件：完整读取（~4.28s），不截断
- 多文件按时间顺序拼接为连续帧
- 校准：CasA 几何时延 + DC mask ±0.5 MHz（基带低/高端）
- 1024 通道，全 250 MHz 基带带宽，40 ms/帧
- 用 Python 脏图逐帧成图（EPIC 离线仅能处理单 gulp，无法直接出多帧视频）
"""
import numpy as np
import os
import sys
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA_DIR = "/mnt/DATA/存储阵列/9197_Base2_ADC16_26-01-18_19-16-51/"
ANTENNA_FILES = {
    "N3": "HBaseband2Spec2", "N4": "VBaseband0Spec0",
    "N5": "VBaseband1Spec1", "N6": "VBaseband2Spec2",
}
# 9197 数据：250 MHz 带宽，中心 1355 MHz，I/Q 复数 250 Msps
SAMPLE_RATE = 250e6
CENTER_FREQ_MHZ = 1355.0
HEADER_SIZE = 256
DATA_SIZE = 65536
PACK_SIZE = HEADER_SIZE + DATA_SIZE
GULP_MS = 40
GULP_SAMPLES = int(SAMPLE_RATE * GULP_MS / 1000)
OUTPUT_CHANNELS = 1024
C_LIGHT = 299792458.0
CASA_RA_DEG, CASA_DEC_DEG = 350.85, 58.815
STATION_LAT_DEG = 36.6578
# 天线 ENU 相对北3 (米)
ANT_ENU = {'N3': (0.0, 0.0), 'N4': (21.06, 0.035), 'N5': (-2.661, -49.898), 'N6': (23.804, -49.902)}
WAVELENGTH = C_LIGHT / (CENTER_FREQ_MHZ * 1e6)


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


def get_baselines():
    pos = np.array([ANT_ENU['N3'], ANT_ENU['N4'], ANT_ENU['N5'], ANT_ENU['N6']])
    bl = []
    for i in range(4):
        for j in range(i + 1, 4):
            u = (pos[j, 0] - pos[i, 0]) / WAVELENGTH
            v = (pos[j, 1] - pos[i, 1]) / WAVELENGTH
            bl.append((i, j, u, v))
    return bl


def process_one_file_to_gulps(file_index, data_dir):
    """读单个文件全部数据，校准后得到 (ngulps, 1024, 4, 2)。基带全 250 MHz（I/Q 复数 FFT 全部 bin）。"""
    ant_names = ['N3', 'N4', 'N5', 'N6']
    delays = geometric_delays_casa()
    # I/Q 复数 250 Msps → FFT 覆盖 0~250 MHz，用全部 bin（不再只用前半 125 MHz）
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
            apply_dc_mask(fft, 0.5)  # 基带低/高端 ±0.5 MHz 掩码
            pos = fft[:freq_bins]    # 全 0~250 MHz
            for ich in range(OUTPUT_CHANNELS):
                b0, b1 = ich * bins_per_ch, (ich + 1) * bins_per_ch
                v = np.mean(pos[b0:b1]) * np.exp(-2j * np.pi * chan_freqs[ich] * delays[iant])
                epic[ig, ich, iant, 0] = epic[ig, ich, iant, 1] = v
    return epic


def gulps_to_video_frames(all_gulps, out_dir, image_size=64, fov_deg=30):
    """(total_gulps, 1024, 4, 2) -> 每帧一张 PNG，返回路径列表。"""
    baselines = get_baselines()
    l = np.linspace(-np.sin(np.deg2rad(fov_deg/2)), np.sin(np.deg2rad(fov_deg/2)), image_size)
    m = np.linspace(-np.sin(np.deg2rad(fov_deg/2)), np.sin(np.deg2rad(fov_deg/2)), image_size)
    L, M = np.meshgrid(l, m)
    os.makedirs(out_dir, exist_ok=True)
    paths = []
    for ig in tqdm(range(all_gulps.shape[0]), desc="  Dirty imaging"):
        g = all_gulps[ig, :, :, 0]
        vis = [np.mean(g[:, i] * np.conj(g[:, j])) for (i, j, u, v) in baselines]
        dirty = np.zeros((image_size, image_size), dtype=np.complex128)
        for (i, j, u, v), v_ij in zip(baselines, vis):
            phase = -2j * np.pi * (u * L + v * M)
            dirty += v_ij * np.exp(phase) + np.conj(v_ij) * np.exp(-phase)
        img = np.real(dirty)
        vmax = np.percentile(np.abs(img), 99)
        vmin = -vmax * 0.3
        plt.style.use("dark_background")
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(img, origin="lower", cmap="viridis", extent=[-fov_deg/2, fov_deg/2, -fov_deg/2, fov_deg/2], vmin=vmin, vmax=vmax)
        ax.set_xlabel("L (deg)")
        ax.set_ylabel("M (deg)")
        ax.set_title(f"9197 CasA | Frame {ig+1} | 40ms")
        plt.savefig(os.path.join(out_dir, f"frame_{ig:05d}.png"), dpi=100, bbox_inches="tight")
        plt.close(fig)
        paths.append(os.path.join(out_dir, f"frame_{ig:05d}.png"))
    return paths


def frames_to_gif(frame_paths, gif_path, duration_ms=80):
    ims = [Image.open(p).convert("P", palette=Image.ADAPTIVE, colors=256) for p in frame_paths]
    ims[0].save(gif_path, save_all=True, append_images=ims[1:], duration=duration_ms, loop=0)
    return gif_path


def main():
    file_indices = [int(x) for x in sys.argv[1:] if x.isdigit()]
    if not file_indices:
        file_indices = [0, 1]
    print("=" * 60)
    print("  9197 多文件全数据校准 → 视频")
    print("=" * 60)
    print(f"  文件索引: {file_indices}")
    print(f"  通道: {OUTPUT_CHANNELS}  带宽: 250 MHz (全基带)  中心: {CENTER_FREQ_MHZ} MHz")
    print(f"  校准: CasA 几何时延 + DC mask ±0.5 MHz")
    print(f"  每帧: 40 ms  不截断，每文件完整时长")
    print("=" * 60)

    proj = Path(__file__).resolve().parents[1]
    out_base = proj / "results" / "multi_full_video"
    out_base.mkdir(parents=True, exist_ok=True)
    frames_dir = out_base / "frames"
    frames_dir.mkdir(exist_ok=True)

    all_gulps_list = []
    for fi in file_indices:
        print(f"\n【文件 {fi}】完整读取 + 校准")
        g = process_one_file_to_gulps(fi, DATA_DIR)
        all_gulps_list.append(g)
        print(f"  -> {g.shape[0]} 帧 ({g.shape[0] * GULP_MS / 1000:.2f} s)")
    all_gulps = np.concatenate(all_gulps_list, axis=0)
    total_frames = all_gulps.shape[0]
    total_sec = total_frames * GULP_MS / 1000.0
    print(f"\n【合计】{total_frames} 帧  {total_sec:.2f} s")

    print("\n【成图】Python 脏图逐帧")
    frame_paths = gulps_to_video_frames(all_gulps, str(frames_dir))
    gif_path = out_base / "9197_multi_full_calib.gif"
    print("\n【合成 GIF】")
    frames_to_gif(frame_paths, str(gif_path), duration_ms=80)
    print(f"  视频: {gif_path}")
    print("\n完成。如需 MP4 可安装 ffmpeg 后:")
    print(f"  ffmpeg -framerate 25 -i {frames_dir}/frame_%05d.png -c:v libx264 -pix_fmt yuv420p {out_base}/9197_multi_full_calib.mp4")


if __name__ == "__main__":
    main()

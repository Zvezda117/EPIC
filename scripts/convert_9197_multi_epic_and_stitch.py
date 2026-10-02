#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
9197 多 .data 文件 → 每文件 1s 转换 → EPIC 成图 → 多图拼接

用法:
  python3 convert_9197_multi_epic_and_stitch.py 0 1 2 3
  python3 convert_9197_multi_epic_and_stitch.py --files 0 1 2 3 --grid 2 2
"""
import numpy as np
import os
import sys
import subprocess
import shutil
from pathlib import Path
from PIL import Image

# 复用 1s 转换逻辑
DATA_DIR = "/mnt/DATA/存储阵列/9197_Base2_ADC16_26-01-18_19-16-51/"
ANTENNA_FILES = {
    "N3": "HBaseband2Spec2", "N4": "VBaseband0Spec0",
    "N5": "VBaseband1Spec1", "N6": "VBaseband2Spec2",
}
C_LIGHT = 299792458.0
SAMPLE_RATE = 250e6
HEADER_SIZE = 256
DATA_SIZE = 65536
GULP_MS = 40
GULP_SAMPLES = int(SAMPLE_RATE * GULP_MS / 1000)
OUTPUT_CHANNELS = 128
SEQ_PER_1S = 25
CASA_RA_DEG, CASA_DEC_DEG = 350.85, 58.815
STATION_LAT_DEG, STATION_LON_DEG = 36.6578, 117.4903


def load_antenna_raw(filepath, num_packs):
    out = []
    with open(filepath, 'rb') as f:
        for _ in range(num_packs):
            f.seek(HEADER_SIZE, 1)
            raw = np.frombuffer(f.read(DATA_SIZE), dtype=np.int8)
            I, Q = raw[0::2].astype(np.float32), raw[1::2].astype(np.float32)
            out.append(I + 1j * Q)
    return np.concatenate(out)


def geometric_delays_casa_enu():
    enu = {'N3': (0.0, 0.0), 'N4': (21.06, 0.035), 'N5': (-2.661, -49.898), 'N6': (23.804, -49.902)}
    dec, lat = np.deg2rad(CASA_DEC_DEG), np.deg2rad(STATION_LAT_DEG)
    ha = 0
    sin_alt = np.sin(dec) * np.sin(lat) + np.cos(dec) * np.cos(lat) * np.cos(ha)
    alt = np.arcsin(sin_alt)
    cos_az = np.clip((np.sin(dec) - np.sin(alt) * np.sin(lat)) / (np.cos(alt) * np.cos(lat) + 1e-12), -1, 1)
    az = np.arccos(cos_az)
    s_enu = np.array([np.cos(alt) * np.sin(az), np.cos(alt) * np.cos(az), np.sin(alt)])
    return np.array([np.dot(np.array([e, n, 0.0]), s_enu) / C_LIGHT for (e, n) in enu.values()])


def apply_dc_mask(fft_data, dc_mask_mhz=0.5):
    nfft = len(fft_data)
    nbins = int(dc_mask_mhz * 1e6 / (SAMPLE_RATE / nfft))
    fft_data[0:nbins] = 0
    fft_data[-nbins:] = 0
    return fft_data


def convert_one_file_to_1s_npz(file_index, out_npz_path, data_dir=DATA_DIR):
    ant_names = ['N3', 'N4', 'N5', 'N6']
    need_samples = SEQ_PER_1S * GULP_SAMPLES
    need_packs = (need_samples + DATA_SIZE // 2 - 1) // (DATA_SIZE // 2)
    delays = geometric_delays_casa_enu()
    freq_bins = GULP_SAMPLES // 2
    bins_per_ch = freq_bins // OUTPUT_CHANNELS
    chan_freqs = (np.arange(OUTPUT_CHANNELS) + 0.5) * (SAMPLE_RATE / GULP_SAMPLES) * bins_per_ch

    ant_data = {}
    for name in ant_names:
        path = os.path.join(data_dir, f"{ANTENNA_FILES[name]}_{file_index}.data")
        if not os.path.isfile(path):
            raise FileNotFoundError(path)
        size = os.path.getsize(path)
        n_pack = min(size // (HEADER_SIZE + DATA_SIZE), need_packs)
        ant_data[name] = load_antenna_raw(path, n_pack)
    nsamples = min(len(ant_data[n]) for n in ant_names)
    nsamples = (nsamples // GULP_SAMPLES) * GULP_SAMPLES
    if nsamples < SEQ_PER_1S * GULP_SAMPLES:
        raise ValueError(f"file {file_index}: 不足 1s, 仅 {nsamples/SAMPLE_RATE:.2f}s")

    voltages = np.zeros((nsamples, 4), dtype=np.complex64)
    for i, n in enumerate(ant_names):
        voltages[:, i] = ant_data[n][:nsamples]

    nseqs = min(SEQ_PER_1S, nsamples // GULP_SAMPLES)
    epic = np.zeros((nseqs, OUTPUT_CHANNELS, 4, 2), dtype=np.complex64)
    for seq in range(nseqs):
        start = seq * GULP_SAMPLES
        end = start + GULP_SAMPLES
        for iant in range(4):
            fft = np.fft.fft(voltages[start:end, iant])
            apply_dc_mask(fft, 0.5)
            pos = fft[:freq_bins]
            for ich in range(OUTPUT_CHANNELS):
                b0, b1 = ich * bins_per_ch, (ich + 1) * bins_per_ch
                v = np.mean(pos[b0:b1]) * np.exp(-2j * np.pi * chan_freqs[ich] * delays[iant])
                epic[seq, ich, iant, 0] = epic[seq, ich, iant, 1] = v
    meta = np.array([0.0, 0.0], dtype=np.float64)
    os.makedirs(os.path.dirname(out_npz_path) or '.', exist_ok=True)
    np.savez_compressed(out_npz_path, data=epic, meta=meta)
    return out_npz_path


def run_epic_for_gulp(npz_path, epic_bin, out_dir, epic_args=None):
    """复制 npz 到 build/data 默认路径并运行 EPIC，返回生成的 QL.png 路径。"""
    build_data = os.path.join(os.path.dirname(epic_bin), "data", "40ms_128chan_600offset_gulp_c64_virtransit.npz")
    os.makedirs(os.path.dirname(build_data), exist_ok=True)
    shutil.copy(npz_path, build_data)
    out_dir = os.path.abspath(out_dir)
    cmd = [
        epic_bin, "--offline", "--out_dir", out_dir, "--disable_metrics",
        "--imagesize", "64", "--channels", "128", "--chan_nbin", "1",
        "--seq_accum", "40", "--nts", "1000", "--imageres", "1.056",
        "--accum_16bit", "--nstreams", "8", "--support", "5", "--aeff", "8",
        "--kernel_oversample", "4", "--ngpus", "1", "--gpu_ids=0",
    ]
    if epic_args:
        cmd.extend(epic_args)
    env = os.environ.copy()
    env.setdefault("MKL_THREADING_LAYER", "GNU")
    try:
        subprocess.run(cmd, check=True, cwd=os.path.dirname(epic_bin), env=env)
    except subprocess.CalledProcessError:
        pass
    for f in Path(out_dir).glob("EPIC_*_QL.png"):
        return str(f)
    return None


def stitch_images(png_paths, out_path, grid=(2, 2)):
    """将多张 PNG 按 grid 拼接成一张图。"""
    if not png_paths:
        raise ValueError("无图片可拼接")
    ims = [Image.open(p).convert("RGB") for p in png_paths]
    w, h = ims[0].size
    rows, cols = grid
    while len(ims) < rows * cols:
        ims.append(Image.new("RGB", (w, h), (0, 0, 0)))
    out = Image.new("RGB", (w * cols, h * rows))
    for i, im in enumerate(ims[: rows * cols]):
        r, c = i // cols, i % cols
        out.paste(im.resize((w, h)), (c * w, r * h))
    out.save(out_path)
    return out_path


def main():
    args = sys.argv[1:]
    if "--grid" in args:
        idx = args.index("--grid")
        grid = (int(args[idx + 1]), int(args[idx + 2]))
        args = args[:idx] + args[idx + 3:]
    else:
        grid = (2, 2)
    if "--files" in args:
        idx = args.index("--files")
        file_indices = [int(x) for x in args[idx + 1:] if x.isdigit()]
    else:
        file_indices = [int(x) for x in args if x.isdigit()]
    if not file_indices:
        file_indices = [0, 1, 2, 3]
    print(f"使用 .data 文件索引: {file_indices}, 拼接网格: {grid}")

    proj = Path(__file__).resolve().parents[1]
    epic_bin = proj / "build" / "epic++"
    if not epic_bin.is_file():
        raise SystemExit("未找到 build/epic++，请先编译 EPIC")
    out_base = proj / "results" / "multi_9197"
    npz_dir = proj / "data" / "epic_ready"
    out_base.mkdir(parents=True, exist_ok=True)
    npz_dir.mkdir(parents=True, exist_ok=True)

    png_paths = []
    for fi in file_indices:
        npz_path = npz_dir / f"9197_1s_file{fi}.npz"
        print(f"\n【文件 {fi}】转换 1s npz -> {npz_path.name}")
        try:
            convert_one_file_to_1s_npz(fi, str(npz_path), DATA_DIR)
        except Exception as e:
            print(f"  跳过: {e}")
            continue
        run_dir = out_base / f"run_{fi}"
        run_dir.mkdir(exist_ok=True)
        print(f"  运行 EPIC -> {run_dir}")
        ql = run_epic_for_gulp(str(npz_path), str(epic_bin), str(run_dir))
        if ql:
            dest = out_base / f"file_{fi}.png"
            shutil.copy(ql, dest)
            png_paths.append(dest)
            print(f"  保存 -> {dest.name}")

    if not png_paths:
        raise SystemExit("没有成功成图，无法拼接")
    stitched = out_base / "stitched.png"
    stitch_images([str(p) for p in png_paths], str(stitched), grid=grid)
    print(f"\n拼接图已保存: {stitched}")


if __name__ == "__main__":
    main()

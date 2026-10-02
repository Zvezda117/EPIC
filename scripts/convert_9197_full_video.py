#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
9197完整数据转换 - 用于生成视频
- 使用完整4秒数据
- DC mask ± 0.5 MHz
- 40ms积分每帧
- 输出多个gulp用于视频
"""
import numpy as np
import os
from tqdm import tqdm

# 观测参数
CENTER_FREQ = 1355.0  # MHz
BANDWIDTH = 250.0     # MHz
SAMPLE_RATE = 250e6   # Hz

# 包结构
HEADER_SIZE = 256
DATA_SIZE = 65536
PACK_SIZE = HEADER_SIZE + DATA_SIZE
SAMPLES_PER_PACK = DATA_SIZE // 2  # 32768 复数采样

# 天线文件
DATA_DIR = "/mnt/DATA/存储阵列/9197_Base2_ADC16_26-01-18_19-16-51/"
ANTENNA_FILES = {
    "N3": "HBaseband2Spec2",
    "N4": "VBaseband0Spec0", 
    "N5": "VBaseband1Spec1",
    "N6": "VBaseband2Spec2"
}

# EPIC参数
OUTPUT_CHANNELS = 128
GULP_MS = 40  # ms per gulp
GULP_SAMPLES = int(SAMPLE_RATE * GULP_MS / 1000)  # 10,000,000 samples per gulp

def read_all_antenna_data(file_index=0):
    """读取所有天线的完整数据"""
    ant_names = ['N3', 'N4', 'N5', 'N6']
    ant_data = {}
    
    for ant_name in ant_names:
        prefix = ANTENNA_FILES[ant_name]
        filepath = os.path.join(DATA_DIR, f"{prefix}_{file_index}.data")
        
        file_size = os.path.getsize(filepath)
        num_packs = file_size // PACK_SIZE
        
        print(f"  {ant_name}: {filepath}")
        print(f"    File size: {file_size/1e9:.2f} GB, Packs: {num_packs:,}")
        
        all_data = []
        with open(filepath, 'rb') as f:
            for i in tqdm(range(num_packs), desc=f"    Reading {ant_name}", leave=False):
                f.seek(i * PACK_SIZE + HEADER_SIZE)
                data = np.frombuffer(f.read(DATA_SIZE), dtype=np.int8)
                I = data[0::2].astype(np.float32)
                Q = data[1::2].astype(np.float32)
                all_data.append(I + 1j * Q)
        
        ant_data[ant_name] = np.concatenate(all_data)
        print(f"    Loaded: {len(ant_data[ant_name]):,} samples ({len(ant_data[ant_name])/SAMPLE_RATE:.3f} s)")
    
    return ant_data

def apply_dc_mask(fft_data, dc_mask_mhz=0.5):
    """
    在频域应用DC mask
    
    Args:
        fft_data: FFT后的数据 (nfft,)
        dc_mask_mhz: DC附近mask的宽度 (MHz)
    """
    nfft = len(fft_data)
    freq_resolution = SAMPLE_RATE / nfft  # Hz per bin
    mask_bins = int(dc_mask_mhz * 1e6 / freq_resolution)  # 单边bins数
    
    # DC在bin 0
    fft_data[0:mask_bins] = 0
    fft_data[-mask_bins:] = 0
    
    return fft_data

def convert_to_epic_format(ant_data, output_channels=128, dc_mask_mhz=0.5):
    """
    转换为EPIC格式
    
    返回:
        epic_data: (ngulps, nchannels, 256, 2) 格式的数据
    """
    ant_names = ['N3', 'N4', 'N5', 'N6']
    
    # 获取最小采样数
    nsamples = min(len(ant_data[name]) for name in ant_names)
    ngulps = nsamples // GULP_SAMPLES
    
    print(f"\n【转换参数】")
    print(f"  总采样数: {nsamples:,}")
    print(f"  Gulp大小: {GULP_SAMPLES:,} ({GULP_MS} ms)")
    print(f"  总Gulp数: {ngulps}")
    print(f"  输出通道: {output_channels}")
    print(f"  DC mask: ±{dc_mask_mhz} MHz")
    
    # 组合数据
    voltages = np.zeros((nsamples, 4), dtype=np.complex64)
    for i, ant_name in enumerate(ant_names):
        voltages[:, i] = ant_data[ant_name][:nsamples]
    
    # FFT参数
    fft_size = GULP_SAMPLES
    freq_bins = fft_size // 2  # 正频率bins
    bins_per_channel = freq_bins // output_channels
    
    print(f"  FFT大小: {fft_size:,}")
    print(f"  Bins/通道: {bins_per_channel}")
    print(f"  通道带宽: {bins_per_channel * SAMPLE_RATE / fft_size / 1e3:.1f} kHz")
    
    # 输出数组
    epic_data = np.zeros((ngulps, output_channels, 256, 2), dtype=np.complex64)
    
    print(f"\n【处理数据】")
    for igulp in tqdm(range(ngulps), desc="  Processing gulps"):
        start = igulp * GULP_SAMPLES
        end = start + GULP_SAMPLES
        
        for iant in range(4):
            gulp_data = voltages[start:end, iant]
            
            # FFT
            fft_full = np.fft.fft(gulp_data)
            
            # DC mask
            fft_full = apply_dc_mask(fft_full, dc_mask_mhz)
            
            # 只取正频率
            fft_positive = fft_full[:freq_bins]
            
            # 通道合并
            for ichan in range(output_channels):
                start_bin = ichan * bins_per_channel
                end_bin = start_bin + bins_per_channel
                epic_data[igulp, ichan, iant, 0] = np.mean(fft_positive[start_bin:end_bin])
                epic_data[igulp, ichan, iant, 1] = np.mean(fft_positive[start_bin:end_bin])
    
    return epic_data, ngulps

def main():
    print("=" * 60)
    print("  9197 完整数据转换 (用于视频生成)")
    print("=" * 60)
    
    # 读取数据
    print("\n【读取数据】")
    ant_data = read_all_antenna_data(file_index=0)
    
    # 转换
    epic_data, ngulps = convert_to_epic_format(
        ant_data, 
        output_channels=128, 
        dc_mask_mhz=0.5
    )
    
    print(f"\n【输出】")
    print(f"  Shape: {epic_data.shape}")
    
    # 保存
    os.makedirs('data/epic_ready', exist_ok=True)
    output_file = f'data/epic_ready/9197_full_{ngulps}gulps_128chan_dcmask.npz'
    
    meta = np.array([ngulps, 0], dtype=np.int64)
    np.savez_compressed(output_file, data=epic_data, meta=meta)
    
    file_size = os.path.getsize(output_file) / 1e6
    print(f"  文件: {output_file}")
    print(f"  大小: {file_size:.1f} MB")
    
    print("\n" + "=" * 60)
    print("  转换完成!")
    print("=" * 60)
    
    return output_file, ngulps

if __name__ == '__main__':
    main()

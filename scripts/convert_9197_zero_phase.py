#!/usr/bin/env python3
"""
9197数据转换：假设相位为零

策略：
1. 不做时延校准
2. 直接将数据转换为EPIC格式
3. DC mask
4. 让EPIC处理成图，看能否看到任何东西

这是一个baseline：如果连零相位假设都看不到CasA，
说明问题不在校准，而在数据质量或观测设置。
"""

import numpy as np
import os
from tqdm import tqdm

# 包结构
HEADER_SIZE = 256
DATA_SIZE = 65536
PACK_SIZE = HEADER_SIZE + DATA_SIZE
SAMPLES_PER_PACK = DATA_SIZE // 2

# 天线文件映射
antenna_files = {
    "北3": "HBaseband2Spec2",
    "北4": "VBaseband0Spec0", 
    "北5": "VBaseband1Spec1",
    "北6": "VBaseband2Spec2"
}

def read_antenna_data(filepath, num_packs=1000):
    """读取天线数据"""
    all_data = []
    with open(filepath, 'rb') as f:
        for i in range(num_packs):
            f.seek(i * PACK_SIZE + HEADER_SIZE)
            data = np.frombuffer(f.read(DATA_SIZE), dtype=np.int8)
            I = data[0::2].astype(np.float32)
            Q = data[1::2].astype(np.float32)
            all_data.append(I + 1j * Q)
    return np.concatenate(all_data)

def convert_to_epic_format(num_packs=1000, output_channels=128, dc_mask_bins=100):
    """
    转换为EPIC格式
    
    参数:
        num_packs: 读取的包数
        output_channels: 输出频率通道数
        dc_mask_bins: DC附近mask的bins数
    """
    print("═══════════════════════════════════════════════════════════")
    print("  9197数据转换（零相位假设 + DC mask）")
    print("═══════════════════════════════════════════════════════════\n")
    
    data_dir = "/mnt/DATA/存储阵列/9197_Base2_ADC16_26-01-18_19-16-51/"
    sample_rate = 250e6
    
    # 读取数据
    print("【读取数据】")
    ant_names = ['北3', '北4', '北5', '北6']
    ant_data = {}
    
    for ant_name in ant_names:
        prefix = antenna_files[ant_name]
        filepath = os.path.join(data_dir, f"{prefix}_0.data")
        data = read_antenna_data(filepath, num_packs=num_packs)
        ant_data[ant_name] = data
        print(f"  {ant_name}: {len(data):,} 采样点")
    
    nsamples = min(len(d) for d in ant_data.values())
    print(f"\n总采样点: {nsamples:,}")
    print(f"时长: {nsamples/sample_rate:.3f} 秒")
    
    # FFT参数
    # 40ms gulp
    gulp_samples = int(sample_rate * 0.04)  # 10,000,000
    
    # 但我们的nsamples可能不够10M，使用实际可用的
    if nsamples < gulp_samples:
        gulp_samples = nsamples
    
    num_gulps = nsamples // gulp_samples
    
    fft_size = gulp_samples
    freq_bins = fft_size // 2
    bins_per_channel = freq_bins // output_channels
    
    print(f"\n【FFT参数】")
    print(f"  Gulp大小: {gulp_samples:,} ({gulp_samples/sample_rate*1000:.1f} ms)")
    print(f"  Gulp数量: {num_gulps}")
    print(f"  FFT大小: {fft_size:,}")
    print(f"  输出通道: {output_channels}")
    print(f"  bins/通道: {bins_per_channel}")
    print(f"  通道带宽: {bins_per_channel * sample_rate / fft_size / 1e3:.1f} kHz")
    print(f"  DC mask: ±{dc_mask_bins//2} bins")
    
    # 组合数据
    voltages = np.zeros((nsamples, 4), dtype=np.complex64)
    for i, ant_name in enumerate(ant_names):
        voltages[:, i] = ant_data[ant_name][:nsamples]
    
    # FFT转换
    print(f"\n【FFT转换 + DC mask】")
    fft_result = np.zeros((num_gulps, output_channels, 4), dtype=np.complex64)
    
    for igulp in tqdm(range(num_gulps), desc="处理"):
        start = igulp * gulp_samples
        end = start + gulp_samples
        
        for iant in range(4):
            gulp_data = voltages[start:end, iant]
            
            # FFT
            fft_full = np.fft.fft(gulp_data)
            
            # DC mask（将DC附近设为0）
            dc_center = 0
            fft_full[dc_center : dc_mask_bins//2] = 0
            fft_full[-dc_mask_bins//2:] = 0
            
            # 只取正频率
            fft_positive = fft_full[:freq_bins]
            
            # 通道合并
            for ichan in range(output_channels):
                start_bin = ichan * bins_per_channel
                end_bin = start_bin + bins_per_channel
                fft_result[igulp, ichan, iant] = np.mean(fft_positive[start_bin:end_bin])
    
    print("✓ FFT完成")
    
    # 填充到EPIC格式
    print(f"\n【填充到EPIC格式】")
    epic_data = np.zeros((num_gulps, output_channels, 256, 2), dtype=np.complex64)
    
    for iant in range(4):
        epic_data[:, :, iant, 0] = fft_result[:, :, iant]
        epic_data[:, :, iant, 1] = fft_result[:, :, iant]
    
    print(f"  输出shape: {epic_data.shape}")
    
    # 统计
    print(f"\n【数据统计】")
    for iant in range(4):
        power = np.mean(np.abs(epic_data[:, :, iant, 0])**2)
        print(f"  天线{iant}: 平均功率 = {power:.2e}")
    
    # 保存
    meta = np.array([num_gulps, 0], dtype=np.int64)
    output_file = f'data/epic_ready/9197_zero_phase_{num_gulps}gulps_{output_channels}chan.npz'
    
    print(f"\n【保存】")
    print(f"  输出文件: {output_file}")
    
    np.savez_compressed(output_file, data=epic_data, meta=meta)
    
    import os as os_module
    file_size = os_module.path.getsize(output_file) / 1e6
    print(f"  文件大小: {file_size:.1f} MB")
    
    return output_file, num_gulps, output_channels

if __name__ == '__main__':
    import sys
    
    num_packs = 1000 if len(sys.argv) <= 1 else int(sys.argv[1])
    output_channels = 128 if len(sys.argv) <= 2 else int(sys.argv[2])
    
    output_file, ngulps, nchans = convert_to_epic_format(
        num_packs=num_packs,
        output_channels=output_channels,
        dc_mask_bins=200
    )
    
    print(f"\n✅ 转换完成！")
    print(f"\n下一步:")
    print(f"  cp {output_file} build/data/40ms_128chan_600offset_gulp_c64_virtransit.npz")
    print(f"  cd build && ./epic++ --offline --channels {nchans} --nts {ngulps} ...")

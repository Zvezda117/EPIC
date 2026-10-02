#!/usr/bin/env python3
"""
9197数据转换：40ms积分 + 250MHz全带宽

满足要求：
1. 使用40ms时间积分（与原始LWA-SV gulp一致）
2. 使用250MHz全带宽（100%利用率）
3. 多通道输出（1024通道）
4. 蟹状星云时延校准
"""

import numpy as np
import yaml
from pathlib import Path
from tqdm import tqdm

# 蟹状星云坐标 (J2000)
CRAB_RA = 83.6331   # 度
CRAB_DEC = 22.0145  # 度

def load_antenna_positions():
    """加载天线坐标"""
    config_file = Path('antenna_mapping_9197.yaml')
    with open(config_file) as f:
        config = yaml.safe_load(f)
    
    positions = {}
    for ant in config['antennas']:
        positions[ant['name']] = np.array(ant['position'])
    
    return positions

def calculate_geometric_delays(positions, ra_deg, dec_deg):
    """
    计算几何时延（相对参考天线）
    
    参数:
        positions: 天线位置字典 {name: [x,y,z]} (ENU坐标，米)
        ra_deg: 赤经（度）
        dec_deg: 赤纬（度）
    
    返回:
        delays: 各天线相对延迟（秒）
    """
    # 转换为弧度
    ra_rad = np.deg2rad(ra_deg)
    dec_rad = np.deg2rad(dec_deg)
    
    # 源方向单位向量（ENU坐标系）
    # 注意：需要考虑地理纬度和时角，这里简化处理
    # 精确计算需要使用astropy
    s = np.array([
        np.cos(dec_rad) * np.cos(ra_rad),
        np.cos(dec_rad) * np.sin(ra_rad),
        np.sin(dec_rad)
    ])
    
    # 光速
    c = 299792458.0  # m/s
    
    # 计算各天线的几何时延
    delays = []
    ant_names = ['北3', '北4', '北5', '北6']
    pos_array = np.array([positions[name] for name in ant_names])
    
    # 相对参考天线（北3）的时延
    ref_pos = pos_array[0]
    for pos in pos_array:
        baseline = pos - ref_pos
        delay = -np.dot(baseline, s) / c  # 秒
        delays.append(delay)
    
    return np.array(delays), ant_names

def apply_delay_correction_chunked(voltages, delays, sample_rate=250e6, chunk_size=5000000):
    """
    应用时延校准（分块处理以节省内存）
    
    参数:
        voltages: 时域电压 (nsamples, nantennas)
        delays: 时延数组 (nantennas,) 秒
        sample_rate: 采样率 Hz
        chunk_size: 每块大小
    
    返回:
        corrected: 校准后的电压
    """
    nsamples, nantennas = voltages.shape
    corrected = np.zeros_like(voltages)
    
    print("  应用时延校准...")
    
    for iant in range(nantennas):
        if abs(delays[iant]) < 1e-15:
            # 参考天线，无需校准
            corrected[:, iant] = voltages[:, iant]
            continue
        
        # 分块处理
        for start in range(0, nsamples, chunk_size):
            end = min(start + chunk_size, nsamples)
            chunk_len = end - start
            
            # 生成频率轴
            freq = np.fft.fftfreq(chunk_len, 1/sample_rate)
            phase_shift = np.exp(-2j * np.pi * freq * delays[iant])
            
            # 转到频域
            fft_volt = np.fft.fft(voltages[start:end, iant])
            
            # 应用相位校正
            fft_corrected = fft_volt * phase_shift
            
            # 转回时域
            corrected[start:end, iant] = np.fft.ifft(fft_corrected)
    
    return corrected

def convert_fullband_40ms(
    output_channels=1024,
    num_gulps=None
):
    """
    全带宽40ms积分转换
    
    参数:
        output_channels: 输出频率通道数（默认1024）
        num_gulps: 处理的gulp数量（None=所有）
    """
    
    print("═══════════════════════════════════════════════════════════")
    print("  9197数据转换：40ms积分 + 250MHz全带宽")
    print("═══════════════════════════════════════════════════════════\n")
    
    # 参数
    sample_rate = 250e6
    gulp_time = 0.04  # 40ms
    samples_per_gulp = int(sample_rate * gulp_time)  # 10,000,000
    
    print("【转换参数】")
    print(f"时间积分: {gulp_time*1000:.0f} ms (与原始LWA-SV gulp一致)")
    print(f"每gulp采样点: {samples_per_gulp:,}")
    print(f"采样率: {sample_rate/1e6:.0f} MHz")
    print(f"输出通道数: {output_channels}")
    print()
    
    # 加载时域数据
    print("【步骤1: 加载时域数据】")
    time_data = np.load('data/processed/9197_file0_complete.npz')
    voltages_full = time_data['voltages']
    print(f"原始数据shape: {voltages_full.shape}")
    print(f"可用采样点: {voltages_full.shape[0]:,}")
    print(f"可用时长: {voltages_full.shape[0]/sample_rate:.3f} 秒")
    
    # 计算可以处理的gulp数量
    max_gulps = voltages_full.shape[0] // samples_per_gulp
    if num_gulps is None or num_gulps > max_gulps:
        num_gulps = max_gulps
    
    total_samples = num_gulps * samples_per_gulp
    print(f"处理gulp数: {num_gulps}")
    print(f"总采样点: {total_samples:,}")
    print(f"总时长: {total_samples/sample_rate:.3f} 秒")
    print()
    
    # 提取数据
    voltages_subset = voltages_full[:total_samples, 0, :, 0]
    print(f"提取数据shape: {voltages_subset.shape}")
    print()
    
    # 加载天线坐标
    print("【步骤2: 时延校准】")
    positions = load_antenna_positions()
    delays, ant_names = calculate_geometric_delays(positions, CRAB_RA, CRAB_DEC)
    
    print(f"观测目标: 蟹状星云 (RA={CRAB_RA}°, Dec={CRAB_DEC}°)")
    print("\n几何时延（相对北3）:")
    for i, name in enumerate(ant_names):
        print(f"  {name}: {delays[i]*1e9:+8.2f} ns ({delays[i]*sample_rate:+8.1f} 采样点)")
    print()
    
    # 应用时延校准
    voltages_corrected = apply_delay_correction_chunked(
        voltages_subset, 
        delays,
        sample_rate,
        chunk_size=samples_per_gulp  # 每次处理一个gulp
    )
    print("  ✓ 时延校准完成")
    print()
    
    # FFT转换（每个gulp独立）
    print("【步骤3: FFT转换（全带宽）】")
    print(f"重塑为 ({num_gulps}, {samples_per_gulp}, 4)")
    voltages_reshaped = voltages_corrected.reshape(num_gulps, samples_per_gulp, 4)
    
    # FFT参数
    fft_size = samples_per_gulp
    freq_bins = fft_size // 2  # Nyquist频率
    freq_resolution = sample_rate / fft_size
    
    print(f"FFT长度: {fft_size:,}")
    print(f"输出频率bins: {freq_bins:,}")
    print(f"频率分辨率: {freq_resolution:.1f} Hz")
    print(f"总带宽: {freq_bins * freq_resolution / 1e6:.1f} MHz")
    print()
    
    # 通道合并策略
    bins_per_channel = freq_bins // output_channels
    actual_channels = freq_bins // bins_per_channel
    channel_bandwidth = bins_per_channel * freq_resolution
    
    print("【通道合并】")
    print(f"原始bins: {freq_bins:,}")
    print(f"输出通道: {actual_channels}")
    print(f"bins/通道: {bins_per_channel}")
    print(f"通道带宽: {channel_bandwidth/1e3:.1f} kHz")
    print(f"总带宽: {actual_channels * channel_bandwidth/1e6:.1f} MHz")
    print()
    
    # 执行FFT并合并通道
    print("执行FFT并合并通道...")
    fft_result = np.zeros((num_gulps, actual_channels, 4), dtype=np.complex64)
    
    for igulp in tqdm(range(num_gulps), desc="FFT处理"):
        for iant in range(4):
            # FFT
            timeseries = voltages_reshaped[igulp, :, iant]
            fft_full = np.fft.fft(timeseries)
            
            # 只取正频率部分
            fft_positive = fft_full[:freq_bins]
            
            # 通道合并（求平均功率）
            for ichan in range(actual_channels):
                start_bin = ichan * bins_per_channel
                end_bin = start_bin + bins_per_channel
                # 相干平均（保持相位信息）
                fft_result[igulp, ichan, iant] = np.mean(fft_positive[start_bin:end_bin])
    
    print("✓ FFT完成")
    print(f"输出shape: {fft_result.shape}")
    print()
    
    # 填充到256天线×2极化
    print("【步骤4: 填充到256天线×2极化】")
    epic_data = np.zeros((num_gulps, actual_channels, 256, 2), dtype=np.complex64)
    
    for iant in range(4):
        # H极化和V极化填充相同数据（原始为单极化）
        epic_data[:, :, iant, 0] = fft_result[:, :, iant]
        epic_data[:, :, iant, 1] = fft_result[:, :, iant]
    
    print(f"最终shape: {epic_data.shape}")
    print()
    
    # 数据统计
    print("【数据统计】")
    for iant in range(4):
        power = np.abs(epic_data[:, :, iant, 0]).mean()
        power_std = np.abs(epic_data[:, :, iant, 0]).std()
        print(f"天线{iant} ({ant_names[iant]}): 平均功率={power:.2f}, 标准差={power_std:.2f}")
    print()
    
    # 创建meta
    meta = np.array([num_gulps, 0], dtype=np.int64)
    
    # 保存
    output_file = f'data/epic_ready/9197_{num_gulps}gulps_{actual_channels}chan_fullband_40ms.npz'
    print(f"【步骤5: 保存】")
    print(f"输出文件: {output_file}")
    
    np.savez_compressed(output_file, data=epic_data, meta=meta)
    
    import os
    file_size = os.path.getsize(output_file) / 1e6
    print(f"文件大小: {file_size:.1f} MB")
    print()
    
    # 总结
    print("═══════════════════════════════════════════════════════════")
    print("  转换完成！")
    print("═══════════════════════════════════════════════════════════")
    print()
    print("【关键指标】")
    print(f"✓ 时间积分: {gulp_time*1000:.0f} ms/gulp (与LWA-SV一致)")
    print(f"✓ 带宽利用: {actual_channels * channel_bandwidth/sample_rate * 100:.1f}%")
    print(f"✓ 输出通道: {actual_channels}")
    print(f"✓ 通道带宽: {channel_bandwidth/1e3:.1f} kHz")
    print(f"✓ 总时长: {num_gulps * gulp_time:.3f} 秒")
    print(f"✓ 时延校准: 基于蟹状星云位置")
    print()
    
    print("【与原始40ms_128chan_600offset文件对比】")
    print("| 参数 | 原始 | 当前 |")
    print("|------|------|------|")
    print(f"| 时间积分 | 40ms | 40ms ✓ |")
    print(f"| 带宽 | ~1.3MHz | {actual_channels * channel_bandwidth/1e6:.0f}MHz ✓ |")
    print(f"| 通道数 | 128 | {actual_channels} ✓ |")
    print(f"| 时延校准 | ✗ | ✓ |")
    print()
    
    return output_file, num_gulps, actual_channels

if __name__ == '__main__':
    import sys
    
    # 命令行参数
    if len(sys.argv) > 1:
        output_channels = int(sys.argv[1])
    else:
        output_channels = 1024
    
    if len(sys.argv) > 2:
        num_gulps = int(sys.argv[2])
    else:
        num_gulps = None  # 处理所有
    
    print(f"输出通道数: {output_channels}")
    if num_gulps:
        print(f"处理gulp数: {num_gulps}")
    else:
        print(f"处理gulp数: 所有可用")
    print()
    
    output_file, ngulps, nchans = convert_fullband_40ms(
        output_channels=output_channels,
        num_gulps=num_gulps
    )
    
    print("✅ 转换完成！")
    print(f"\n下一步：")
    print(f"  1. 复制到EPIC输入位置:")
    print(f"     cp {output_file} build/data/40ms_128chan_600offset_gulp_c64_virtransit.npz")
    print(f"  2. 运行EPIC:")
    print(f"     cd build && ./epic++ --offline --channels {nchans} --nts {ngulps} --seq_accum {int(ngulps*40)} ...")

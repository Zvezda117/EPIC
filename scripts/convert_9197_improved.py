#!/usr/bin/env python3
"""
改进的9197数据转换脚本

改进点:
1. 更长的积分时间（10ms vs 0.1ms）→ 提升信噪比10倍
2. 更大的带宽利用（512通道 vs 128通道）→ 4倍带宽
3. 使用所有可用数据（2.14秒 vs 0.1秒）
4. 添加时延校准（基于天线坐标）
"""

import numpy as np
import yaml
from pathlib import Path

def load_antenna_positions():
    """加载天线坐标"""
    config_file = Path('antenna_mapping_9197.yaml')
    with open(config_file) as f:
        config = yaml.safe_load(f)
    
    positions = {}
    for ant in config['antennas']:
        positions[ant['name']] = ant['position']
    
    return positions

def calculate_geometric_delays(positions, ra_deg, dec_deg):
    """
    计算几何时延
    
    参数:
        positions: 天线位置字典 {name: [x,y,z]}
        ra_deg: 赤经（度）
        dec_deg: 赤纬（度）
    
    返回:
        delays: 各天线相对延迟（秒）
    """
    import numpy as np
    
    # 转换为弧度
    ra_rad = np.deg2rad(ra_deg)
    dec_rad = np.deg2rad(dec_deg)
    
    # 源方向单位向量（J2000）
    s = np.array([
        np.cos(dec_rad) * np.cos(ra_rad),
        np.cos(dec_rad) * np.sin(ra_rad),
        np.sin(dec_rad)
    ])
    
    # 计算各天线的几何时延
    c = 299792458.0  # 光速 m/s
    delays = []
    pos_array = []
    
    for name in ['北3', '北4', '北5', '北6']:
        pos = positions[name]
        pos_array.append(pos)
    
    pos_array = np.array(pos_array)
    
    # 相对参考天线（北3）的时延
    ref_pos = pos_array[0]
    for pos in pos_array:
        baseline = pos - ref_pos
        delay = -np.dot(baseline, s) / c  # 秒
        delays.append(delay)
    
    return np.array(delays)

def apply_delay_correction(voltages, delays, sample_rate=250e6):
    """
    应用时延校准
    
    参数:
        voltages: 时域电压 (nsamples, nantennas)
        delays: 时延数组 (nantennas,) 秒
        sample_rate: 采样率 Hz
    
    返回:
        corrected: 校准后的电压
    """
    nsamples, nantennas = voltages.shape
    corrected = np.zeros_like(voltages)
    
    for iant in range(nantennas):
        delay_samples = delays[iant] * sample_rate
        
        # 使用FFT进行分数样本延迟
        freq = np.fft.fftfreq(nsamples, 1/sample_rate)
        phase_shift = np.exp(-2j * np.pi * freq * delays[iant])
        
        # 转到频域
        fft_volt = np.fft.fft(voltages[:, iant])
        
        # 应用相位校正
        fft_corrected = fft_volt * phase_shift
        
        # 转回时域
        corrected[:, iant] = np.fft.ifft(fft_corrected)
    
    return corrected

def convert_with_improved_settings(
    ntime_steps=214,           # 使用所有数据
    samples_per_step=2500000,  # 10ms积分
    nchan=512,                 # 512通道
    apply_delay_cal=True,      # 应用时延校准
    source_ra=None,            # 源赤经（度）
    source_dec=None            # 源赤纬（度）
):
    """
    改进的转换函数
    
    参数:
        ntime_steps: 时间步数（默认214，用满所有数据）
        samples_per_step: 每步采样点（默认2,500,000 = 10ms）
        nchan: 输出频率通道数（默认512）
        apply_delay_cal: 是否应用时延校准
        source_ra, source_dec: 源位置（用于时延校准）
    """
    
    print("═══════════════════════════════════════════")
    print("  9197数据改进转换")
    print("═══════════════════════════════════════════\n")
    
    # 参数总结
    print("【转换参数】")
    print(f"时间步数: {ntime_steps}")
    print(f"每步采样点: {samples_per_step:,}")
    print(f"每步积分时间: {samples_per_step/250e6*1000:.1f} ms")
    print(f"总积分时间: {ntime_steps * samples_per_step/250e6:.2f} 秒")
    print(f"输出频率通道: {nchan}")
    print(f"频率分辨率: {250e6/samples_per_step/1000:.1f} kHz")
    print(f"总带宽: {nchan * 250e6/samples_per_step/1e6:.1f} MHz")
    print(f"时延校准: {'✓ 启用' if apply_delay_cal else '✗ 禁用'}")
    print()
    
    # 加载时域数据
    print("【步骤1: 加载时域数据】")
    time_data = np.load('data/processed/9197_file0_complete.npz')
    voltages_full = time_data['voltages']
    print(f"原始数据: {voltages_full.shape}")
    print(f"可用采样点: {voltages_full.shape[0]:,}")
    print(f"可用时长: {voltages_full.shape[0]/250e6:.2f} 秒")
    print()
    
    # 计算需要的总采样点
    total_samples = ntime_steps * samples_per_step
    
    if total_samples > voltages_full.shape[0]:
        print(f"⚠️  请求{total_samples:,}个采样点，但只有{voltages_full.shape[0]:,}个")
        total_samples = voltages_full.shape[0]
        ntime_steps = total_samples // samples_per_step
        print(f"调整为: {ntime_steps} 时间步")
    
    # 提取子集
    voltages_subset = voltages_full[:total_samples, 0, :, 0]
    print(f"使用数据: {voltages_subset.shape}")
    print(f"使用时长: {total_samples/250e6:.2f} 秒")
    print()
    
    # 时延校准
    if apply_delay_cal and source_ra is not None and source_dec is not None:
        print("【步骤2: 时延校准】")
        positions = load_antenna_positions()
        delays = calculate_geometric_delays(positions, source_ra, source_dec)
        
        print("几何时延:")
        for i, name in enumerate(['北3', '北4', '北5', '北6']):
            print(f"  {name}: {delays[i]*1e9:.2f} ns ({delays[i]*250e6:.1f} 采样点)")
        
        print("\n应用时延校准...")
        # 分块校准以节省内存
        chunk_size = 10000000
        voltages_corrected = np.zeros_like(voltages_subset)
        for start in range(0, total_samples, chunk_size):
            end = min(start + chunk_size, total_samples)
            voltages_corrected[start:end] = apply_delay_correction(
                voltages_subset[start:end], 
                delays,
                250e6
            )
        voltages_subset = voltages_corrected
        print("✓ 时延校准完成")
        print()
    else:
        print("【步骤2: 跳过时延校准】")
        if not apply_delay_cal:
            print("原因: 未启用")
        else:
            print("原因: 未提供源位置")
        print("⚠️  警告: 无校准会严重影响成像质量！")
        print()
    
    # FFT转换
    print("【步骤3: FFT转换】")
    print(f"重塑为 ({ntime_steps}, {samples_per_step}, 4)")
    voltages_reshaped = voltages_subset.reshape(ntime_steps, samples_per_step, 4)
    
    fft_result = np.zeros((ntime_steps, nchan, 4), dtype=np.complex64)
    
    print("执行FFT...")
    for itime in range(ntime_steps):
        for iant in range(4):
            timeseries = voltages_reshaped[itime, :, iant]
            fft_full = np.fft.fft(timeseries)
            fft_result[itime, :, iant] = fft_full[:nchan]
        
        if (itime+1) % 50 == 0:
            print(f"  进度: {itime+1}/{ntime_steps}")
    
    print("✓ FFT完成")
    print(f"输出shape: {fft_result.shape}")
    print()
    
    # 填充到256天线
    print("【步骤4: 填充到256天线×2极化】")
    epic_data = np.zeros((ntime_steps, nchan, 256, 2), dtype=np.complex64)
    
    for iant in range(4):
        epic_data[:, :, iant, 0] = fft_result[:, :, iant]
        epic_data[:, :, iant, 1] = fft_result[:, :, iant]
    
    print(f"最终shape: {epic_data.shape}")
    print(f"  时间步: {epic_data.shape[0]}")
    print(f"  频道: {epic_data.shape[1]}")
    print(f"  天线: {epic_data.shape[2]} (前4个有数据)")
    print(f"  极化: {epic_data.shape[3]}")
    print()
    
    # 统计
    print("【数据统计】")
    for iant in range(4):
        power = np.abs(epic_data[:, :, iant, 0]).mean()
        print(f"天线{iant}: 平均功率 = {power:.2f}")
    print()
    
    # 创建meta
    meta = np.array([ntime_steps, 0], dtype=np.int64)
    
    # 保存
    output_file = f'data/epic_ready/9197_{ntime_steps}steps_{nchan}chan_improved.npz'
    print(f"【步骤5: 保存】")
    print(f"输出文件: {output_file}")
    
    np.savez_compressed(output_file, data=epic_data, meta=meta)
    
    import os
    file_size = os.path.getsize(output_file) / 1e6
    print(f"文件大小: {file_size:.1f} MB")
    print()
    
    # 对比原始版本
    print("═══════════════════════════════════════════")
    print("  改进对比")
    print("═══════════════════════════════════════════")
    print()
    print("| 参数 | 原始 | 改进 | 提升 |")
    print("|------|------|------|------|")
    print(f"| 积分时间 | 0.1 秒 | {ntime_steps*samples_per_step/250e6:.2f} 秒 | {ntime_steps*samples_per_step/250e6/0.1:.0f}x |")
    print(f"| 每步积分 | 0.1 ms | {samples_per_step/250e6*1000:.0f} ms | {samples_per_step/25000:.0f}x |")
    print(f"| 带宽 | 1.28 MHz | {nchan*250e6/samples_per_step/1e6:.1f} MHz | {nchan*250e6/samples_per_step/1.28e6:.0f}x |")
    print(f"| 时延校准 | ✗ | {'✓' if apply_delay_cal else '✗'} | - |")
    print()
    print(f"信噪比预期提升: √({samples_per_step/25000} × {ntime_steps/1000:.1f}) ≈ {np.sqrt(samples_per_step/25000 * ntime_steps/1000):.0f}x")
    print()
    
    return output_file

if __name__ == '__main__':
    import sys
    
    # 命令行参数
    if len(sys.argv) > 1:
        source_ra = float(sys.argv[1])
        source_dec = float(sys.argv[2])
        print(f"使用源位置: RA={source_ra}°, Dec={source_dec}°")
    else:
        source_ra = None
        source_dec = None
        print("未指定源位置，跳过时延校准")
        print("用法: python convert_9197_improved.py <RA> <Dec>")
        print("例如: python convert_9197_improved.py 83.6331 22.0145  # 蟹状星云")
    
    print()
    
    output_file = convert_with_improved_settings(
        ntime_steps=214,
        samples_per_step=2500000,
        nchan=512,
        apply_delay_cal=(source_ra is not None),
        source_ra=source_ra,
        source_dec=source_dec
    )
    
    print("✅ 转换完成！")
    print(f"输出文件: {output_file}")

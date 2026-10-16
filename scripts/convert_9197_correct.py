#!/usr/bin/env python3
"""
9197数据正确转换脚本

修正：
1. 从CSV正确读取天线坐标
2. 目标源是CasA (仙后座A)，不是蟹状星云
3. 使用ITRF坐标正确计算几何时延
4. 40ms积分 + 250MHz全带宽
"""

import numpy as np
import pandas as pd
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src/python"))
from tqdm import tqdm

# ═══════════════════════════════════════════════════════════════
# CasA (仙后座A) 坐标 - J2000
# ═══════════════════════════════════════════════════════════════
CASA_RA_DEG = 350.866    # 23h 23m 27.9s → 度
CASA_DEC_DEG = 58.812    # +58° 48' 42" → 度

# 光速
C_LIGHT = 299792458.0  # m/s

def load_antenna_positions_from_csv():
    """
    从CSV读取天线的ITRF坐标（地心坐标系）
    
    返回：
        positions: dict {天线名: [X, Y, Z]} 米（ITRF）
        station_lat_deg: 站点纬度（度）
        station_lon_deg: 站点经度（度）
    """
    csv_file = Path('antenna_coordinates.csv')
    df = pd.read_csv(csv_file)
    
    positions = {}
    for _, row in df.iterrows():
        name = row['名称']
        # 使用ITRF坐标（地心直角坐标）
        positions[name] = np.array([
            row['ITRF_X'],
            row['ITRF_Y'], 
            row['ITRF_Z']
        ])
    
    # Station frame must come from the same authoritative ITRF positions.
    from array_geometry import load_itrf_array
    geometry = load_itrf_array(csv_file, list(positions))
    station_lat = geometry['site_lat_deg']
    station_lon = geometry['site_lon_deg']
    
    return positions, station_lat, station_lon

def compute_source_direction_itrf(ra_deg, dec_deg, lst_hours):
    """
    计算源在ITRF坐标系中的方向向量
    
    参数:
        ra_deg: 赤经（度，J2000）
        dec_deg: 赤纬（度，J2000）
        lst_hours: 本地恒星时（小时）
    
    返回:
        s: 单位方向向量 [X, Y, Z] (ITRF)
    
    注意: 这是简化计算，忽略了岁差、章动等
    """
    # 转换为弧度
    ra = np.deg2rad(ra_deg)
    dec = np.deg2rad(dec_deg)
    
    # 时角 = LST - RA
    ha = np.deg2rad(lst_hours * 15) - ra
    
    # 源在地心惯性系(GEI)中的方向
    # 然后旋转到ITRF（简化：忽略地球自转在积分时间内的变化）
    
    # 在赤道坐标系中的单位向量
    x_eq = np.cos(dec) * np.cos(ra)
    y_eq = np.cos(dec) * np.sin(ra)
    z_eq = np.sin(dec)
    
    return np.array([x_eq, y_eq, z_eq])

def calculate_geometric_delays_itrf(positions, ra_deg, dec_deg, 
                                     station_lat_deg, station_lon_deg,
                                     obs_time_utc=None):
    """
    使用ITRF坐标计算几何时延
    
    对于小阵列（基线<100m），可以使用简化方法：
    τ_ij = (pos_j - pos_i) · s / c
    
    其中 s 是源方向的单位向量
    
    参数:
        positions: 天线ITRF坐标 {name: [X,Y,Z]}
        ra_deg, dec_deg: 源的赤道坐标（J2000）
        station_lat_deg, station_lon_deg: 站点坐标
        obs_time_utc: 观测时间（用于计算LST），如果None则假设源在子午圈
    
    返回:
        delays: 各天线相对参考天线的时延（秒）
        ant_names: 天线名列表
    """
    
    # 如果没有指定时间，假设源在子午圈附近（时角=0）
    # 这对于短积分是合理的近似
    if obs_time_utc is None:
        # 假设LST = RA（源在子午圈）
        lst_hours = ra_deg / 15.0
    else:
        # TODO: 使用astropy计算精确LST
        lst_hours = ra_deg / 15.0
    
    # 源方向（简化：使用赤道坐标的方向向量）
    dec_rad = np.deg2rad(dec_deg)
    ra_rad = np.deg2rad(ra_deg)
    
    # 在ITRF坐标系中，对于短基线，我们可以用
    # 源方向在本地水平坐标系中的投影
    
    # 站点纬度
    lat_rad = np.deg2rad(station_lat_deg)
    lon_rad = np.deg2rad(station_lon_deg)
    
    # 计算时角（假设源在子午圈，ha=0）
    ha = 0  # 简化假设
    
    # 源的地平坐标
    sin_alt = np.sin(dec_rad) * np.sin(lat_rad) + \
              np.cos(dec_rad) * np.cos(lat_rad) * np.cos(ha)
    alt = np.arcsin(sin_alt)
    
    cos_az = (np.sin(dec_rad) - np.sin(alt) * np.sin(lat_rad)) / \
             (np.cos(alt) * np.cos(lat_rad))
    cos_az = np.clip(cos_az, -1, 1)  # 数值保护
    az = np.arccos(cos_az)
    
    if np.sin(ha) > 0:
        az = 2 * np.pi - az
    
    # 源方向在本地ENU坐标系中的单位向量
    # E: 东, N: 北, U: 天顶
    s_E = np.cos(alt) * np.sin(az)
    s_N = np.cos(alt) * np.cos(az)
    s_U = np.sin(alt)
    s_enu = np.array([s_E, s_N, s_U])
    
    print(f"\n  源方向计算:")
    print(f"    高度角: {np.rad2deg(alt):.1f}°")
    print(f"    方位角: {np.rad2deg(az):.1f}°")
    print(f"    ENU方向向量: [{s_E:.4f}, {s_N:.4f}, {s_U:.4f}]")
    
    # 计算各天线相对于参考天线的位置（ENU坐标）
    ant_names = ['北3', '北4', '北5', '北6']
    
    # 参考天线（北3）的位置
    ref_name = ant_names[0]
    ref_itrf = positions[ref_name]
    
    # 将ITRF差转换为本地ENU坐标
    # 简化：对于小基线，直接用局部坐标差
    delays = []
    baselines_enu = []
    
    for name in ant_names:
        pos_itrf = positions[name]
        
        # ITRF坐标差（近似为局部ENU）
        # 对于小阵列，可以直接用差值
        # 更精确需要旋转矩阵
        delta_itrf = pos_itrf - ref_itrf
        
        # 简化的ITRF到ENU转换（对于站点纬度36.66°）
        # 更精确的转换需要考虑站点位置
        sin_lat = np.sin(lat_rad)
        cos_lat = np.cos(lat_rad)
        sin_lon = np.sin(lon_rad)
        cos_lon = np.cos(lon_rad)
        
        # ITRF到ENU的旋转矩阵
        R = np.array([
            [-sin_lon, cos_lon, 0],
            [-sin_lat*cos_lon, -sin_lat*sin_lon, cos_lat],
            [cos_lat*cos_lon, cos_lat*sin_lon, sin_lat]
        ])
        
        delta_enu = R @ delta_itrf
        baselines_enu.append(delta_enu)
        
        # 几何时延 = 基线投影到源方向 / 光速
        delay = np.dot(delta_enu, s_enu) / C_LIGHT
        delays.append(delay)
    
    delays = np.array(delays)
    
    # 打印基线信息
    print(f"\n  基线和时延:")
    for i, name in enumerate(ant_names):
        b = baselines_enu[i]
        print(f"    {name}: 基线=[{b[0]:+7.2f}E, {b[1]:+7.2f}N, {b[2]:+6.2f}U]m, "
              f"时延={delays[i]*1e9:+8.2f} ns")
    
    return delays, ant_names, baselines_enu

def apply_delay_correction(voltages, delays, sample_rate=250e6):
    """
    在频域应用时延校准
    
    原理：
        时域延迟 τ 对应频域相位旋转
        V_corrected(f) = V(f) × exp(-i 2π f τ)
    
    参数:
        voltages: 时域电压 (nsamples, nantennas)
        delays: 各天线时延 (nantennas,) 秒
        sample_rate: 采样率 Hz
    
    返回:
        corrected: 校准后的电压
    """
    nsamples, nantennas = voltages.shape
    corrected = np.zeros_like(voltages, dtype=np.complex64)
    
    # 频率轴
    freqs = np.fft.fftfreq(nsamples, 1/sample_rate)
    
    for iant in range(nantennas):
        if abs(delays[iant]) < 1e-15:
            # 参考天线，无需校准
            corrected[:, iant] = voltages[:, iant]
        else:
            # 相位校正因子
            phase_correction = np.exp(-2j * np.pi * freqs * delays[iant])
            
            # 频域校准
            V_freq = np.fft.fft(voltages[:, iant])
            V_corrected = V_freq * phase_correction
            corrected[:, iant] = np.fft.ifft(V_corrected)
    
    return corrected

def convert_fullband_40ms_correct(output_channels=1024, num_gulps=None):
    """
    正确的全带宽40ms转换
    """
    
    print("═══════════════════════════════════════════════════════════════")
    print("  9197数据正确转换：CasA目标 + 40ms积分 + 250MHz全带宽")
    print("═══════════════════════════════════════════════════════════════\n")
    
    # 参数
    sample_rate = 250e6
    gulp_time = 0.04  # 40ms
    samples_per_gulp = int(sample_rate * gulp_time)
    
    print("【目标源信息】")
    print(f"  名称: CasA (仙后座A)")
    print(f"  RA (J2000): {CASA_RA_DEG:.3f}° = 23h 23m 27.9s")
    print(f"  Dec (J2000): {CASA_DEC_DEG:.3f}° = +58° 48' 42\"")
    print()
    
    print("【转换参数】")
    print(f"  时间积分: {gulp_time*1000:.0f} ms/gulp")
    print(f"  每gulp采样点: {samples_per_gulp:,}")
    print(f"  采样率: {sample_rate/1e6:.0f} MHz")
    print(f"  输出通道数: {output_channels}")
    print()
    
    # 加载时域数据
    print("【步骤1: 加载时域数据】")
    time_data = np.load('data/processed/9197_file0_complete.npz')
    voltages_full = time_data['voltages']
    print(f"  数据shape: {voltages_full.shape}")
    
    # 计算gulp数
    max_gulps = voltages_full.shape[0] // samples_per_gulp
    if num_gulps is None or num_gulps > max_gulps:
        num_gulps = max_gulps
    
    total_samples = num_gulps * samples_per_gulp
    print(f"  处理gulp数: {num_gulps}")
    print(f"  总时长: {num_gulps * gulp_time:.3f} 秒")
    print()
    
    # 提取数据
    voltages_subset = voltages_full[:total_samples, 0, :, 0]
    
    # 加载天线坐标
    print("【步骤2: 加载天线坐标并计算时延】")
    positions, station_lat, station_lon = load_antenna_positions_from_csv()
    
    print(f"  站点位置: 纬度={station_lat:.4f}°, 经度={station_lon:.4f}°")
    print(f"  天线数量: {len(positions)}")
    
    # 打印使用的天线坐标
    print("\n  使用的天线ITRF坐标:")
    for name in ['北3', '北4', '北5', '北6']:
        pos = positions[name]
        print(f"    {name}: X={pos[0]:.3f}, Y={pos[1]:.3f}, Z={pos[2]:.3f}")
    
    # 计算几何时延
    delays, ant_names, baselines = calculate_geometric_delays_itrf(
        positions, 
        CASA_RA_DEG, 
        CASA_DEC_DEG,
        station_lat,
        station_lon
    )
    
    # 时延校准
    print("\n【步骤3: 应用时延校准】")
    print("  对每个gulp独立进行时延校准...")
    
    voltages_corrected = np.zeros_like(voltages_subset)
    
    for igulp in tqdm(range(num_gulps), desc="时延校准"):
        start = igulp * samples_per_gulp
        end = start + samples_per_gulp
        
        gulp_data = voltages_subset[start:end, :]
        gulp_corrected = apply_delay_correction(gulp_data, delays, sample_rate)
        voltages_corrected[start:end, :] = gulp_corrected
    
    print("  ✓ 时延校准完成")
    print()
    
    # FFT转换
    print("【步骤4: FFT转换（全带宽）】")
    voltages_reshaped = voltages_corrected.reshape(num_gulps, samples_per_gulp, 4)
    
    freq_bins = samples_per_gulp // 2
    bins_per_channel = freq_bins // output_channels
    actual_channels = freq_bins // bins_per_channel
    channel_bandwidth = bins_per_channel * sample_rate / samples_per_gulp
    
    print(f"  FFT长度: {samples_per_gulp:,}")
    print(f"  Nyquist bins: {freq_bins:,}")
    print(f"  输出通道: {actual_channels}")
    print(f"  bins/通道: {bins_per_channel}")
    print(f"  通道带宽: {channel_bandwidth/1e3:.1f} kHz")
    print(f"  总带宽: {actual_channels * channel_bandwidth/1e6:.1f} MHz")
    print()
    
    fft_result = np.zeros((num_gulps, actual_channels, 4), dtype=np.complex64)
    
    for igulp in tqdm(range(num_gulps), desc="FFT处理"):
        for iant in range(4):
            fft_full = np.fft.fft(voltages_reshaped[igulp, :, iant])
            fft_positive = fft_full[:freq_bins]
            
            for ichan in range(actual_channels):
                start_bin = ichan * bins_per_channel
                end_bin = start_bin + bins_per_channel
                fft_result[igulp, ichan, iant] = np.mean(fft_positive[start_bin:end_bin])
    
    print("  ✓ FFT完成")
    print()
    
    # 填充到256天线
    print("【步骤5: 填充到EPIC格式】")
    epic_data = np.zeros((num_gulps, actual_channels, 256, 2), dtype=np.complex64)
    
    for iant in range(4):
        epic_data[:, :, iant, 0] = fft_result[:, :, iant]
        epic_data[:, :, iant, 1] = fft_result[:, :, iant]
    
    print(f"  输出shape: {epic_data.shape}")
    print()
    
    # 保存
    meta = np.array([num_gulps, 0], dtype=np.int64)
    output_file = f'data/epic_ready/9197_CasA_{num_gulps}gulps_{actual_channels}chan_fullband.npz'
    
    print(f"【步骤6: 保存】")
    print(f"  输出文件: {output_file}")
    
    np.savez_compressed(output_file, data=epic_data, meta=meta)
    
    import os
    file_size = os.path.getsize(output_file) / 1e6
    print(f"  文件大小: {file_size:.1f} MB")
    print()
    
    # 总结
    print("═══════════════════════════════════════════════════════════════")
    print("  转换完成！")
    print("═══════════════════════════════════════════════════════════════")
    print()
    print("【关键指标】")
    print(f"  ✓ 目标源: CasA (RA={CASA_RA_DEG}°, Dec={CASA_DEC_DEG}°)")
    print(f"  ✓ 时间积分: {gulp_time*1000:.0f} ms/gulp")
    print(f"  ✓ 总时长: {num_gulps * gulp_time:.3f} 秒")
    print(f"  ✓ 带宽: {actual_channels * channel_bandwidth/1e6:.0f} MHz (100%)")
    print(f"  ✓ 通道数: {actual_channels}")
    print(f"  ✓ 时延校准: 基于CasA位置的几何时延")
    print()
    
    return output_file, num_gulps, actual_channels

if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1:
        output_channels = int(sys.argv[1])
    else:
        output_channels = 1024
    
    if len(sys.argv) > 2:
        num_gulps = int(sys.argv[2])
    else:
        num_gulps = None
    
    output_file, ngulps, nchans = convert_fullband_40ms_correct(
        output_channels=output_channels,
        num_gulps=num_gulps
    )
    
    print("✅ 转换完成！")
    print(f"\n下一步:")
    print(f"  cp {output_file} build/data/40ms_128chan_600offset_gulp_c64_virtransit.npz")
    print(f"  cd build && ./epic++ --offline --channels {min(nchans, 1024)} --nts {ngulps} ...")

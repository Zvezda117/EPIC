#!/usr/bin/env python3
"""
9197数据迭代自校准

方法：
1. 假设源是点源，可见度相位 vs 频率应该是线性的
2. 线性斜率 = 2π × 时延
3. 从基线时延解出各天线的相对时延
4. 迭代直到收敛

优点：
- 不需要几何模型（天线坐标、源位置）
- 自动包含所有时延（几何 + 链路 + 电缆）
- 利用250MHz宽带的优势，时延精度极高
"""

import numpy as np
from tqdm import tqdm
from pathlib import Path

def compute_visibility(voltages, ant_i, ant_j):
    """
    计算基线可见度 V_ij = E_i × E_j*
    
    参数:
        voltages: (nsamples, nant) 时域电压
        ant_i, ant_j: 天线索引
    
    返回:
        vis: 复数可见度
    """
    return voltages[:, ant_i] * np.conj(voltages[:, ant_j])

def compute_visibility_spectrum(voltages, ant_i, ant_j, fft_size=None):
    """
    计算频域可见度谱
    
    参数:
        voltages: (nsamples, nant) 时域电压
        fft_size: FFT长度，默认=nsamples
    
    返回:
        vis_spectrum: (nfreq,) 复数，频域可见度
        freqs: (nfreq,) 频率轴 (Hz)
    """
    nsamples = voltages.shape[0]
    if fft_size is None:
        fft_size = nsamples
    
    # FFT
    E_i = np.fft.fft(voltages[:fft_size, ant_i])
    E_j = np.fft.fft(voltages[:fft_size, ant_j])
    
    # 可见度 = E_i × E_j*
    vis = E_i * np.conj(E_j)
    
    # 只取正频率
    nfreq = fft_size // 2
    vis_positive = vis[:nfreq]
    
    sample_rate = 250e6
    freqs = np.fft.fftfreq(fft_size, 1/sample_rate)[:nfreq]
    
    return vis_positive, freqs

def fit_delay_from_phase(freqs, phases):
    """
    从相位 vs 频率的线性关系拟合时延
    
    相位模型: φ = 2π × f × τ + φ_0
    
    参数:
        freqs: 频率轴 (Hz)
        phases: 相位 (rad)
    
    返回:
        delay: 时延 (秒)
        phase_offset: 常数相位偏移 (rad)
        residual: 残差标准差 (rad)
    """
    # 相位展开（处理2π跳变）
    phases_unwrapped = np.unwrap(phases)
    
    # 线性拟合: phase = slope * freq + offset
    # slope = 2π × delay
    coeffs = np.polyfit(freqs, phases_unwrapped, 1)
    slope = coeffs[0]
    offset = coeffs[1]
    
    delay = slope / (2 * np.pi)
    
    # 计算残差
    fitted = slope * freqs + offset
    residual = np.std(phases_unwrapped - fitted)
    
    return delay, offset, residual

def solve_antenna_delays(baseline_delays, nant):
    """
    从基线时延解出各天线的相对时延
    
    模型: τ_ij = τ_i - τ_j
    约束: τ_0 = 0 (参考天线)
    
    参数:
        baseline_delays: dict {(i,j): τ_ij}
        nant: 天线数
    
    返回:
        ant_delays: (nant,) 各天线时延
    """
    # 构建方程组 A @ x = b
    # x = [τ_1, τ_2, ..., τ_{n-1}] (不包含τ_0=0)
    
    equations = list(baseline_delays.keys())
    neq = len(equations)
    nunk = nant - 1  # 未知数 = 天线数 - 1
    
    A = np.zeros((neq, nunk))
    b = np.zeros(neq)
    
    for eq_idx, (i, j) in enumerate(equations):
        b[eq_idx] = baseline_delays[(i, j)]
        
        # τ_ij = τ_i - τ_j
        # 如果i=0，τ_i=0，方程变为 -τ_j
        # 如果j=0，τ_j=0，方程变为 τ_i
        
        if i > 0:
            A[eq_idx, i - 1] = 1
        if j > 0:
            A[eq_idx, j - 1] = -1
    
    # 最小二乘解
    x, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
    
    # 重建完整的天线时延数组
    ant_delays = np.zeros(nant)
    ant_delays[1:] = x
    
    return ant_delays

def apply_delay_correction(voltages, delays, sample_rate=250e6):
    """
    在频域应用时延校准
    """
    nsamples, nant = voltages.shape
    corrected = np.zeros_like(voltages)
    
    freqs = np.fft.fftfreq(nsamples, 1/sample_rate)
    
    for iant in range(nant):
        if abs(delays[iant]) < 1e-15:
            corrected[:, iant] = voltages[:, iant]
        else:
            phase_correction = np.exp(-2j * np.pi * freqs * delays[iant])
            V_freq = np.fft.fft(voltages[:, iant])
            V_corrected = V_freq * phase_correction
            corrected[:, iant] = np.fft.ifft(V_corrected)
    
    return corrected

def self_calibrate(voltages, max_iter=10, tol=1e-12, verbose=True):
    """
    迭代自校准
    
    参数:
        voltages: (nsamples, nant) 时域电压
        max_iter: 最大迭代次数
        tol: 时延更新收敛阈值 (秒)
        verbose: 是否打印详细信息
    
    返回:
        delays: (nant,) 最终时延估计
        calibrated: 校准后的电压
        history: 迭代历史
    """
    nsamples, nant = voltages.shape
    sample_rate = 250e6
    
    # 累积时延估计
    total_delays = np.zeros(nant)
    
    # 当前电压（会被逐步校准）
    current_voltages = voltages.copy()
    
    history = []
    
    if verbose:
        print("\n【迭代自校准】")
        print(f"  数据: {nsamples:,} 采样点, {nant} 天线")
        print(f"  FFT长度: {nsamples:,}")
        print(f"  频率分辨率: {sample_rate/nsamples:.1f} Hz")
        print()
    
    for iteration in range(max_iter):
        if verbose:
            print(f"━━━ 迭代 {iteration + 1} ━━━")
        
        # 步骤1: 计算所有基线的频域可见度
        baseline_delays = {}
        residuals = {}
        
        for i in range(nant):
            for j in range(i + 1, nant):
                # 计算可见度谱
                vis, freqs = compute_visibility_spectrum(
                    current_voltages, i, j, fft_size=nsamples
                )
                
                # 提取相位
                phases = np.angle(vis)
                
                # 拟合时延
                delay, offset, resid = fit_delay_from_phase(freqs, phases)
                
                baseline_delays[(i, j)] = delay
                residuals[(i, j)] = resid
                
                if verbose:
                    print(f"    基线{i}-{j}: τ={delay*1e9:+8.3f} ns, "
                          f"残差={np.rad2deg(resid):.2f}°")
        
        # 步骤2: 从基线时延解出天线时延
        ant_delays_update = solve_antenna_delays(baseline_delays, nant)
        
        if verbose:
            print(f"\n  天线时延更新:")
            for i in range(nant):
                print(f"    天线{i}: Δτ={ant_delays_update[i]*1e9:+8.3f} ns")
        
        # 步骤3: 应用校准
        current_voltages = apply_delay_correction(
            current_voltages, ant_delays_update, sample_rate
        )
        
        # 累积总时延
        total_delays += ant_delays_update
        
        # 记录历史
        max_update = np.max(np.abs(ant_delays_update))
        avg_residual = np.mean(list(residuals.values()))
        history.append({
            'iteration': iteration + 1,
            'delays': total_delays.copy(),
            'max_update': max_update,
            'avg_residual': avg_residual
        })
        
        if verbose:
            print(f"\n  累积时延:")
            for i in range(nant):
                print(f"    天线{i}: τ={total_delays[i]*1e9:+8.3f} ns")
            print(f"\n  最大更新: {max_update*1e9:.4f} ns")
            print(f"  平均残差: {np.rad2deg(avg_residual):.3f}°")
        
        # 检查收敛
        if max_update < tol:
            if verbose:
                print(f"\n  ✓ 收敛！更新 < {tol*1e9:.1f} ns")
            break
        
        print()
    
    return total_delays, current_voltages, history

def main():
    """主函数"""
    print("═══════════════════════════════════════════════════════════════")
    print("  9197数据迭代自校准")
    print("═══════════════════════════════════════════════════════════════\n")
    
    # 加载时域数据
    print("【加载数据】")
    time_data = np.load('data/processed/9197_file0_complete.npz')
    voltages_full = time_data['voltages']
    print(f"  数据shape: {voltages_full.shape}")
    
    # 只用一个40ms gulp进行自校准（足够了）
    sample_rate = 250e6
    gulp_samples = int(sample_rate * 0.04)  # 10M样本
    
    # 实际上，为了速度，我们先用更短的数据测试
    # 1ms = 250000 样本，足以看到效果
    test_samples = 250000  # 1ms数据
    
    print(f"  使用前 {test_samples:,} 采样点 ({test_samples/sample_rate*1000:.1f} ms) 进行校准")
    
    # 提取数据
    voltages = voltages_full[:test_samples, 0, :, 0]
    print(f"  校准数据shape: {voltages.shape}")
    print()
    
    # 运行自校准
    delays, calibrated, history = self_calibrate(
        voltages, 
        max_iter=5,
        tol=1e-12,
        verbose=True
    )
    
    # 总结
    print("\n═══════════════════════════════════════════════════════════════")
    print("  自校准结果")
    print("═══════════════════════════════════════════════════════════════\n")
    
    ant_names = ['北3', '北4', '北5', '北6']
    print("【最终时延估计（相对北3）】")
    for i, name in enumerate(ant_names):
        samples = delays[i] * sample_rate
        phase_1355 = delays[i] * 1355e6 * 2 * np.pi
        print(f"  {name}: {delays[i]*1e9:+10.3f} ns = {samples:+8.1f} 采样点 "
              f"= {np.rad2deg(phase_1355):+10.1f}° @1355MHz")
    
    print("\n【与几何模型对比】")
    print("  几何模型（子午圈假设）:")
    print("    北3:    +0.00 ns")
    print("    北4:    +0.28 ns") 
    print("    北5:   -60.98 ns")
    print("    北6:   -61.16 ns")
    print()
    print("  差异 = 自校准 - 几何模型 = 链路时延！")
    
    # 保存结果
    output_file = 'data/calibration/9197_self_cal_delays.npz'
    Path('data/calibration').mkdir(parents=True, exist_ok=True)
    np.savez(output_file,
             delays=delays,
             ant_names=ant_names,
             history=history)
    print(f"\n结果已保存到: {output_file}")
    
    return delays

if __name__ == '__main__':
    delays = main()

#!/usr/bin/env python3
"""
将9197数据转换为EPIC兼容格式

输入：时域基带数据 (nsamples, 1, 4, 1)  
输出：频域成像数据 (ntime, nchan, 256, 2)

处理步骤：
1. 时域分段
2. FFT转频域
3. 填充到256天线×2极化
"""

import numpy as np
import os
import sys
import argparse


def convert_to_epic(input_file, output_file, nchan=128, ntime_samples=25000):
    """
    转换数据格式
    
    Args:
        input_file: 输入NPZ文件
        output_file: 输出NPZ文件
        nchan: 输出频率通道数
        ntime_samples: 每个时间步包含多少采样点
    """
    print("="*70)
    print("将9197数据转换为EPIC格式")
    print("="*70)
    
    # 加载数据
    print(f"\n1. 加载数据from {input_file}...")
    data = np.load(input_file, allow_pickle=True)
    voltages = data['voltages']
    
    print(f"   输入形状: {voltages.shape}")
    
    nsamples_total = voltages.shape[0]
    nstands_input = voltages.shape[2]
    
    # 计算时间步数
    ntime_chunks = nsamples_total // ntime_samples
    useful_samples = ntime_chunks * ntime_samples
    
    print(f"   总采样点: {nsamples_total:,}")
    print(f"   每时间步: {ntime_samples:,} 采样点")
    print(f"   时间步数: {ntime_chunks}")
    print(f"   有效采样: {useful_samples:,} ({useful_samples/nsamples_total*100:.1f}%)")
    
    # 计算时间统计
    采样率 = 250e6
    总时间 = nsamples_total / 采样率
    时间步长 = ntime_samples / 采样率
    
    print(f"   总时长: {总时间:.3f} 秒")
    print(f"   每步时长: {时间步长*1000:.2f} ms")
    
    # 步骤2：FFT转换
    print(f"\n2. FFT转换到频域...")
    print(f"   目标频道数: {nchan}")
    
    # 重塑数据
    voltages_reshaped = voltages[:useful_samples].reshape(
        ntime_chunks, ntime_samples, 1, nstands_input, 1
    )
    
    # 初始化FFT结果
    fft_result = np.zeros((ntime_chunks, nchan, nstands_input, 1), dtype=np.complex64)
    
    print("   处理中...")
    for i in range(ntime_chunks):
        for ant in range(nstands_input):
            # 取出时间序列
            timeseries = voltages_reshaped[i, :, 0, ant, 0]
            
            # FFT
            fft_full = np.fft.fft(timeseries)
            
            # 取前nchan个频率bin（正频率）
            fft_result[i, :, ant, 0] = fft_full[:nchan]
        
        if (i + 1) % 1000 == 0 or (i + 1) == ntime_chunks:
            print(f"   进度: {i+1}/{ntime_chunks} ({(i+1)/ntime_chunks*100:.1f}%)")
    
    print(f"   FFT完成！形状: {fft_result.shape}")
    
    # 步骤3：填充到256天线×2极化
    print(f"\n3. 填充到EPIC格式 (256天线, 2极化)...")
    
    epic_data = np.zeros((ntime_chunks, nchan, 256, 2), dtype=np.complex64)
    
    # 把4个天线放到前4个位置
    for ant in range(nstands_input):
        # 两个极化都用同样的数据（因为我们只有单极化）
        epic_data[:, :, ant, 0] = fft_result[:, :, ant, 0]  # H极化
        epic_data[:, :, ant, 1] = fft_result[:, :, ant, 0]  # V极化（复制）
    
    print(f"   ✓ 前4个天线填充完成")
    print(f"   ✓ 后252个天线保持为0（EPIC会自动忽略）")
    
    final_shape = epic_data.shape
    print(f"   最终形状: {final_shape}")
    print(f"   期望格式: (ntime, nchan, 256, 2)")
    
    # 步骤4：保存
    print(f"\n4. 保存到 {output_file}...")
    
    # EPIC期望的元数据格式（从示例文件学习）
    meta = np.array([ntime_chunks, 0], dtype=np.int64)
    
    np.savez_compressed(
        output_file,
        data=epic_data,
        meta=meta
    )
    
    file_size_mb = os.path.getsize(output_file) / 1e6
    
    print(f"   ✓ 保存成功！")
    print(f"   文件大小: {file_size_mb:.2f} MB")
    
    # 统计
    print(f"\n" + "="*70)
    print("转换完成！")
    print("="*70)
    print(f"\n数据统计:")
    print(f"  时间步数: {ntime_chunks}")
    print(f"  频率通道数: {nchan}")
    print(f"  天线数: 256 (实际有效: 4)")
    print(f"  极化数: 2 (实际有效: 1)")
    print(f"  数据占用率: {4*1 / (256*2) * 100:.1f}% (4天线1极化 / 256天线2极化)")
    
    # 给出EPIC运行建议
    print(f"\n建议的EPIC参数:")
    print(f"  --imagesize 32")
    print(f"  --imageres 10.0  # 10度/像素（分辨率很粗）")
    print(f"  --channels {nchan}")
    print(f"  --nts {ntime_chunks}")
    print(f"  --seq_accum {int(时间步长*1000)}")
    
    return epic_data


def main():
    parser = argparse.ArgumentParser(
        description='Convert 9197 data to EPIC format',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('input', type=str,
                       help='Input NPZ file (from process_9197_single_channel.py)')
    parser.add_argument('-o', '--output', type=str,
                       default='9197_epic_format.npz',
                       help='Output NPZ file for EPIC')
    parser.add_argument('--nchan', type=int, default=512,
                        help='Number of frequency channels (推荐512-1024用于250MHz带宽)')
    parser.add_argument('--ntime-samples', type=int, default=25000,
                       help='Samples per time step')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}")
        sys.exit(1)
    
    convert_to_epic(args.input, args.output, args.nchan, args.ntime_samples)
    
    print(f"\n下一步:")
    print(f"  cd build")
    print(f"  ./epic++ --offline --npytbnfile ../{args.output} \\")
    print(f"           --out_dir ../9197_images \\")
    print(f"           --imagesize 32 --imageres 10.0 --channels {args.nchan}")


if __name__ == '__main__':
    main()

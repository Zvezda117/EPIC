#!/usr/bin/env python3
"""
验证天线接线的正确性

检查：
1. 每个文件的极化是否符合预期
2. "同一天线不同极化"的相关性（应该低）
3. 数据质量和信号强度
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

try:
    from UnpackCustom import UnpackCustom
except ImportError:
    print("ERROR: UnpackCustom not found")
    print("Please ensure UnpackCustom.py is in the same directory")
    sys.exit(1)


def analyze_file(file_path, npackets=100):
    """
    分析单个基带文件
    
    Returns:
        dict with analysis results
    """
    print(f"\n{'='*60}")
    print(f"Analyzing: {os.path.basename(file_path)}")
    print('='*60)
    
    # 读取包信息
    pack_length, pack_channel_num, pack_datatype, file_size = \
        UnpackCustom.onepack_read(file_path)
    
    total_packets = int(file_size / pack_length)
    print(f"File size: {file_size / 1e9:.2f} GB")
    print(f"Total packets: {total_packets}")
    print(f"Channels per packet: {pack_channel_num}")
    print(f"Analyzing first {npackets} packets...")
    
    # 读取数据
    base_iq_head, base_iq_data = UnpackCustom.read_and_unpack(
        file_path, 0, npackets-1, pack_length, 
        pack_datatype, file_size, pack_channel_num
    )
    
    # 分离极化
    if pack_channel_num >= 2:
        pol0_data = []
        pol1_data = []
        
        for i in range(len(base_iq_data)):
            if i % pack_channel_num == 0:
                pol0_data.append(base_iq_data[i])
            elif i % pack_channel_num == 1:
                pol1_data.append(base_iq_data[i])
        
        pol0_data = np.concatenate(pol0_data)
        pol1_data = np.concatenate(pol1_data)
    else:
        pol0_data = np.concatenate(base_iq_data)
        pol1_data = None
    
    # 计算统计量
    results = {
        'file': os.path.basename(file_path),
        'npackets_total': total_packets,
        'nchannels': pack_channel_num,
    }
    
    print(f"\nPol 0 Statistics:")
    print(f"  Length: {len(pol0_data)}")
    print(f"  Mean power: {np.abs(pol0_data).mean():.6f}")
    print(f"  Std power: {np.abs(pol0_data).std():.6f}")
    print(f"  Max power: {np.abs(pol0_data).max():.6f}")
    
    results['pol0'] = {
        'length': len(pol0_data),
        'mean_power': float(np.abs(pol0_data).mean()),
        'std_power': float(np.abs(pol0_data).std()),
        'max_power': float(np.abs(pol0_data).max()),
    }
    
    if pol1_data is not None:
        print(f"\nPol 1 Statistics:")
        print(f"  Length: {len(pol1_data)}")
        print(f"  Mean power: {np.abs(pol1_data).mean():.6f}")
        print(f"  Std power: {np.abs(pol1_data).std():.6f}")
        print(f"  Max power: {np.abs(pol1_data).max():.6f}")
        
        results['pol1'] = {
            'length': len(pol1_data),
            'mean_power': float(np.abs(pol1_data).mean()),
            'std_power': float(np.abs(pol1_data).std()),
            'max_power': float(np.abs(pol1_data).max()),
        }
        
        # 计算互相关
        n_corr = min(10000, len(pol0_data), len(pol1_data))
        corr = np.corrcoef(
            np.abs(pol0_data[:n_corr]),
            np.abs(pol1_data[:n_corr])
        )[0, 1]
        
        print(f"\nCross-correlation (power): {corr:.4f}")
        print(f"  Interpretation:")
        if corr > 0.7:
            print(f"    HIGH - Likely same antenna, different pols")
        elif corr > 0.3:
            print(f"    MEDIUM - Possibly correlated signals")
        else:
            print(f"    LOW - Likely different antennas")
        
        results['cross_correlation'] = float(corr)
        
        # 简单频谱分析
        print(f"\nSimple spectrum (first 1000 samples):")
        fft0 = np.fft.fft(pol0_data[:1000])
        fft1 = np.fft.fft(pol1_data[:1000])
        
        print(f"  Pol0 DC component: {np.abs(fft0[0]):.2f}")
        print(f"  Pol1 DC component: {np.abs(fft1[0]):.2f}")
        print(f"  Pol0 peak freq bin: {np.argmax(np.abs(fft0[1:100]))+1}")
        print(f"  Pol1 peak freq bin: {np.argmax(np.abs(fft1[1:100]))+1}")
        
        results['pol0']['dc_component'] = float(np.abs(fft0[0]))
        results['pol1']['dc_component'] = float(np.abs(fft1[0]))
    
    return results, pol0_data, pol1_data


def check_antenna_consistency():
    """
    检查声称是同一天线的不同极化之间的相关性
    
    根据接线图：
    - 东4: HBaseband1Spec1_pol0 (H) vs VBaseband0Spec0_pol0 (V)
    - 东5: HBaseband2Spec2_pol0 (H) vs VBaseband0Spec0_pol1 (V)
    """
    print("\n" + "="*70)
    print("CHECKING ANTENNA CONSISTENCY")
    print("="*70)
    
    data_dir = Path("/mnt/data_ssd/zdj/data/crab")
    
    # 检查东4
    print("\n### Checking 东4 (H vs V) ###")
    file_dong4_h = data_dir / "HBaseband1Spec1_4304.data"
    file_dong4_v = data_dir / "VBaseband0Spec0_4304.data"
    
    if file_dong4_h.exists() and file_dong4_v.exists():
        _, dong4_h_pol0, _ = analyze_file(str(file_dong4_h), npackets=50)
        _, dong4_v_pol0, _ = analyze_file(str(file_dong4_v), npackets=50)
        
        n_corr = min(5000, len(dong4_h_pol0), len(dong4_v_pol0))
        corr = np.corrcoef(
            np.abs(dong4_h_pol0[:n_corr]),
            np.abs(dong4_v_pol0[:n_corr])
        )[0, 1]
        
        print(f"\n东4 H-V correlation: {corr:.4f}")
        if corr > 0.5:
            print("  ✓ HIGH correlation - consistent with same antenna")
        else:
            print("  ✗ LOW correlation - may be different antennas!")
    
    # 检查东5
    print("\n### Checking 东5 (H vs V) ###")
    file_dong5_h = data_dir / "HBaseband2Spec2_4304.data"
    file_dong5_v = data_dir / "VBaseband0Spec0_4304.data"
    
    if file_dong5_h.exists() and file_dong5_v.exists():
        _, dong5_h_pol0, _ = analyze_file(str(file_dong5_h), npackets=50)
        # 注意：东5V是pol1
        _, _, dong5_v_pol1 = analyze_file(str(file_dong5_v), npackets=50)
        
        n_corr = min(5000, len(dong5_h_pol0), len(dong5_v_pol1))
        corr = np.corrcoef(
            np.abs(dong5_h_pol0[:n_corr]),
            np.abs(dong5_v_pol1[:n_corr])
        )[0, 1]
        
        print(f"\n东5 H-V correlation: {corr:.4f}")
        if corr > 0.5:
            print("  ✓ HIGH correlation - consistent with same antenna")
        else:
            print("  ✗ LOW correlation - may be different antennas!")


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Verify antenna wiring')
    parser.add_argument('--data-dir', type=str, 
                       default='/mnt/data_ssd/zdj/data/crab',
                       help='Data directory')
    parser.add_argument('--npackets', type=int, default=100,
                       help='Number of packets to analyze per file')
    parser.add_argument('--check-consistency', action='store_true',
                       help='Check cross-file antenna consistency')
    
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    
    if not data_dir.exists():
        print(f"ERROR: Data directory not found: {data_dir}")
        sys.exit(1)
    
    # 查找所有数据文件
    data_files = sorted(data_dir.glob('*Baseband*.data'))
    
    if len(data_files) == 0:
        print(f"ERROR: No data files found in {data_dir}")
        sys.exit(1)
    
    print(f"Found {len(data_files)} data files")
    
    # 分析每个文件
    all_results = []
    for file_path in data_files:
        try:
            result, _, _ = analyze_file(str(file_path), args.npackets)
            all_results.append(result)
        except Exception as e:
            print(f"ERROR analyzing {file_path}: {e}")
            import traceback
            traceback.print_exc()
    
    # 打印汇总
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    for result in all_results:
        print(f"\n{result['file']}:")
        print(f"  Channels: {result['nchannels']}")
        print(f"  Pol0 mean power: {result['pol0']['mean_power']:.6f}")
        if 'pol1' in result:
            print(f"  Pol1 mean power: {result['pol1']['mean_power']:.6f}")
            print(f"  Cross-correlation: {result.get('cross_correlation', 'N/A')}")
    
    # 检查一致性
    if args.check_consistency:
        check_antenna_consistency()
    
    print("\n" + "="*70)
    print("VERIFICATION COMPLETE")
    print("="*70)
    print("\nNext steps:")
    print("1. Review the cross-correlations")
    print("2. If correlations are LOW (<0.3), the wiring diagram is correct")
    print("3. If correlations are HIGH (>0.7), same antenna's pols are split")
    print("4. Update antenna_mapping.yaml accordingly")


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
处理9197数据 - 单通道格式（无需解交织）

特点：
- 每个文件只有1个通道
- 4个天线: 北3, 4, 5, 6
- 137个时间序列文件
"""

import os
import sys
import yaml
import numpy as np
from pathlib import Path
import argparse

try:
    from UnpackCustom import UnpackCustom
except ImportError:
    print("ERROR: UnpackCustom not found")
    sys.exit(1)


class SimpleDataProcessor:
    """简化的单通道数据处理器"""
    
    def __init__(self, config_file='antenna_mapping_9197.yaml'):
        with open(config_file, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.data_dir = Path(self.config['observation']['data_dir'])
        self.terminal_map = self.config['terminal_mapping']
        self.nstands = self.config['station']['nstands']
        
        print(f"Initialized for {self.nstands} antennas")
        print(f"Data directory: {self.data_dir}")
    
    def process_single_file(self, file_path, pack_start=0, pack_end=None):
        """
        处理单个文件（只有1个通道，超简单！）
        
        Returns:
            复数数据数组
        """
        print(f"Processing: {os.path.basename(file_path)}")
        
        # 读取包信息
        pack_length, pack_channel_num, pack_datatype, file_size = \
            UnpackCustom.onepack_read(file_path, DEBUG=False)
        
        if pack_channel_num != 1:
            raise ValueError(f"Expected 1 channel, got {pack_channel_num}")
        
        if pack_end is None:
            pack_end = int(file_size / pack_length) - 1
        
        # 读取数据
        base_iq_head, base_iq_data = UnpackCustom.read_and_unpack(
            file_path, pack_start, pack_end, pack_length,
            pack_datatype, file_size, pack_channel_num
        )
        
        # 合并所有包的数据
        data = np.concatenate(base_iq_data)
        
        print(f"  Read {len(base_iq_data)} packets")
        print(f"  Total samples: {len(data)}")
        print(f"  Mean power: {np.abs(data).mean():.6f}")
        
        return data
    
    def process_antenna_group(self, file_pattern, file_numbers, pack_start=0, pack_end=None):
        """
        处理一个天线的多个文件
        
        Args:
            file_pattern: 文件名模式，如 "HBaseband2Spec2"
            file_numbers: 文件编号列表，如 [0, 1, 2]
            
        Returns:
            所有文件合并后的数据
        """
        all_data = []
        
        for file_num in file_numbers:
            file_path = self.data_dir / f"{file_pattern}_{file_num}.data"
            
            if not file_path.exists():
                print(f"Warning: {file_path} not found, skipping")
                continue
            
            try:
                data = self.process_single_file(str(file_path), pack_start, pack_end)
                all_data.append(data)
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
                continue
        
        if len(all_data) == 0:
            raise ValueError("No data successfully processed")
        
        # 合并所有文件
        combined = np.concatenate(all_data)
        print(f"\nCombined {len(all_data)} files:")
        print(f"  Total samples: {len(combined)}")
        print(f"  Duration: ~{len(combined) / 500e6:.2f} seconds (assuming 500MHz sampling)")
        
        return combined
    
    def process_all_antennas(self, file_numbers=[0], pack_start=0, pack_end=None, output_file=None):
        """
        处理所有4个天线的数据
        
        Args:
            file_numbers: 要处理的文件编号列表
            pack_start: 起始包号
            pack_end: 结束包号
            output_file: 输出NPZ文件
        """
        print("\n" + "="*70)
        print("Processing All Antennas")
        print("="*70)
        
        # 定义天线和文件模式的映射
        antenna_files = {
            "北3": "HBaseband2Spec2",
            "北4": "VBaseband0Spec0", 
            "北5": "VBaseband1Spec1",
            "北6": "VBaseband2Spec2"
        }
        
        # 处理每个天线
        antenna_data = {}
        for ant_name, file_pattern in antenna_files.items():
            print(f"\n### Processing {ant_name} ({file_pattern}) ###")
            try:
                data = self.process_antenna_group(
                    file_pattern, file_numbers, pack_start, pack_end
                )
                antenna_data[ant_name] = data
            except Exception as e:
                print(f"Failed to process {ant_name}: {e}")
                continue
        
        if len(antenna_data) == 0:
            raise ValueError("No antennas successfully processed")
        
        # 检查所有天线数据长度是否一致
        lengths = {name: len(data) for name, data in antenna_data.items()}
        print(f"\nData lengths: {lengths}")
        
        min_length = min(lengths.values())
        if max(lengths.values()) != min_length:
            print(f"Warning: Data lengths differ, truncating to {min_length}")
        
        # 构建EPIC格式数据
        # voltages shape: (ntime, nchan=1, nstands=4, npol=1)
        nsamples = min_length
        voltages = np.zeros((nsamples, 1, self.nstands, 1), dtype=np.complex64)
        
        # 按顺序填充（北3, 4, 5, 6 → 索引0, 1, 2, 3）
        for i, ant_name in enumerate(["北3", "北4", "北5", "北6"]):
            if ant_name in antenna_data:
                voltages[:, 0, i, 0] = antenna_data[ant_name][:nsamples]
                print(f"  Filled {ant_name} → stand {i}")
        
        # 天线位置（从CSV读取）
        positions = self.get_antenna_positions()
        
        # 保存
        if output_file is None:
            output_file = f"9197_processed_files{file_numbers[0]}-{file_numbers[-1]}.npz"
        
        print(f"\n=== Saving to {output_file} ===")
        np.savez_compressed(
            output_file,
            voltages=voltages,
            antenna_positions=positions,
            file_numbers=np.array(file_numbers),
            config=self.config
        )
        
        # 统计
        file_size_mb = os.path.getsize(output_file) / 1e6
        print(f"\nOutput statistics:")
        print(f"  Voltages shape: {voltages.shape}")
        print(f"  Antenna positions shape: {positions.shape}")
        print(f"  File size: {file_size_mb:.2f} MB")
        print(f"  Samples per antenna: {nsamples:,}")
        print(f"  Duration: ~{nsamples / 500e6:.3f} seconds")
        
        return voltages, positions
    
    def get_antenna_positions(self):
        """读取天线位置"""
        import pandas as pd
        
        coord_file = 'antenna_coordinates.csv'
        if not os.path.exists(coord_file):
            print("Warning: antenna_coordinates.csv not found, using zeros")
            return np.zeros((self.nstands, 3))
        
        df = pd.read_csv(coord_file)
        
        positions = np.zeros((self.nstands, 3))
        for i, ant_name in enumerate(["北3", "北4", "北5", "北6"]):
            row = df[df['名称'] == ant_name]
            if len(row) > 0:
                positions[i] = [
                    row['X/E'].values[0],
                    row['Y/N'].values[0],
                    row['Z/U'].values[0]
                ]
        
        # 相对位置
        positions -= positions[0]
        return positions


def main():
    parser = argparse.ArgumentParser(
        description='Process 9197 single-channel data',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--files', type=str, default='0',
                       help='File numbers to process (e.g. "0" or "0,1,2" or "0-10")')
    parser.add_argument('-o', '--output', type=str, 
                       default=None,
                       help='Output NPZ file')
    parser.add_argument('--pack-start', type=int, default=0,
                       help='Starting packet number')
    parser.add_argument('--pack-end', type=int, default=None,
                       help='Ending packet number (None=all)')
    parser.add_argument('--config', type=str,
                       default='antenna_mapping_9197.yaml',
                       help='Config file')
    
    args = parser.parse_args()
    
    # 解析文件编号
    if '-' in args.files:
        start, end = map(int, args.files.split('-'))
        file_numbers = list(range(start, end + 1))
    elif ',' in args.files:
        file_numbers = [int(x) for x in args.files.split(',')]
    else:
        file_numbers = [int(args.files)]
    
    print(f"Processing files: {file_numbers}")
    
    # 处理数据
    processor = SimpleDataProcessor(args.config)
    processor.process_all_antennas(
        file_numbers=file_numbers,
        pack_start=args.pack_start,
        pack_end=args.pack_end,
        output_file=args.output
    )
    
    print("\n=== Processing complete! ===")


if __name__ == '__main__':
    main()

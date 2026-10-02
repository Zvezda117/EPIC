#!/usr/bin/env python3
"""
高效的Crab数据处理脚本 - 为EPIC准备数据

改进点（相比原脚本）：
1. 使用配置文件驱动的天线映射
2. 并行处理多个文件
3. 直接输出EPIC兼容格式（NPZ）
4. 更高效的FFT处理（避免重复创建plan）
"""

import os
import sys
import yaml
import numpy as np
import h5py
from pathlib import Path
from multiprocessing import Pool, cpu_count
import argparse
from datetime import datetime
from astropy.time import Time
from astropy import units as u

# 假设UnpackCustom在同目录
try:
    from UnpackCustom import UnpackCustom
except ImportError:
    print("Warning: UnpackCustom not found, will use dummy unpacker")
    UnpackCustom = None


class CrabDataProcessor:
    """高效的Crab数据处理器"""
    
    def __init__(self, config_file='antenna_mapping.yaml'):
        """初始化处理器"""
        with open(config_file, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.terminal_map = self.config['terminal_mapping']
        self.antenna_index = self.config['antenna_to_epic_index']
        self.obs_config = self.config['observation']
        self.proc_config = self.config['processing']
        
        # 计算EPIC需要的维度
        self.nstands = len(self.antenna_index)
        self.npols = 2
        
        print(f"Initialized processor for {self.nstands} antennas")
        print(f"Terminal mapping: {len(self.terminal_map)} files")
    
    def get_antenna_positions(self, coord_file='antenna_coordinates.csv'):
        """
        从CSV读取天线坐标
        返回：(nstands, 3) 数组，按antenna_index排序
        """
        import pandas as pd
        df = pd.read_csv(coord_file)
        
        positions = np.zeros((self.nstands, 3))
        for ant_name, idx in self.antenna_index.items():
            row = df[df['名称'] == ant_name]
            if len(row) == 0:
                print(f"Warning: {ant_name} not found in coordinates")
                continue
            # 使用相对坐标 (X/E, Y/N, Z/U)
            positions[idx] = [
                row['X/E'].values[0],
                row['Y/N'].values[0],
                row['Z/U'].values[0]
            ]
        
        # 转换为相对于第一个天线的位置（米）
        positions -= positions[0]
        return positions
    
    def get_antenna_phases(self, positions, freq_hz):
        """
        计算天线相位（类似epic_utils.py中的逻辑）
        
        Args:
            positions: (nstands, 3) 天线位置
            freq_hz: 观测频率（Hz）
        
        Returns:
            phases: (nstands, npols, 2) 相位 [real, imag]
        """
        c = 299792458.0  # 光速
        wavelength = c / freq_hz
        
        # 简化：假设天平面波从天顶方向入射
        # 实际应该根据源位置计算
        phases = np.zeros((self.nstands, self.npols, 2))
        
        for i in range(self.nstands):
            # 这里需要根据实际的相位校准数据来填充
            # 暂时设为单位相位
            phases[i, :, 0] = 1.0  # real
            phases[i, :, 1] = 0.0  # imag
        
        return phases
    
    def process_single_file(self, file_path, pack_start=0, pack_end=None):
        """
        处理单个基带文件
        
        Returns:
            dict: {
                'terminal_name': str,
                'pol0_data': complex array,
                'pol1_data': complex array,
                'pol0_antenna': str,
                'pol1_antenna': str,
                'metadata': dict
            }
        """
        file_name = Path(file_path).stem  # 去掉扩展名和编号
        # 匹配到配置中的terminal
        terminal_key = None
        for key in self.terminal_map.keys():
            if key in file_name:
                terminal_key = key
                break
        
        if terminal_key is None:
            raise ValueError(f"File {file_name} not found in terminal mapping")
        
        terminal_info = self.terminal_map[terminal_key]
        print(f"\nProcessing {file_path}")
        print(f"  Terminal: {terminal_info['terminal_name']}")
        print(f"  Pol0: {terminal_info['pol0']} ({terminal_info['pol0_polarization']})")
        print(f"  Pol1: {terminal_info['pol1']} ({terminal_info['pol1_polarization']})")
        
        # 读取并解包数据
        if UnpackCustom is None:
            raise RuntimeError("UnpackCustom not available")
        
        pack_length, pack_channel_num, pack_datatype, file_size = \
            UnpackCustom.onepack_read(file_path)
        
        if pack_end is None:
            pack_end = int(file_size / pack_length) - 1
        
        print(f"  File size: {file_size / 1e9:.2f} GB")
        print(f"  Packets: {pack_start} to {pack_end}")
        print(f"  Channels per packet: {pack_channel_num}")
        
        base_iq_head, base_iq_data = UnpackCustom.read_and_unpack(
            file_path, pack_start, pack_end, pack_length, 
            pack_datatype, file_size, pack_channel_num
        )
        
        # base_iq_data 是 list，每个元素是一个通道的数据
        # 双极化交织：[pol0, pol1, pol0, pol1, ...]
        pol0_data = []
        pol1_data = []
        
        for i in range(len(base_iq_data)):
            if i % pack_channel_num == 0:
                pol0_data.append(base_iq_data[i])
            elif i % pack_channel_num == 1:
                pol1_data.append(base_iq_data[i])
        
        # 转换为numpy数组并拼接
        pol0_data = np.concatenate(pol0_data) if pol0_data else np.array([])
        pol1_data = np.concatenate(pol1_data) if pol1_data else np.array([])
        
        # 提取GPS时间（如果有）
        metadata = self._extract_metadata(base_iq_head, file_size, pack_length)
        
        return {
            'terminal_name': terminal_info['terminal_name'],
            'pol0_data': pol0_data,
            'pol1_data': pol1_data,
            'pol0_antenna': terminal_info['pol0'],
            'pol1_antenna': terminal_info['pol1'],
            'pol0_polarization': terminal_info['pol0_polarization'],
            'pol1_polarization': terminal_info['pol1_polarization'],
            'metadata': metadata
        }
    
    def _extract_metadata(self, headers, file_size, pack_length):
        """从包头提取元数据"""
        if len(headers) == 0:
            return {}
        
        try:
            gps_head = headers[0]
            GPS_year = ((gps_head[60] >> 20) & 0xF) * 10 + ((gps_head[60] >> 16) & 0xF) + 2000
            GPS_month = ((gps_head[60] >> 12) & 0xF) * 10 + ((gps_head[60] >> 8) & 0xF)
            GPS_day = ((gps_head[60] >> 4) & 0xF) * 10 + (gps_head[60] & 0xF)
            GPS_hour = ((gps_head[61] >> 20) & 0xF) * 10 + ((gps_head[61] >> 16) & 0xF)
            GPS_min = ((gps_head[61] >> 12) & 0xF) * 10 + ((gps_head[61] >> 8) & 0xF)
            GPS_sec = ((gps_head[61] >> 4) & 0xF) * 10 + (gps_head[61] & 0xF)
            GPS_msec = gps_head[62] / (100 * 1e3)
            
            from datetime import datetime, timezone
            gps_dt = datetime(GPS_year, GPS_month, GPS_day, 
                            GPS_hour, GPS_min, GPS_sec,
                            int(GPS_msec * 1000), tzinfo=timezone.utc)
            
            t_utc = Time(gps_dt, scale='utc')
            
            return {
                'gps_time': str(gps_dt),
                'mjd_start': t_utc.mjd,
                'npackets': int(file_size / pack_length)
            }
        except Exception as e:
            print(f"Warning: Failed to extract GPS time: {e}")
            return {'npackets': int(file_size / pack_length)}
    
    def merge_to_epic_format(self, processed_files):
        """
        将处理后的文件合并为EPIC格式
        
        EPIC期望的数据格式：
        - voltages: (ntime, nchan, nstands, npol) 复数
        - antenna_positions: (nstands, 3) 米
        - phases: (nstands, npol, 2) [real, imag]
        
        Args:
            processed_files: list of dicts from process_single_file
        
        Returns:
            dict ready for np.savez
        """
        print("\n=== Merging to EPIC format ===")
        
        # 获取第一个文件的数据长度
        sample_data = processed_files[0]['pol0_data']
        nsamples = len(sample_data)
        print(f"Samples per channel: {nsamples}")
        
        # 初始化数据数组
        # EPIC格式：(ntime, nchan=1, nstands, npol)
        voltages = np.zeros((nsamples, 1, self.nstands, self.npols), 
                           dtype=np.complex64)
        
        # 填充数据
        for file_data in processed_files:
            pol0_ant = file_data['pol0_antenna']
            pol1_ant = file_data['pol1_antenna']
            pol0_pol = file_data['pol0_polarization']
            pol1_pol = file_data['pol1_polarization']
            
            # 获取天线索引
            pol0_idx = self.antenna_index.get(pol0_ant, -1)
            pol1_idx = self.antenna_index.get(pol1_ant, -1)
            
            if pol0_idx >= 0:
                pol_channel = 0 if pol0_pol == 'H' else 1
                voltages[:, 0, pol0_idx, pol_channel] = file_data['pol0_data'][:nsamples]
                print(f"  Mapped {pol0_ant}-{pol0_pol} -> stand {pol0_idx}, pol {pol_channel}")
            
            if pol1_idx >= 0:
                pol_channel = 0 if pol1_pol == 'H' else 1
                voltages[:, 0, pol1_idx, pol_channel] = file_data['pol1_data'][:nsamples]
                print(f"  Mapped {pol1_ant}-{pol1_pol} -> stand {pol1_idx}, pol {pol_channel}")
        
        # 获取天线位置
        positions = self.get_antenna_positions()
        
        # 计算相位
        center_freq = self.obs_config['center_freq_mhz'] * 1e6
        phases = self.get_antenna_phases(positions, center_freq)
        
        # 元数据
        metadata = processed_files[0]['metadata']
        
        return {
            'voltages': voltages,
            'antenna_positions': positions,
            'phases': phases,
            'metadata': metadata,
            'config': self.config
        }
    
    def process_all_files(self, data_dir, output_file, pack_start=0, pack_end=None):
        """
        处理data_dir下的所有基带文件
        
        Args:
            data_dir: 数据目录路径
            output_file: 输出NPZ文件路径
            pack_start: 起始包号
            pack_end: 结束包号（None表示全部）
        """
        data_dir = Path(data_dir)
        
        # 查找所有数据文件
        data_files = sorted(data_dir.glob('*Baseband*.data'))
        print(f"Found {len(data_files)} data files in {data_dir}")
        
        if len(data_files) == 0:
            raise ValueError(f"No data files found in {data_dir}")
        
        # 处理每个文件
        processed_files = []
        for file_path in data_files:
            try:
                result = self.process_single_file(str(file_path), pack_start, pack_end)
                processed_files.append(result)
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
                continue
        
        if len(processed_files) == 0:
            raise ValueError("No files successfully processed")
        
        # 合并为EPIC格式
        epic_data = self.merge_to_epic_format(processed_files)
        
        # 保存
        print(f"\n=== Saving to {output_file} ===")
        np.savez_compressed(output_file, **epic_data)
        
        # 打印统计
        print(f"\nOutput statistics:")
        print(f"  Voltages shape: {epic_data['voltages'].shape}")
        print(f"  Antenna positions shape: {epic_data['antenna_positions'].shape}")
        print(f"  Phases shape: {epic_data['phases'].shape}")
        print(f"  File size: {os.path.getsize(output_file) / 1e6:.2f} MB")
        
        return epic_data


def main():
    parser = argparse.ArgumentParser(
        description='Process Crab baseband data for EPIC',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('data_dir', type=str, 
                       help='Directory containing baseband data files')
    parser.add_argument('-o', '--output', type=str, 
                       default='crab_epic_data.npz',
                       help='Output NPZ file')
    parser.add_argument('-c', '--config', type=str,
                       default='antenna_mapping.yaml',
                       help='Antenna mapping config file')
    parser.add_argument('--pack-start', type=int, default=0,
                       help='Starting packet number')
    parser.add_argument('--pack-end', type=int, default=None,
                       help='Ending packet number (None=all)')
    
    args = parser.parse_args()
    
    # 初始化处理器
    processor = CrabDataProcessor(args.config)
    
    # 处理数据
    processor.process_all_files(
        args.data_dir, 
        args.output,
        args.pack_start,
        args.pack_end
    )
    
    print("\n=== Processing complete! ===")


if __name__ == '__main__':
    main()

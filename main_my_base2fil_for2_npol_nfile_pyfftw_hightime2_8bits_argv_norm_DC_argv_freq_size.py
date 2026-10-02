import gc
import os
import sys
import re
import glob
import pandas as pd
import numpy as np
from UnpackCustom import UnpackCustom 
import time
from datetime import datetime, timezone, timedelta
from astropy.time import Time
from astropy import units as u

import pyfftw

#your
from your import Your, Writer
#import numpy as np
#import tempfile
#from urllib.request import urlretrieve

import numpy as np
from your.formats.filwriter import make_sigproc_object

def extract_number_from_filename(filename):
    """从文件名中提取数字部分"""
    match = re.search(r'(\d+)\.data$', filename)
    if match:
        return int(match.group(1))
    else:
        raise ValueError(f"Invalid filename format: {filename}")

def base2fil(filepath, outpath='.', PackNumStart=0, PackNumEnd=8184, npol=2, DEBUG=False, NbitsType='float32', mjdstart=-1, center_freq=1300, bandwidth=500, size=-1):
    start_time = time.time()
    # 提取文件所在的目录
    directory = os.path.dirname(filepath)

    # 获取同目录下的所有 .data 文件
    patterns = [
        "?B*_33?.data",
        "?B*_34?.data",
        "?B*_83?.data",
        "?B*_99?.data",
        "?B*_100?.data",
        "?B*_157?.data",
        "?B*_158?.data",
        "?B*_172?.data",
        "?B*_20[56]?.data"
    ]    
    patterns = ["?Baseband*_321[1].data","?Baseband*_3211.data"  ]
    patterns = ["?Baseband*_*.data"]
    # 使用列表推导式合并匹配结果
    data_files = []
    for pattern in patterns:
        data_files.extend(glob.glob(os.path.join(directory, pattern)))

    print(data_files[0])
    sorted_data_files = sorted(data_files, key=extract_number_from_filename)

    for file in sorted_data_files:
        try:
            # ========== 新增：文件大小判断 ==========
            if size != -1:
                try:
                    file_size = os.path.getsize(file)
                except (OSError, FileNotFoundError):
                    print(f"无法获取文件大小，跳过: {file}")
                    continue
                if file_size < size:
                    print(f"文件太小 ({file_size} < {size})，跳过: {file}")
                    continue
            # =====================================
            
            # 1. 提取文件名
            filename = os.path.basename(file)

            # 2. 替换中文字符
            filename = filename.replace("频谱", "Spec").replace("基带", "Baseband")

            # 3. 提取并补齐序号为4位
            match = re.search(r'(\d+)\.data$', filename)
            if not match:
                raise ValueError("文件名中未找到有效的序号")
            seq_number = int(match.group(1))
            padded_seq_number = f"{seq_number:04d}"

            # 4. 去掉后缀 .data 并添加新的后缀 _p0.fil 或 _p1.fil
            name, ext = os.path.splitext(filename)

            outdatafile_p0 = os.path.join(outpath, f"{name[:-len(match.group(1)) - 1]}{padded_seq_number}_p0.fil")
            outdatafile_p1 = os.path.join(outpath, f"{name[:-len(match.group(1)) - 1]}{padded_seq_number}_p1.fil")

            #outdatafile_p0 = os.path.join( f"{name[:-len(match.group(1)) - 1]}{padded_seq_number}_p0.fil")
            #outdatafile_p1 = os.path.join( f"{name[:-len(match.group(1)) - 1]}{padded_seq_number}_p1.fil")

            try:
                # 使用 'x' 模式尝试原子性创建文件，如果文件已存在则抛出 FileExistsError
                with open(outdatafile_p0, 'x'):
                    pass  # 创建空文件，不写入内容

                # 只有成功创建文件的进程才会执行下面的逻辑
                print("input: ", file)
                print("output: ", outdatafile_p0)

                # 在此处添加你实际的文件处理逻辑（读取输入文件、写入输出文件等）

            except FileExistsError:
                # 文件已被其他进程创建，跳过处理
                print(f"Skipping file {file} because output files {outdatafile_p0} already exist.")
                continue

            # 读取单包信息
            #print("XX00")
            pack_length, pack_channel_num, pack_datatype, file_size = UnpackCustom.onepack_read(file)

            if pack_channel_num < npol:
               npol=pack_channel_num 

            PackNumStart=0
            PackNumEnd=int(file_size/pack_length)-1
            #PackNumEnd=100

            # 解包文件并处理
            base_iq_head, base_iq_data = UnpackCustom.read_and_unpack(
                file, PackNumStart, PackNumEnd, pack_length, pack_datatype, file_size, pack_channel_num
            )

            print("file_size/pack_length", file_size,pack_length)
            #print("XX1")
            if DEBUG:
                print(f"Processing file: {file}")
                print(f"Total packets processed: {len(base_iq_head)}")
                print(f"Total packets in file : {len(base_iq_data)}")
                print(f"Total packets in file each channel : {int(file_size/pack_length)}")
                print(f"Channel nums : {pack_channel_num}")
                print(f"Total data points in first channel: {len(base_iq_data[0])}")
                print("###### 数据读取完毕 ######")
 
            print("Starting FFT...")
            number_t=4
            # 初始化 FFTW 计划
            N = len(base_iq_data[0])  # 每个数据块的长度
            num_segments = 32         # 将数据分成32段
            segment_length = N // num_segments  # 每段的长度
            
            # 初始化 fftw 输入输出数组
            fft_input = pyfftw.empty_aligned(segment_length, dtype='complex128')
            fft_output = pyfftw.empty_aligned(segment_length, dtype='complex128')
            
            # 创建 FFT 计划
            fft_plan = pyfftw.FFTW(fft_input, fft_output, threads=4)  # 使用 4 个线程
            
            fft_results_power = []
            
            for i in range(0, (PackNumEnd + 1) * pack_channel_num, 1):
                # 获取当前数据块
                data_block = base_iq_data[i]
                
                # 将数据块分成32段
                segments = [data_block[j:j + segment_length] for j in range(0, N, segment_length)]
                
                # 存储每段的FFT功率
                segment_powers = []
                
                # 对每段执行 FFT 并计算功率谱
                for segment in segments:
                    # 如果段长度不足 segment_length，可以选择填充或截断
                    if len(segment) < segment_length:
                        segment = np.pad(segment, (0, segment_length - len(segment)), 'constant', constant_values=(0,))
                    
                    # 将数据复制到对齐的数组中
                    fft_input[:] = segment
                    
                    # 执行 FFT
                    fft_result = fft_plan()
                    
                    # 进行频谱移位
                    shifted_fft = np.fft.fftshift(fft_result)
                    
                    # 计算幅度和功率
                    abs_fft = np.abs(shifted_fft)
                    power_fft = abs_fft ** 2 / segment_length
                    
                    # 存储每段的功率
                    segment_powers.append(power_fft)
                
                # 将每8段的FFT功率相加
                for k in range(0, num_segments, 8):
                    combined_power = sum(segment_powers[k:k+8])
                    fft_results_power.append(combined_power)
            
            # 转换为 NumPy 数组
            fft_results_power_array = np.array(fft_results_power)
           
            print("XX fft_results_power_array ",np.shape(fft_results_power_array))
            print("FFT finished!")

            # 6. 计算 tshift

            tsamp = 64*1024*2*(500./bandwidth)*1e-3/number_t *1 # us (微秒) rigol 64k/pack,4Gsamplerate,ADCsample8, is 500MHz bandwidth,2ns/point. if ADCsample16, 4G/16=250MHz datarate/datapoints in 1s, so 4ns/point.
            n_samples = int(file_size/pack_length)*number_t
            tshift = tsamp * n_samples * (seq_number-0) / 1e6 / (60 * 60 * 24)  # 转换为天
            #print("mjdstart ",mjdstart)
            if mjdstart ==0:
                gps_ii=0
                GPS_year_temp = ((base_iq_head[gps_ii][60] >> 20) & 0b1111)*10+((base_iq_head[gps_ii][60]  >> 16) & 0b1111)+2000
                GPS_month_temp = ((base_iq_head[gps_ii][60]  >> 12) & 0b1111)*10+((base_iq_head[gps_ii][60]  >> 8) & 0b1111)
                GPS_day_temp = ((base_iq_head[gps_ii][60]  >> 4) & 0b1111)*10+(base_iq_head[gps_ii][60]  & 0b1111)
                GPS_hour_temp = ((base_iq_head[gps_ii][61]  >> 20) & 0b1111)*10+((base_iq_head[gps_ii][61] >> 16) & 0b1111)
                GPS_min_temp = ((base_iq_head[gps_ii][61] >> 12) & 0b1111)*10+((base_iq_head[gps_ii][61] >> 8) & 0b1111)
                GPS_sec_temp = ((base_iq_head[gps_ii][61] >> 4) & 0b1111)*10+(base_iq_head[gps_ii][61] & 0b1111)
                GPS_msec_temp =base_iq_head[gps_ii][62]/(100*1e3);

                gps_dt_naive = datetime(
                    year=GPS_year_temp,
                    month=GPS_month_temp,
                    day=GPS_day_temp,
                    hour=GPS_hour_temp,
                    minute=GPS_min_temp,
                    second=GPS_sec_temp,
                    microsecond=int(GPS_msec_temp*1000) #millisec to micro
                )

                gps_dt = gps_dt_naive.replace(tzinfo=timezone.utc)
                t_utc = Time(gps_dt, scale='utc')
                tstart = t_utc.mjd
                print("GPS Time (as calendar):", gps_dt)
                print("start MJD from gps = ", tstart)

            elif mjdstart < 0:
               stat_info = os.stat(file)
               # 定义一个函数，用于将纳秒时间戳转换为精确到纳秒的时间字符串
               def format_timestamp_ns(ns_timestamp):
                   seconds = ns_timestamp // 1_000_000_000  # 秒部分
                   nanoseconds = ns_timestamp % 1_000_000_000  # 纳秒部分
                   dt = datetime.fromtimestamp(seconds)  # 转换为 datetime 对象
                   return dt.strftime('%Y-%m-%d %H:%M:%S') + f".{nanoseconds:09d}"  # 拼接纳秒部分
               # 修改时间 (mtime)
               mtime_ns = stat_info.st_mtime_ns
               print("修改时间 (mtime):", format_timestamp_ns(mtime_ns))
               mtime_utc = Time(format_timestamp_ns(mtime_ns), scale='utc') -8*u.hour
               tstart = mtime_utc.mjd - tsamp * n_samples * (1-0) / 1e6 / (60 * 60 * 24)
               print("mtime_utc ", mtime_utc)
               print("start MJD from mtime = ", tstart)

            else:
               tstart = mjdstart +tshift

            #print("XX ",file_size, pack_length,tsamp,n_samples,seq_number)
            nchans=int(N // num_segments)
            foff= -bandwidth/segment_length #-0.244140625, 500MHz bandwidth, segment_length=FFT points
            nbits=getattr(np, NbitsType)().itemsize * 8

            # 创建 SIGPROC 对象并写入文件头
            print("Writing fil header...")
            for channel in range(pack_channel_num):
                print("XX outpath ",outpath)
                outdatafile_p0=os.path.join(outpath, f"{name[:-len(match.group(1)) - 1]}{padded_seq_number}_p{channel}.fil")
                print("XX outdatafile_p0 ",outdatafile_p0)
                sigproc_object_p0=None
                sigproc_object_p0 = make_sigproc_object(
                       rawdatafile=os.path.basename(outdatafile_p0),
                       source_name="B0359+54",
                       nchans=nchans,
                       foff=foff,  # MHz
                       fch1=center_freq+bandwidth/2.,  # MHz
                       tsamp=tsamp*1e-6,  # seconds
                       tstart=tstart,  # MJD
                       src_raj=33259.00,  # HHMMSS.SS
                       src_dej=543443.33,  # DDMMSS.SS
                       machine_id=6,
                       nbeams=0,
                       ibeam=0,
                       nbits=nbits,
                       nifs=1,
                       barycentric=0,
                       pulsarcentric=0,
                       telescope_id=4,
                       data_type=1,
                       az_start=-1,
                       za_start=-1,
                )

                sigproc_object_p0.write_header(outdatafile_p0)

                # Normalize to [0, 255]
                print("Normalizing fft_results_power_array...")

                # 假设 fft_results_power_array 的维度为 (65408, 2048)
                idx = (np.arange(PackNumEnd + 1) * pack_channel_num * number_t + channel * number_t)[:, None] + np.arange(number_t)
                fft_results_power_array_temp=fft_results_power_array[idx.flatten()]
                print("fft_results_power_array_temp ",np.shape(fft_results_power_array_temp))
                N = len(fft_results_power_array_temp)
                M = fft_results_power_array_temp.shape[1]  # 第二维长度

                # 替换中间位置的值为附近平均值
                for i in range(N):
                    # 中间位置
                    mid = M // 2  # 2048 // 2 = 1024

                    # 提取附近范围的值（不包括中间值）
                    left_values = fft_results_power_array_temp[i, mid - 40:mid]     # [1014, 1023]
                    right_values = fft_results_power_array_temp[i, mid + 1:mid + 41]  # [1025, 1034]
                    nearby_values = np.concatenate((left_values, right_values))  # 拼接左右两侧的值

                    # 计算附近范围的平均值
                    average_value = np.mean(nearby_values)
                    
                    # 替换中间位置的值
                    fft_results_power_array_temp[i, mid] = average_value

                # Log 归一化到 [0, 255]
                # Step 1: 对数变换 (避免 log(0) 错误，确保所有值 > 0)
                fft_results_power_array_temp = np.clip(fft_results_power_array_temp, a_min=1e-10, a_max=None)  # 避免 log(0)
                data_log = np.log10(fft_results_power_array_temp)

                # Step 2: 归一化到 [0, 1] 范围
                log_min = np.min(data_log)
                log_max = np.max(data_log)
                data_normalized = np.clip((data_log - log_min) / (log_max - log_min), 0.01, 0.99)

                # Step 3: 量化到 [0, 255] 范围，并转换为 uint8 类型
                fft_results_power_array_uint8 = np.round(data_normalized * 255).astype(np.uint8)

                # 打印调试信息
                # print("XXX ", np.shape(fft_results_power_array), data_log[2000], log_min, log_max, data_normalized[2000], fft_results_power_array_uint8[2000])

                # 写入频谱数据
                print(f"Writing to channel p{channel} fil finished!")
                #print("XX ",np.shape(fft_results_power_array),PackNumEnd,channel,(PackNumEnd+1)*pack_channel_num,pack_channel_num)
                #sigproc_object_p0.append_spectra(fft_results_power_array[np.arange(channel,(PackNumEnd+1)*number_t*pack_channel_num,pack_channel_num)].astype(getattr(np, NbitsType)), outdatafile_p0)
                sigproc_object_p0.append_spectra(fft_results_power_array_uint8, outdatafile_p0)

                del sigproc_object_p0,fft_results_power_array_uint8
                gc.collect()

            del base_iq_head, base_iq_data, fft_results_power, fft_results_power_array
            gc.collect()
            print("Writing to fil finished!")
            print(f"Processed file: {file}")
            print(f"Output files: {outdatafile_p0}, {outdatafile_p1 if npol == 2 else ''}")

        except Exception as e:
            print(f"Error processing file {file}: {e}")
        print(f"Time taken: {time.time() - start_time:.4f} seconds")

import argparse
# 示例调用
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Monitor and manage recvfil program.')
    parser.add_argument('--filepath', type=str, required=True, help='rigol raw data directory.')
    parser.add_argument('--outpath', type=str, default='.', help='rigol raw data directory.')
    parser.add_argument('--datatype', type=str, default='uint8', help='which data type in fil.')
    parser.add_argument('--mjd', type=np.float64, default=-1, help='the 1st package in this run, run 0s start time')

    parser.add_argument('--center_freq', type=np.int32, default='1300', help='which center freq in fil.')
    parser.add_argument('--bandwidth', type=np.int32, default='500', help='which bandwidth in fil.')

    parser.add_argument('--size', type=np.uint32, default=-1, help='file size wheter can process')

    args = parser.parse_args()

    filepath = args.filepath
    outpath = args.outpath
    NbitsType = args.datatype
    mjd =  args.mjd #if args.mjd else '-1'
    center_freq = args.center_freq
    bandwidth = args.bandwidth
    size = args.size

    base2fil(
        filepath = filepath,
        outpath = outpath,
        #PackNumStart=0,
        #PackNumEnd=16352-1,
        #npol=1,
        DEBUG=True,
        NbitsType='uint8',
        #NbitsType='uint16'  #'float32'
        #NbitsType= NbitsType,
        mjdstart = mjd, #np.float64(sys.argv[3]),
        center_freq=center_freq,
        bandwidth=bandwidth,
        size=size,
    )


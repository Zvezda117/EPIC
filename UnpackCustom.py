#import tkinter as tk
#from tkinter import filedialog
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import time
from tqdm import tqdm
import pandas as pd

DEBUG = True

from unpack import UnPackage

class UnpackCustom:
    @staticmethod
    def select_folder():
        """选择文件路径"""
        root = tk.Tk()
        root.withdraw()
        filepath = filedialog.askopenfilename()
        if filepath:
            print(f"当前读取的数据路径={filepath}")
        return filepath

    @staticmethod
    def onepack_read(file_path,DEBUG = True):
        """读取文件的单包信息"""
        with open(file_path, "rb") as f:
            onepack_data = np.frombuffer(f.read(32), dtype='uint32')
        pack_length = onepack_data[2]
        channel_num = np.mod(onepack_data[4], 2**16)
        pack_datatype = np.mod(onepack_data[6], 2**16)
        file_size = os.path.getsize(file_path)

        common_data_type_relate = {
            4: 'uint16',
            5: 'int16',
            2: 'uint8',
            3: 'int8',
            0: 'uint32',
            1: 'int32',
            6: 'QI拼接,分别为int16',
            13: 'QI拼接,分别为int8',
            7: '单精度浮点型,32bit'
        }
        if DEBUG:
            print(f"当前解包数据单包数据长度={pack_length}")
            print(f"当前解包数据单包包含通道数={channel_num}")
            print(f"当前解包数据类型={common_data_type_relate[pack_datatype]}")
            print(f"当前文件大小={file_size}")
            print(f"Total packs in file: {file_size / pack_length}")

        return pack_length, channel_num, pack_datatype, file_size

    @staticmethod
    def read_and_unpack(file_path, pack_num_start, pack_num_end, pack_length, pack_datatype, file_size,channel_num):
        """解包并处理文件数据"""
        print("pack_num_start: ",pack_num_start," pack_num_end: ",pack_num_end)
        pack_byte_start = pack_num_start * pack_length
        pack_byte_end = (pack_num_end - pack_num_start + 1) * pack_length
        pack_byte_read = np.longdouble(pack_byte_end) - np.longdouble(pack_byte_start)

        if pack_byte_start > file_size or pack_byte_read > file_size:
            raise ValueError("###### 读取包数设置错误，请检查 ######")

        base_iq_head = []
        base_iq_data = []

        with open(file_path, "rb") as f:
            for num in tqdm(range(pack_num_start, pack_num_end + 1)):
                f.seek(int(pack_length * num), 0)
                data = np.frombuffer(f.read(int(pack_length)), dtype='uint32')
                unpack_data = UnPackage.common_solve_data(data, step=0, qv_length=pack_length, head_tag=0x18EFDC01)

                if isinstance(unpack_data, tuple) and len(unpack_data) == 2:
                    unpack_head, (heads, _data) = unpack_data
                    base_iq_head.append(heads)
                    if pack_datatype == 13:
                        for i in range(min(len(_data), channel_num)):
                            data_a = _data[i][0]
                            base_iq_data.append(UnpackCustom.process_data_13(data_a))
                    elif pack_datatype == 7:
                        for i in range(min(len(_data), channel_num)):
                            data_a = _data[i]
                            base_iq_data.append(data_a)
                    elif pack_datatype == 6:
                           base_iq_data.append(UnpackCustom.process_data_6(_data[0]))
                    else:
                        for i in range(min(len(_data), channel_num)):
                            base_iq_data.append(_data[i])

                    # 如果_data的实际长度小于channel_num，打印警告信息
                    if len(_data) < channel_num:
                        print(f"Warning: Expected {channel_num} channels but found {len(_data)}. Check the data.")

        return base_iq_head, base_iq_data

    @staticmethod
    def process_data_old(data, pack_datatype): #list slow
        data = np.array(data)
        if pack_datatype == 13:
            channel_data = data.reshape(-1, 2)
            complex_data = channel_data[:, 0] + 1j * channel_data[:, 1]
            return complex_data.tolist()
        else:
            return data.tolist()

    @staticmethod
    def process_data_13(data):
        data = np.array(data, dtype=np.int32)  # Ensure the correct data type
        # Reshape the data to pairs of (real, imaginary) values
        if len(data) % 2 != 0:
           raise ValueError("Data length must be even for complex data")
        complex_data = data.astype(np.float32).view(np.complex64)  # Interpret pairs as complex numbers
        #complex_data = data[::2] + 1j * data[1::2] #slow

        return complex_data

    @staticmethod
    def process_data_6(data):
        data = np.array(data, dtype=np.int32)  # Ensure the correct data type
        # Reshape the data to pairs of (real, imaginary) values
        if len(data) % 2 != 0:
           raise ValueError("Data length must be even for complex data")
        complex_data = data.T.copy().astype(np.float32).view(np.complex64)  # Interpret pairs as complex numbers
        #complex_data = data[::2] + 1j * data[1::2] #slow

        return complex_data[:,0]

    @staticmethod
    def process_data(data, pack_datatype):
        data = np.array(data, dtype=np.int32)  # Ensure the correct data type
        if pack_datatype == 13:
            # Reshape the data to pairs of (real, imaginary) values
            if len(data) % 2 != 0:
                raise ValueError("Data length must be even for complex data")
            complex_data = data.astype(np.float32).view(np.complex64)  # Interpret pairs as complex numbers
            #complex_data = data[::2] + 1j * data[1::2] #slow
            return complex_data
        elif pack_datatype == 6:
            # Reshape the data to pairs of (real, imaginary) values
            if len(data) % 2 != 0:
                raise ValueError("Data length must be even for complex data")
            complex_data = data.T.copy().astype(np.float32).view(np.complex64)  # Interpret pairs as complex numbers
            #complex_data = data[0] + 1j * data[1] #slow
            return complex_data[:,0]

        else:
            return data

    @staticmethod
    def visualize_data(base_iq_data, sample_rate=500, freq_range=250):
        """可视化数据"""
        for i in range(min(1, len(base_iq_data))):  # 可视化第一个通道
            fdata = np.abs(np.fft.fftshift(np.fft.fft(base_iq_data[i])))
            xfp = 20 * np.log10(np.clip(fdata, 0, 1e100)) + 10
            qdata = xfp - np.max(xfp)
            x = np.around(
                np.arange(1400 - freq_range, 1400 + freq_range, sample_rate / len(qdata)),
                decimals=2,
            ).tolist()
            plt.figure()
            plt.plot(x, np.flipud(qdata))
        plt.show()

# 主程序入口
def main(DEBUG = True):
    start_time = time.time()

    # 初始化参数
    PackNumStart = 0
    PackNumEnd = 500
    filepath = "../H频谱0I基带0IDBF_354.data"

    # 读取单包信息
    pack_length, pack_channel_num, pack_datatype, file_size = UnpackCustom.onepack_read(filepath)

    # 解包文件并处理
    base_iq_head, base_iq_data = UnpackCustom.read_and_unpack(
        filepath, PackNumStart, PackNumEnd, pack_length, pack_datatype, file_size, pack_channel_num
    )
    if DEBUG:
        print(f"Total packets processed: {len(base_iq_head)}")
        print(f"Total data points: {len(base_iq_data)}")
        print(f"Total data points in first channel: {len(base_iq_data[0])}")
        print("###### 数据读取完毕 ######")

    print(f"Time taken: {time.time() - start_time:.4f} seconds")
    UnpackCustom.visualize_data(base_iq_data)

    fs = 500e6
    f = np.arange(-fs/2, fs/2, fs/len(SpecData[0])) / 1e6
    # 绘图
    print(f"频谱数据绘图")
    plt.figure()
    plt.plot(1400 - f, SpecData_temp[0])
    plt.show()

if __name__ == "__main__":
    main(DEBUG = True)


# 9197数据转换详细步骤

## 原始数据信息

### 来源
```
目录: /mnt/DATA/存储阵列/9197_Base2_ADC16_26-01-18_19-16-51/
天线: 北3, 北4, 北5, 北6 (4天线, 单极化)
文件: H22_*.data, V00_*.data, V11_*.data, V22_*.data (137个文件/天线)
```

### 原始数据格式
```
格式: int8 IQ交织
采样率: 250 MHz
单文件长度: 1,071,644,672 字节 = 535,822,336 复数采样点
时长: 535,822,336 / 250e6 = 2.143秒/文件
总时长: 137 × 2.143 = 293.6秒 ≈ 4.9分钟
```

---

## 转换步骤详解

### 步骤1：解包原始基带数据 (时域)

**脚本**: `scripts/process_9197_single_channel.py`

```python
# 1.1 读取原始.data文件
raw_bytes = np.fromfile('H22_*.data', dtype=np.int8)
# Shape: (1071644672,)

# 1.2 IQ解交织
I = raw_bytes[0::2].astype(np.float32)  # 偶数索引
Q = raw_bytes[1::2].astype(np.float32)  # 奇数索引
# 每个Shape: (535822336,)

# 1.3 组合为复数
voltages_ant0 = I + 1j*Q
# Shape: (535822336,)

# 1.4 重复4个天线
voltages = np.zeros((535822336, 1, 4, 1), dtype=np.complex64)
voltages[:, 0, 0, 0] = voltages_ant0  # 北3
voltages[:, 0, 1, 0] = voltages_ant1  # 北4
voltages[:, 0, 2, 0] = voltages_ant2  # 北5
voltages[:, 0, 3, 0] = voltages_ant3  # 北6

# 1.5 保存时域数据
np.savez_compressed('9197_file0_complete.npz', voltages=voltages)
```

**输出**:
- 文件: `data/processed/9197_file0_complete.npz`
- Shape: `(535822336, 1, 4, 1)` = (采样点, 通道, 天线, 极化)
- Dtype: `complex64`
- 大小: ~4 GB (压缩后)
- **这是纯时域数据，每个采样点是250MHz采样的IQ电压值**

---

### 步骤2：时域→频域转换 (FFT)

**当前方法** (在首次成图脚本中):

```python
# 2.1 定义参数
ntime_steps = 1000        # 输出的时间步数
samples_per_step = 25000  # 每个时间步的采样点数
nchan = 128               # 输出频率通道数

# 2.2 时间分箱
# 每个时间步对应的时长
time_per_step = 25000 / 250e6 = 0.0001秒 = 0.1毫秒

# 总积分时间
total_time = 1000 × 0.0001 = 0.1秒

# 2.3 提取子集
start_sample = 0
end_sample = 1000 * 25000 = 25,000,000
voltages_subset = voltages[:25000000, 0, :, 0]
# Shape: (25000000, 4)

# 2.4 重塑为时间步
voltages_reshaped = voltages_subset.reshape(1000, 25000, 4)
# Shape: (时间步, 每步采样点, 天线)

# 2.5 对每个时间步进行FFT
fft_result = np.zeros((1000, 128, 4), dtype=np.complex64)

for itime in range(1000):
    for iant in range(4):
        # 取出一个时间步的时间序列
        timeseries = voltages_reshaped[itime, :, iant]
        # Shape: (25000,)
        
        # FFT转换到频域
        fft_full = np.fft.fft(timeseries)
        # Shape: (25000,) -> 频率分辨率 = 250MHz/25000 = 10 kHz
        
        # 取前128个频率通道
        fft_result[itime, :, iant] = fft_full[:128]
        # 对应频率范围: 0 - 1.28 MHz

# 2.6 输出
# Shape: (1000, 128, 4) = (时间步, 频道, 天线)
# 每个时间步: 0.1ms积分
# 每个频道: 10 kHz带宽
# 总带宽: 128 × 10kHz = 1.28 MHz
```

**关键问题**:
- ❌ **只用了1.28 MHz带宽，浪费了248.72 MHz！**
- ❌ **每个时间步只积分0.1ms，太短！**
- ❌ **总积分只有0.1秒**

---

### 步骤3：填充到256天线

```python
# 3.1 创建EPIC格式数组
epic_data = np.zeros((1000, 128, 256, 2), dtype=np.complex64)
#                     时间  频道  天线  极化

# 3.2 填充前4个天线
for iant in range(4):
    epic_data[:, :, iant, 0] = fft_result[:, :, iant]  # H极化
    epic_data[:, :, iant, 1] = fft_result[:, :, iant]  # V极化(重复)

# 3.3 其余252个天线保持为0
# epic_data[:, :, 4:256, :] = 0 (已经是0)
```

**说明**:
- ✅ 确实是256个站点，只填入4个
- ⚠️  H和V极化填充了相同数据（因为原始数据是单极化）
- 后252个站点全为0，不参与成图

---

### 步骤4：创建meta数据

```python
meta = np.array([ntime_steps, 0], dtype=np.int64)
# meta[0] = 1000 (时间步数)
# meta[1] = 0 (保留)
```

---

### 步骤5：保存为EPIC输入格式

```python
np.savez_compressed(
    'build/data/40ms_128chan_600offset_gulp_c64_virtransit.npz',
    data=epic_data,
    meta=meta
)
```

**输出**:
- 文件: `build/data/40ms_128chan_600offset_gulp_c64_virtransit.npz`
- data shape: `(1000, 128, 256, 2)`
- data dtype: `complex64`
- meta: `[1000, 0]`
- 大小: ~5.7 MB

---

### 步骤6：EPIC处理

```bash
./epic++ --offline \
    --out_dir ../results/images \
    --imagesize 128 \
    --channels 128 \
    --nts 1000 \
    --seq_accum 40 \
    --imageres 1.056 \
    ...
```

**EPIC内部处理**:
1. 从NPZ加载数据
2. 计算visibility (相关性):
   ```
   V_ij(ν,t) = <E_i(ν,t) × E_j*(ν,t)>
   ```
   - 只有前4个天线有数据，生成 C(4,2) = 6 条基线
3. UV gridding: 将基线数据映射到UV平面
4. 逆FFT: UV平面 → 图像平面
5. 保存FITS

---

## 关键问题回答

### Q1: 是否使用了真实的9197数据？

**答**: ✅ **是的**，但只用了很小一部分：
- 可用: 535,822,336 采样点 = 2.14秒
- 实际使用: 25,000,000 采样点 = 0.1秒
- **利用率: 4.7%**

### Q2: 256个站点但只填入4个？

**答**: ✅ **正确**
```
epic_data.shape = (1000, 128, 256, 2)
                          ^^^
                          256个天线槽位

使用情况:
- 天线0-3: 有数据 (北3/4/5/6)
- 天线4-255: 全0 (不参与成像)
```

### Q3: 积分时间和信噪比

**当前配置**:
```
每个时间步积分时间: 0.1 ms
时间步数: 1000
总积分时间: 0.1 秒
```

**改进空间**:
```
可用数据: 2.14 秒
如果每个时间步积分 10 ms (100倍):
  - samples_per_step = 2,500,000
  - ntime_steps = 214
  - 总积分时间 = 2.14秒
  - 信噪比提升: √(100 × 214) ≈ 146倍！
```

### Q4: 时延校准

**当前状态**: ❌ **完全没有做时延校准！**

需要的校准:
1. **几何时延**: 不同天线到源的光程差
2. **电缆时延**: 硬件路径差异
3. **钟差**: ADC采样时钟偏差

**后果**:
- 各天线信号相位不对齐
- 干涉图样破坏
- 成像质量极差

**EPIC是否自动校准？**
- ⚠️  需要检查EPIC源码
- 可能需要提供天线坐标和源位置
- 或者在输入数据中预先校准

---

## 带宽利用率

**当前**:
```
采样带宽: 250 MHz
使用带宽: 128 × 10kHz = 1.28 MHz
利用率: 0.51%
```

**为什么这么低？**
- FFT长度太短 (25000点)
- 只取了前128个频率bin
- 硬编码限制channels=128

**改进方案**:
```
方案1: 更大的FFT
  - FFT长度: 250,000 点
  - 频率分辨率: 1 kHz
  - 输出通道: 125,000 (需要修改EPIC源码)
  
方案2: 保持FFT=25000，输出更多通道
  - 取前12500个频率bin
  - 带宽: 125 MHz
  - 利用率: 50%
```

---

## 总结

| 参数 | 当前值 | 可用值 | 利用率 |
|------|--------|--------|--------|
| 时长 | 0.1秒 | 2.14秒 | 4.7% |
| 带宽 | 1.28 MHz | 250 MHz | 0.51% |
| 天线 | 4 | 4 | 100% |
| 时延校准 | ❌ | - | - |

**数据是真的，但处理非常粗糙！**

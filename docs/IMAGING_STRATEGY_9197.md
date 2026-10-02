# 9197数据成图策略

## 数据现状分析

### 原始数据
```
文件：4个天线 × 137个文件
每文件：16,352包 × 65,536采样点 = 1,071,644,672个复数点
采样率：250 MHz（复数）
时间长度：4.29秒/文件
总时长：137 × 4.29秒 = 588秒 ≈ 9.8分钟
```

### 处理后的数据格式
```python
voltages shape: (1071644672, 1, 4, 1)
                 nsamples   nchan  nstands  npol

# 这是时域基带数据，不是频域数据！
```

---

## 问题：为什么不能直接给EPIC？

### EPIC期望的格式
```python
EPIC期望: (ntime, nchan, nstands, npol)
示例:     (1000, 128, 256, 2)

关键：
- ntime: 时间积分步数（如1000个40ms积分）
- nchan: 频率通道数（已经FFT过的）
- nstands: 天线数（256，硬编码！）
- npol: 极化数（2）
```

### 我们的数据
```python
我们有: (1071644672, 1, 4, 1)

问题：
❌ 只有1个频率通道（基带数据，需要FFT）
❌ 只有4个天线（EPIC硬编码期望256个）
❌ 只有1个极化（EPIC期望2个）
❌ 时间维度是原始采样点，不是积分步数
```

---

## 解决方案：两条路径

### **路径A：修改EPIC支持4天线（推荐，但需要2小时）**

这是**正确的做法**，但需要修改C++代码：

#### 需要修改的文件
1. `src/ex/constants.h`
```cpp
// 添加新的站点定义
enum CONSTANTS {
  // 原有的...
  LBAND_NSTANDS = 4,  // 新增
  LBAND_NPOLS = 1,
};

// 或者用条件编译
#ifdef USE_LBAND_4ANT
  #define CURRENT_NSTANDS 4
#else
  #define CURRENT_NSTANDS 256
#endif
```

2. `src/raft_kernels/dummy_packet_gen.hpp`
```cpp
// line 102: 修改gulp_size计算
size_t gulp_size = m_nchan_in * LWA_SV_NPOLS * nseqs * LWA_SV_NSTANDS * 1;
// 改为
size_t gulp_size = m_nchan_in * npols * nseqs * nstands * 1;
```

3. GPU kernels (重要！)
```cpp
// src/ex/fft_dx.cuh:143
auto VoltageGridderF = DualPolEpicFullGridder<FFT, support, LWA_SV_NSTANDS>;
// 改为
auto VoltageGridderF = DualPolEpicFullGridder<FFT, support, CURRENT_NSTANDS>;
```

**时间估计**：2-4小时（如果我给你生成patch）

---

### **路径B：数据填充（快速但丑陋，5分钟）**

这是Linus会骂的hack，但**能立即work**：

```python
# convert_for_epic.py
import numpy as np

def pad_to_epic_format(input_file, output_file):
    """
    把4天线1极化数据填充成256天线2极化
    """
    print("加载数据...")
    data = np.load(input_file, allow_pickle=True)
    voltages = data['voltages']  # (N, 1, 4, 1)
    
    nsamples = voltages.shape[0]
    
    # 步骤1：做FFT转换到频域
    print("FFT转换...")
    # 决定频率通道数和时间积分
    nchan = 128  # 频率通道数
    ntime_per_chunk = 25000  # 每个时间步多少采样点
    
    # 计算能产生多少个时间步
    ntime_chunks = nsamples // ntime_per_chunk
    print(f"  将产生 {ntime_chunks} 个时间步")
    
    # 重塑数据用于FFT
    # 截断到整数个chunk
    useful_samples = ntime_chunks * ntime_per_chunk
    voltages_reshaped = voltages[:useful_samples].reshape(
        ntime_chunks, ntime_per_chunk, 1, 4, 1
    )
    
    # 对每个时间chunk做FFT
    fft_result = np.zeros((ntime_chunks, nchan, 4, 1), dtype=np.complex64)
    
    for i in range(ntime_chunks):
        for ant in range(4):
            # 对时间序列做FFT
            timeseries = voltages_reshaped[i, :, 0, ant, 0]
            fft = np.fft.fft(timeseries)
            
            # 取前nchan个频率bin
            fft_result[i, :, ant, 0] = fft[:nchan]
        
        if i % 100 == 0:
            print(f"  处理进度: {i}/{ntime_chunks}")
    
    print(f"FFT完成！形状: {fft_result.shape}")
    
    # 步骤2：填充到256天线，2极化
    print("填充到EPIC格式...")
    epic_data = np.zeros((ntime_chunks, nchan, 256, 2), dtype=np.complex64)
    
    # 把4个天线放到前4个位置，两个极化都用同样的数据
    for ant in range(4):
        epic_data[:, :, ant, 0] = fft_result[:, :, ant, 0]  # H极化
        epic_data[:, :, ant, 1] = fft_result[:, :, ant, 0]  # V极化（复制）
    
    # 其余252个天线保持为0
    
    print(f"最终形状: {epic_data.shape}")
    print(f"  期望: (ntime, nchan, 256, 2)")
    print(f"  实际: ({ntime_chunks}, {nchan}, 256, 2)")
    
    # 保存
    print(f"保存到 {output_file}...")
    np.savez_compressed(
        output_file,
        data=epic_data,
        meta=np.array([ntime_chunks, 0])  # 简化的元数据
    )
    
    file_size_mb = os.path.getsize(output_file) / 1e6
    print(f"✓ 完成！文件大小: {file_size_mb:.2f} MB")
    
    return epic_data

if __name__ == '__main__':
    import sys
    import os
    
    input_file = sys.argv[1] if len(sys.argv) > 1 else '9197_file0_complete.npz'
    output_file = sys.argv[2] if len(sys.argv) > 2 else '9197_epic_padded.npz'
    
    pad_to_epic_format(input_file, output_file)
```

**使用方法**：
```bash
# 等处理完成后
python convert_for_epic.py 9197_file0_complete.npz 9197_epic_ready.npz

# 然后直接运行EPIC
cd build
./epic++ --offline \
    --npytbnfile ../9197_epic_ready.npz \
    --out_dir ../9197_images \
    --imagesize 64 \
    --channels 64 \
    --nts 1000 \
    --seq_accum 40
```

---

## 关键参数调整（针对4天线）

### 成像参数
```bash
--imagesize 32      # 从128降到32（天线太少）
--imageres 10.0     # 从1度增到10度（分辨率很差）
--channels 64       # 从128降到64
--support 2         # 从5降到2（UV覆盖稀疏）
--aeff 10           # L-band小天线估计值
```

### 为什么这些值？

**imagesize = 32**
- 4个天线只能形成6条基线
- 最大基线长度假设50m，L-band波长0.22m
- 最大uv距离 ≈ 50/0.22 = 227λ
- 图像大小应该 > 2×227 ≈ 454像素，但实际UV覆盖稀疏
- 32像素已经足够（过大会有很多空白）

**imageres = 10度**
- 4天线阵列的分辨率：λ/D ≈ 0.22m / 50m ≈ 0.25° 
- 但UV覆盖不完整，实际分辨率会差很多
- 10度像素可以覆盖大部分天空

**channels = 64**
- 原数据250MHz带宽
- 64通道 → 每通道3.9MHz
- 合理的频谱分辨率

---

## 预期结果

### 能看到什么？
✓ 非常明亮的点源（如果9197是强源）
✓ 大致的天空位置

### 看不到什么？
❌ 细节结构
❌ 弱源
❌ 准确的形态

### 图像质量估计
```
动态范围：~10:1（vs LWA的100:1）
分辨率：~10度（vs LWA的1度）
灵敏度：约低20dB（天线数差64倍）
```

---

## 我的建议（Linus式）

### 如果你的目标是"快速验证流程"
→ **用路径B**（填充法），今天就能看到图

### 如果你的目标是"建立生产系统"
→ **用路径A**（修改EPIC），虽然慢但正确

### 如果你的目标是"科学观测"
→ **先增加天线！** 4个天线对于成像来说**太少了**
   - 最少需要16个天线才能有decent的图
   - 或者用这4个天线做波束成形而不是成像

---

## 下一步行动

等处理完成后（还需要约2分钟），我会：

1. ✅ 创建转换脚本（convert_for_epic.py）
2. ✅ 运行转换
3. ✅ 生成第一张测试图
4. 📊 分析图像质量
5. 🔧 如果需要，修改EPIC源码

**你决定用哪条路径？**

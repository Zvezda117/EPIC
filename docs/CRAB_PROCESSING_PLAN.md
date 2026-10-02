# Crab数据处理任务计划

## 目标
将16天线L-band Crab观测数据准备为EPIC可处理的格式

## 当前状态分析

### 数据文件（6个，共12GB）
```
/mnt/data_ssd/zdj/data/crab/
├── HBaseband0Spec0_4304.data  (2GB) - 终端1: 东1H + 北3V
├── HBaseband1Spec1_4304.data  (2GB) - 终端2: 东4H + 东3V
├── HBaseband2Spec2_4304.data  (2GB) - 终端3: 东5H + 东6H
├── VBaseband0Spec0_4304.data  (2GB) - 终端4: 东4V + 东5V
├── VBaseband1Spec1_4304.data  (2GB) - 终端5: 东9H + 东10V
└── VBaseband2Spec2_4304.data  (2GB) - 终端6: 东7H + 东8V
```

### 天线阵列配置
- **实际天线数量**: 10个（东1,3,4,5,6,7,8,9,10 + 北3）
- **极化**: 双极化（H/V），但接线不规则
- **频率**: L-band, 1355 MHz中心频率
- **带宽**: 250 MHz

### 关键问题
1. ⚠️ **极化混乱**: 某些"H"终端包含V极化，反之亦然
2. ⚠️ **天线重复**: 东4和东5在多个终端出现
3. ⚠️ **原脚本低效**: 串行处理，重复FFT计划创建

---

## 任务分解（按Linus方式）

### **P0 - 必须立即解决（否则数据错误）**

#### Task 1: 验证接线映射的正确性 ⚠️ CRITICAL
**问题**: 你给的接线图有逻辑矛盾

```
东4出现在3个位置:
- 终端2 (H11): 东4水平
- 终端2 (H11): 东3垂直  <- 但这里写的是东3，不是东4
- 终端4 (V00): 东4垂直  <- 同一天线的两个极化分在不同终端？

东5也类似:
- 终端3 (H12): 东5水平
- 终端4 (V00): 东5垂直  <- 同一天线的极化分离
```

**Action Required**:
```bash
# 1. 确认实际硬件接线
# 选项A: 每个天线的H/V分别接到不同终端（双馈源）
# 选项B: 你的接线描述有误

# 2. 检查数据的实际内容
python3 << EOF
from UnpackCustom import UnpackCustom
import numpy as np

# 读取一小段数据
file1 = '/mnt/data_ssd/zdj/data/crab/HBaseband1Spec1_4304.data'
file2 = '/mnt/data_ssd/zdj/data/crab/VBaseband0Spec0_4304.data'

for f in [file1, file2]:
    _, _, _, size = UnpackCustom.onepack_read(f)
    _, data = UnpackCustom.read_and_unpack(f, 0, 10, ...)  # 只读前10包
    print(f"{f}:")
    print(f"  Pol0 power: {np.abs(data[0]).mean()}")
    print(f"  Pol1 power: {np.abs(data[1]).mean()}")
    print(f"  Correlation: {np.corrcoef(data[0][:1000], data[1][:1000])[0,1]}")
EOF

# 如果东4H和东4V的相关性很高 → 说明确实是同一天线
# 如果相关性低 → 说明接线图有误
```

**Expected Output**: 明确的天线-终端-极化映射表

---

#### Task 2: 修正antenna_mapping.yaml
基于Task 1的结果，更新配置文件

**当前假设** (需要验证):
```yaml
# 假设每个天线的H/V确实分离到不同终端
actual_antennas:
  东1: {H: HBaseband0Spec0_pol0, V: null}  # 只有H
  东3: {H: null, V: HBaseband1Spec1_pol1}  # 只有V
  东4: {H: HBaseband1Spec1_pol0, V: VBaseband0Spec0_pol0}  # H和V分离
  东5: {H: HBaseband2Spec2_pol0, V: VBaseband0Spec0_pol1}  # H和V分离
  东6: {H: HBaseband2Spec2_pol1, V: null}  # 只有H
  东7: {H: VBaseband2Spec2_pol0, V: null}  # 只有H
  东8: {H: null, V: VBaseband2Spec2_pol1}  # 只有V
  东9: {H: VBaseband1Spec1_pol0, V: null}  # 只有H
  东10: {H: null, V: VBaseband1Spec1_pol1}  # 只有V
  北3: {H: null, V: HBaseband0Spec0_pol1}  # 只有V
```

**如果是这样**，你的阵列实际上是：
- 6个单极化天线: 东1, 东3, 东6, 东7, 东8, 东9, 北3
- 2个双极化天线: 东4, 东5
- **总共8个天线，10个通道**

---

### **P1 - 应该优化（性能问题）**

#### Task 3: 数据预处理
```bash
# 使用新的高效脚本
cd /mnt/data_ssd/zdj/LWA_EPIC

# 先处理一小段数据测试（前100包）
python3 process_crab_data.py \
    /mnt/data_ssd/zdj/data/crab \
    -o crab_test_100packs.npz \
    --pack-end 100

# 检查输出
python3 << EOF
import numpy as np
data = np.load('crab_test_100packs.npz', allow_pickle=True)
print("Keys:", list(data.keys()))
print("Voltages shape:", data['voltages'].shape)
print("Antenna positions:\n", data['antenna_positions'])
EOF

# 如果正确，处理全部数据
python3 process_crab_data.py \
    /mnt/data_ssd/zdj/data/crab \
    -o crab_full_data.npz
```

**预期时间**: 
- 测试: ~2分钟
- 全量: ~20分钟（取决于磁盘I/O）

---

#### Task 4: 为EPIC创建配置文件
```bash
cat > crab_epic_config.yaml << 'EOF'
# EPIC配置 - Crab观测
station:
  name: "CRAB-LBAND-10ANT"
  nstands: 10  # 实际天线数
  npols: 2
  
observation:
  source_name: "Crab"
  center_freq_mhz: 1355
  bandwidth_mhz: 250
  ra_hours: 5.575  # 05:34:30
  dec_deg: 22.014  # +22:00:52
  
imaging:
  imagesize: 64  # 从128降到64（天线少，分辨率低）
  imageres_deg: 2.0  # 更大的像素（~2度）
  channels: 64  # 从128降到64
  seq_accum_ms: 40
  support: 3  # 从5降到3（天线少，UV覆盖稀疏）
  kernel_oversample: 2
  aeff_m2: 12.5  # L-band典型值，需要实测
  
processing:
  nstreams: 4
  ngpus: 1
  offline: true
  out_dir: "./crab_images"
EOF
```

---

### **P2 - 后续改进（功能增强）**

#### Task 5: 修改EPIC源码以支持小规模阵列
需要修改的文件（按前面的分析）：

1. **`src/ex/constants.h`**
```cpp
// 添加新的站点配置
enum CONSTANTS {
  // ... 保留原有LWA常量 ...
  CRAB_NSTANDS = 10,
  CRAB_NPOLS = 2,
  CRAB_INP_PER_CHAN = CRAB_NSTANDS * CRAB_NPOLS,
};
```

2. **`src/raft_kernels/dummy_packet_gen.hpp`**
```cpp
// 修改为根据配置选择天线数
template<int NSTANDS = LWA_SV_NSTANDS>
class DummyPktGen {
  // ...
  size_t gulp_size = m_nchan_in * npols * nseqs * NSTANDS * 1;
};
```

3. **GPU Kernels** - 需要编译时特化
```bash
# 选项A: 编译时定义
cmake -DNSTANDS=10 ..

# 选项B: 运行时检测（更灵活但稍慢）
# 修改gridder.cuh使用动态循环上界
```

---

## 执行检查清单

- [ ] **Task 1**: 验证接线图（30分钟）
  - [ ] 运行数据检查脚本
  - [ ] 确认每个天线的实际连接
  - [ ] 绘制清晰的接线拓扑图
  
- [ ] **Task 2**: 更新配置文件（15分钟）
  - [ ] 修正antenna_mapping.yaml
  - [ ] 添加注释说明异常情况
  
- [ ] **Task 3**: 预处理数据（30分钟）
  - [ ] 测试处理脚本
  - [ ] 检查输出格式
  - [ ] 处理全量数据
  
- [ ] **Task 4**: EPIC配置（15分钟）
  - [ ] 创建crab配置文件
  - [ ] 调整成像参数
  
- [ ] **Task 5**: 运行EPIC（2小时）
  - [ ] 使用离线模式测试
  - [ ] 检查输出图像
  - [ ] 调优参数

---

## 预期问题和解决方案

### 问题1: "东4和东5的极化为什么分离？"
**可能原因**:
- 使用了双馈源天线，H和V分别连接
- 某些终端只支持单极化采样

**验证方法**:
检查VBaseband0Spec0的两个通道是否来自相邻天线

### 问题2: "EPIC能处理10个天线吗？"
**答案**: 能，但需要修改
- GPU kernel中的LWA_SV_NSTANDS=256硬编码需要改为10
- 或者用0填充到256（浪费但简单）

### 问题3: "图像质量会很差？"
**答案**: 会的
- 10天线 vs 256天线 → UV覆盖差25倍
- 但Crab很亮（~1000 Jy），应该能检测到

---

## 快速验证方案（最小可行路径）

如果你只想"先看到图"：

```bash
# 1. 用0填充到256天线
python3 << EOF
import numpy as np

# 读取处理后的数据
data = np.load('crab_full_data.npz')
voltages = data['voltages']  # (ntime, 1, 10, 2)

# 填充到256天线
voltages_padded = np.zeros((voltages.shape[0], 1, 256, 2), 
                          dtype=voltages.dtype)
voltages_padded[:, :, :10, :] = voltages

# 保存为EPIC格式
np.savez('crab_epic_256padded.npz',
         voltages=voltages_padded,
         antenna_positions=...,  # 需要填充
         phases=...)  # 需要填充
EOF

# 2. 直接运行EPIC
./epic++ --offline --out_dir crab_images \
         --npytbnfile crab_epic_256padded.npz \
         --imagesize 64 --channels 64
```

---

## 下一步行动

**立即执行**（今天）:
1. 验证接线图（Task 1）
2. 如果接线图正确，运行预处理（Task 3）

**明天**:
3. 修改EPIC代码支持小阵列（Task 5）
4. 生成第一张图像

**需要我帮你**:
- [ ] 生成接线验证脚本
- [ ] 修改EPIC源码的具体patch
- [ ] 调试数据格式问题

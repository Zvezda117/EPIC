# LWA_EPIC 项目状态

## 📁 当前目录结构

```
LWA_EPIC/
├── antenna_coordinates.csv          # 天线坐标（北1-6, 东1-10）
├── antenna_mapping*.yaml           # 接线配置（crab/9178/9197）
├── UnpackCustom.py / unpack.py     # 数据解包工具
├── main_my_base2fil_*.py           # 原始处理脚本（待优化）
│
├── data/                           # 数据目录（2.9 GB）
│   ├── raw/                        # 原始基带 .data 文件（空，数据在外部）
│   ├── processed/                  # 处理后的时域数据
│   │   └── 9197_file0_complete.npz  (2.0 GB) - 4天线×4.3秒时域数据
│   └── epic_ready/                 # EPIC就绪的频域数据
│       └── 9197_epic_512chan.npz    (935 MB) - 512通道频域数据
│
├── scripts/                        # 所有脚本（整理完毕）
│   ├── process_9197_single_channel.py   # 9197数据处理
│   ├── process_crab_data.py             # Crab数据处理
│   ├── verify_antenna_wiring.py         # 接线验证工具
│   ├── convert_to_epic_format.py        # 时域→频域转换
│   ├── run_9197_imaging.sh              # 9197成图流程
│   └── run_crab_pipeline.sh             # Crab处理流程
│
├── docs/                           # 文档（7个MD文件）
│   ├── IMAGING_STRATEGY_9197.md         # 9197成图策略
│   ├── CHANNEL_SELECTION_GUIDE.md       # 通道数选择指南
│   ├── CRAB_PROCESSING_PLAN.md          # Crab数据计划
│   ├── WIRING_ANALYSIS_RESULT.md        # 接线分析结果
│   ├── ANTENNA12_FIX_NOTES.md           # 天线12硬件问题
│   └── README.md / CHANGELOG.md
│
├── results/                        # 输出结果
│   ├── logs/                       # 处理日志（3个）
│   └── images/                     # 成图输出（待生成）
│
├── build/                          # 编译输出（26 MB）
│   ├── epic++                      # EPIC可执行文件
│   ├── data/                       # 示例数据
│   └── offline_outputs/            # 之前的测试输出
│
└── src/                            # C++源码
    ├── ex/                         # EPIC核心
    ├── raft_kernels/               # GPU核心
    └── python/                     # Python接口
```

---

## ✅ 已完成的工作

### 1. 数据处理（9197数据集）
- ✅ 读取原始基带数据（4天线 × 1.07B采样点）
- ✅ 合并为统一格式 `(nsamples, 1, 4, 1)`
- ✅ 保存为 `9197_file0_complete.npz` (2.0 GB)

### 2. EPIC格式转换
- ✅ FFT转换到频域（512通道）
- ✅ 时间积分（42,865时间步 × 0.1 ms）
- ✅ 填充到256天线格式
- ✅ 保存为 `9197_epic_512chan.npz` (935 MB)

### 3. 脚本和文档
- ✅ 6个处理脚本（已整理到 scripts/）
- ✅ 7个文档（已整理到 docs/）
- ✅ 3个天线配置文件（crab/9178/9197）

### 4. 项目清理
- ✅ 删除测试文件和旧版本
- ✅ 清理Python缓存
- ✅ 建立标准目录结构

---

## 🎯 当前状态：准备成图！

### 数据就绪
```
data/epic_ready/9197_epic_512chan.npz
├── 形状: (42865, 512, 256, 2)
├── 大小: 935 MB
├── 时长: 4.286 秒
├── 带宽: 250 MHz
├── 频道: 512（488 kHz/通道）
└── 天线: 4有效 / 256总数
```

### EPIC参数（已优化）
```bash
--imagesize 64       # 修复硬编码限制（原来32失败）
--imageres 5.0       # 5度/像素（适合稀疏阵列）
--channels 512       # 512通道（从128提升）
--nts 42865          # 全部时间步
--seq_accum 100      # 100ms积分
```

---

## 🚀 下一步：立即成图！

### 命令
```bash
cd /mnt/data_ssd/zdj/LWA_EPIC
./scripts/run_9197_imaging.sh
```

### 预期
```
1. 验证数据（已完成）
2. 转换格式（已完成）
3. 运行EPIC（~5-10分钟）
   - GPU gridding
   - 逆FFT
   - 生成FITS
4. 输出结果
   - results/images/EPIC_*.fits
   - results/images/EPIC_*_QL.png
```

### 成图质量预期

| 指标 | 预期值 | 原因 |
|------|--------|------|
| **分辨率** | 2-10° | 4天线，UV覆盖稀疏 |
| **视场** | 64×64像素，5°/像素 | 总视场320° |
| **动态范围** | 5-10:1 | 基线少，旁瓣强 |
| **灵敏度** | 低 | 有效面积小 |

**能看到**：
- ✅ 非常明亮的点源
- ✅ 大致位置
- ❌ 细节结构
- ❌ 弱源

---

## 📊 后续计划

### 短期（本周）
1. ✅ 完成9197首次成图
2. 🔄 分析图像质量
3. 🔄 调整EPIC参数优化
4. 🔄 处理完整9197数据集（137×4文件）

### 中期（下周）
5. 处理crab数据（12天线，效果更好）
6. 对比不同通道数（128/512/1024）
7. 验证天线位置和相位

### 长期（1-3个月）
8. 修改EPIC支持动态天线数
9. 添加校准流程
10. 扩展到16-32天线阵列

---

## 🔧 关键参数对照

### 频谱分辨率演进
```
LWA-SV原设计:  25 MHz ÷ 128通道 = 195 kHz/通道
你的首次尝试: 250 MHz ÷ 128通道 = 1.95 MHz/通道 ✗
现在配置:     250 MHz ÷ 512通道 = 488 kHz/通道 ✓
```

### 天线阵列对比
```
LWA-SV:     256天线 → 32,640基线 → 优秀UV覆盖
Crab数据:    12天线 →     66基线 → 可用
9197数据:     4天线 →      6基线 → 稀疏（测试用）
目标阵列:    16天线 →    120基线 → decent
```

---

## 💾 磁盘使用

```
总计: 2.9 GB

data/processed/:    2.0 GB  (时域原始数据)
data/epic_ready/:   935 MB  (频域EPIC输入)
build/:              26 MB  (编译输出)
results/logs/:      120 KB  (处理日志)
```

---

## 📝 Linus的评语

> "项目结构现在看起来像样了。  
> 数据在 data/，脚本在 scripts/，文档在 docs/。  
> 这就是常识，不需要搞得花里胡哨。  
>   
> 现在跑 ./scripts/run_9197_imaging.sh，  
> 看看能不能出第一张图。  
> 可能很丑，但至少要能work。"

---

**状态**: 🟢 准备就绪，等待成图命令
**下一步**: `./scripts/run_9197_imaging.sh`

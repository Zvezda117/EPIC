# 🚀 下一步：立即成图！

## 当前状态
```
✅ 项目结构已整理
✅ 数据已处理完成（2.0 GB时域数据）
✅ EPIC格式已转换（935 MB，512通道）
✅ 脚本已更新路径
```

---

## 【立即执行】

### 单命令成图
```bash
cd /mnt/data_ssd/zdj/LWA_EPIC && ./scripts/run_9197_imaging.sh
```

### 预期流程
```
步骤1: 验证数据 ✓ (已完成)
步骤2: 检查EPIC格式 ✓ (已完成)
步骤3: 检查EPIC编译 ✓ (应该已编译)
步骤4: 运行EPIC成像 → 5-10分钟
```

### 输出位置
```
results/images/EPIC_*.fits    # FITS格式天图
results/images/EPIC_*_QL.png  # 快视图PNG
```

---

## 【预期结果】

### 数据参数
- **时长**: 4.286 秒
- **频率**: 1355 MHz中心，250 MHz带宽
- **通道**: 512 (488 kHz/通道)
- **天线**: 4个（北3, 北4, 北5, 北6）
- **基线**: 6条

### 图像参数
- **尺寸**: 64×64 像素
- **分辨率**: 5°/像素
- **视场**: 320° × 320° (全天)
- **有效分辨率**: ~2-10° (UV稀疏)

### 图像质量
```
✅ 能看到：极亮点源（如果在视场内）
✅ 能看到：大致位置
⚠️  可能看到：强烈旁瓣和伪影
❌ 看不到：弱源、细节、精确流量
```

**Linus评价**: "It works. It's shit, but it works."

---

## 【可能遇到的问题】

### 问题1: EPIC未编译
```bash
# 解决方案
cd build
cmake ..
make -j$(nproc)
cd ..
./scripts/run_9197_imaging.sh
```

### 问题2: GPU内存不足
```bash
# 解决方案：减少时间步
# 编辑 scripts/run_9197_imaging.sh
# 修改 --nts 42865 为 --nts 10000
```

### 问题3: 数据路径错误
```bash
# 验证数据位置
ls -lh data/processed/9197_file0_complete.npz
ls -lh data/epic_ready/9197_epic_512chan.npz

# 如果不存在，重新运行处理
cd scripts
python process_9197_single_channel.py --files 0 -o ../data/processed/9197_file0.npz
python convert_to_epic_format.py \
    ../data/processed/9197_file0.npz \
    -o ../data/epic_ready/9197_epic_512chan.npz \
    --nchan 512
```

---

## 【成功后的下一步】

### 1. 查看图像
```bash
# 用DS9查看FITS
ds9 results/images/EPIC_*.fits

# 或直接查看PNG
eog results/images/EPIC_*_QL.png
```

### 2. 分析结果
- 是否有明亮点源？
- 旁瓣强度如何？
- RFI污染情况？

### 3. 参数优化
根据结果调整：
- `--seq_accum`: 积分时间（100ms → 500ms）
- `--imageres`: 像素分辨率（5° → 10°）
- `--support`: 卷积核支持（2 → 3）

### 4. 处理更多数据
```bash
# 处理完整9197数据集（137个文件）
cd scripts
python process_9197_single_channel.py --files 0-136
```

### 5. 处理crab数据（12天线）
```bash
# Crab有12个天线，效果会好得多
./scripts/run_crab_pipeline.sh
```

---

## 【关键命令速查】

```bash
# 成图
./scripts/run_9197_imaging.sh

# 查看数据
python -c "import numpy as np; d=np.load('data/epic_ready/9197_epic_512chan.npz'); print(d.files); print(d['data'].shape)"

# 检查EPIC状态
./build/epic++ --help

# 查看结果
ls -lh results/images/

# 清理重跑（如需要）
rm -rf results/images/EPIC_*
./scripts/run_9197_imaging.sh
```

---

## 🎯 立即行动

```bash
cd /mnt/data_ssd/zdj/LWA_EPIC
./scripts/run_9197_imaging.sh
```

**预计时间**: 5-10分钟  
**预计输出**: 1-2个FITS文件 + 快视图

Let's make it work! 🚀

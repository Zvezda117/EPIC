# 9197数据处理进展总结

## 时间线：Jan 28-29, 2026

---

## ✅ 已完成的工作

### 1. 项目整理
- ✅ 清理临时文件和测试数据
- ✅ 建立标准目录结构 (data/, scripts/, docs/, results/)
- ✅ 整理6个处理脚本
- ✅ 创建7个技术文档

### 2. 数据处理
- ✅ 解包9197原始基带数据（4天线×4.3秒）
- ✅ 合并为统一格式 `(1071644672, 1, 4, 1)`
- ✅ FFT转换到频域（512通道）
- ✅ 填充到256天线格式
- ✅ 保存为EPIC输入格式

### 3. 源码修复
- ✅ 识别3个硬编码限制Bug
- ✅ 修改`MAX_CHANNELS_4090`: 128 → 1024
- ✅ 修复`gulp_size`逻辑bug
- ✅ 重新编译EPIC（成功）

---

## 🔴 当前问题

### 错误信息
```
F20260129 16:12:07.095445 MOFF_correlator.hpp:272] 
Number of output channels exceeds the input channels
```

### 原因分析

EPIC在运行时检查：
```cpp
if (m_nchan_out > m_nchan_in) {
    LOG(FATAL) << "Number of output channels exceeds the input channels";
}
```

- `--channels 512` 设置 `m_nchan_out = 512`
- 但 `m_nchan_in` 可能从NPZ文件读取时不正确

### 可能原因

1. **NPZ文件格式不对**
   - EPIC期望特定的键名和维度顺序
   - 我们的转换脚本可能不匹配

2. **数据读取逻辑问题**
   - EPIC的NPZ读取器可能有特定假设
   - 需要查看`dummy_packet_gen.hpp`

3. **元数据缺失**
   - 可能缺少`meta`数组或格式不对

---

## 📋 发现的Bug文档

### Bug #1: gulp_size ∝ nts
- **位置**: `src/ex/option_parser.hpp:144`
- **问题**: `gulp_size = nts × 0.04` 导致nts大时需要超大seq_accum
- **修复**: 改为固定`gulp_size = 40ms`
- **状态**: ✅ 已修复并编译

### Bug #2: 128通道限制
- **位置**: `src/ex/constants.h:63`
- **问题**: `MAX_CHANNELS_4090 = 128` 但GPU实际可支持更多
- **修复**: 改为1024
- **状态**: ✅ 已修复并编译

### Bug #3: imagesize硬编码
- **位置**: `src/ex/option_parser.hpp:140`
- **问题**: 只允许64或128
- **修复**: （未修复，可用64暂时绕过）
- **状态**: ⏭️ 待修复

### Bug #4: 输入/输出通道数不匹配（新发现）
- **位置**: `src/ex/MOFF_correlator.hpp:272`
- **问题**: NPZ加载时通道数识别错误
- **修复**: 需要查看数据加载逻辑
- **状态**: 🔄 调查中

---

## 📊 数据统计

### 原始数据
```
位置: /mnt/DATA/存储阵列/9197_Base2_ADC16_26-01-18_19-16-51/
文件: 137 × 4天线 = 548文件
格式: int8复数，IQ交织
大小: ~1GB/天线
时长: 4.3秒/文件 × 137 = 588秒 = 9.8分钟
```

### 处理后数据
```
时域数据: data/processed/9197_file0_complete.npz (2.0 GB)
  - 形状: (1071644672, 1, 4, 1)
  - 时长: 4.287秒
  - 天线: 4 (北3,4,5,6)
  
频域数据: data/epic_ready/9197_epic_512chan.npz (935 MB)
  - 形状: (42865, 512, 256, 2)
  - 时间步: 42865 (0.1ms/步)
  - 通道: 512 (488 kHz/通道)
  - 天线槽: 256 (填充，实际4)
```

---

## 🎯 下一步行动

### 立即（0-1小时）
1. ✅ 检查NPZ数据格式和示例数据对比
2. 🔄 查看EPIC的NPZ加载代码
3. 🔄 修复数据格式或转换脚本
4. 🔄 重新生成NPZ文件
5. 🚀 运行EPIC成图

### 短期（1-3天）
6. 处理完整9197数据（137文件）
7. 处理crab数据（12天线）
8. 调整EPIC参数优化图像质量
9. 验证天线位置和相位

### 中期（1-2周）
10. 修改EPIC支持动态天线数
11. 添加校准流程
12. 批量处理pipeline
13. 自动化成图脚本

---

## 💡 关键技术洞察

### Linus的评价

> "We spent hours debugging hardcoded limits  
> that shouldn't exist in the first place.  
>   
> 128 channel limit? Just a CLI check.  
> gulp_size bug? Variable naming fuck-up.  
> NPZ format issue? Probably undocumented assumptions.  
>   
> This is what happens when code is optimized  
> for ONE specific use case (LWA-SV)  
> and then needs to adapt to something else.  
>   
> But we're close. Fix the NPZ format,  
> and we'll have our first image."

### 数据利用率演进

```
初始workaround:  0.6% (128通道 × 1000步)
源码修复后:      可以用100% (512通道 × 42865步)
当前阻塞:        NPZ格式问题
```

### 工程经验

1. **Don't workaround, fix it**
   - 修改源码比绕过限制更快
   - 20分钟修复 vs 无尽的妥协

2. **Read the fucking code**
   - 所有"限制"都只是代码
   - 大部分可以安全修改

3. **Test incrementally**
   - 从小数据开始 (1000步)
   - 逐步增加到完整数据 (42865步)

4. **Document everything**
   - Bug文档帮助理解系统
   - 避免重复踩坑

---

## 📁 文件清单

### 数据文件
- `data/processed/9197_file0_complete.npz` - 时域数据 (2.0 GB)
- `data/epic_ready/9197_epic_512chan.npz` - 频域数据 (935 MB)

### 脚本
- `scripts/process_9197_single_channel.py` - 解包并处理基带数据
- `scripts/convert_to_epic_format.py` - 时域→频域转换
- `scripts/run_9197_final.sh` - 自动化成图流程
- `scripts/verify_antenna_wiring.py` - 接线验证

### 文档
- `docs/EPIC_PARAMETER_BUGS.md` - Bug详细文档
- `docs/CHANNEL_LIMIT_ANALYSIS.md` - 128通道限制分析
- `docs/IMAGING_STRATEGY_9197.md` - 成图策略
- `docs/CHANNEL_SELECTION_GUIDE.md` - 通道数选择指南
- `docs/PROJECT_STATUS.md` - 项目状态
- `docs/NEXT_STEP.md` - 下一步指南
- `PROGRESS_SUMMARY.md` - 本文档

### 日志
- `results/logs/epic_512chan_full_*.log` - EPIC运行日志
- `results/logs/recompile_*.log` - 编译日志

---

## 🔧 修改的源文件

1. `src/ex/constants.h`
   - Line 63: `MAX_CHANNELS_4090 = 1024`

2. `src/ex/option_parser.hpp`
   - Lines 144-151: 修复gulp_size逻辑

---

## 📞 当前状态

**等待**: NPZ数据格式检查完成  
**下一步**: 修复数据格式并重新测试  
**预计时间**: 1-2小时内出第一张图

**最后更新**: 2026-01-29 16:12 UTC+8

# EPIC参数Bug文档

## Bug #1: gulp_size与nts的反常关系

### 问题描述

EPIC的参数验证逻辑存在设计缺陷：

```cpp
// src/ex/option_parser.hpp:144-148
int nts = result["nts"].as<int>() * 40e-3;  // ← 这里把nts转换成了毫秒！
int accumulate = result["seq_accum"].as<int>();

if (accumulate < nts) {  // ← 但变量名还叫nts，实际是gulp_size
    return "Sequence accumulation time must be greater than the gulp size.";
}
```

### 问题本质

1. **变量命名混乱**
   - `nts` 在第144行被乘以`0.04`后，已经不是"时间步数"，而是"gulp大小（毫秒）"
   - 但变量名还是`nts`，导致代码难以理解

2. **反常的缩放关系**
   ```
   gulp_size = nts × 0.04 ms
   
   例如：
   nts=1000  → gulp_size=40 ms   ✓ 合理
   nts=10000 → gulp_size=400 ms  ⚠️ 变大了
   nts=42865 → gulp_size=1714.6 ms ❌ 太大！
   ```

3. **违反直觉的约束**
   - **直觉**：处理更多时间步（nts），不应该要求更长的积分时间（seq_accum）
   - **实际**：nts越大，gulp_size越大，seq_accum必须更大才能满足条件

### Linus评价

> "This is backwards as fuck.  
> More timestamps should give you MORE flexibility, not LESS.  
> And that variable naming? `nts` is not nts anymore after line 144.  
> This is the kind of shit that makes debugging a nightmare."

### 实际影响

#### 场景A：短观测（正常工作）
```bash
--nts 1000 --seq_accum 100
gulp_size = 1000 × 0.04 = 40 ms
100 > 40 ✓ 通过
```

#### 场景B：长观测（失败）
```bash
--nts 42865 --seq_accum 500
gulp_size = 42865 × 0.04 = 1714.6 ms
500 < 1714.6 ✗ 失败："Sequence accumulation time must be greater than the gulp size"
```

### 解决方案

#### 方案1：减少nts（临时workaround）
```bash
# 只处理前1000个时间步（约100ms数据）
--nts 1000 --seq_accum 100
```

**优点**：
- 简单，立即work
- 适合快速测试

**缺点**：
- 浪费数据（4.3秒数据只用了0.1秒）
- 损失灵敏度

#### 方案2：增大seq_accum（不推荐）
```bash
# 强行满足条件
--nts 42865 --seq_accum 2000
```

**优点**：
- 使用全部数据

**缺点**：
- 时间分辨率变差（2秒积分）
- 违背观测设计意图

#### 方案3：分块处理（正确但复杂）
```python
# 将42865时间步分成43批，每批1000步
for i in range(43):
    epic++ --nts 1000 --seq_accum 100 --offset $((i*1000))
```

**优点**：
- 使用全部数据
- 保持时间分辨率

**缺点**：
- 需要修改EPIC支持offset参数（当前不支持）
- 或者手动切分NPZ文件

#### 方案4：修改EPIC源码（根本解决）

**修改 src/ex/option_parser.hpp**：

```cpp
// 修改前（Line 144-148）
int nts = result["nts"].as<int>() * 40e-3;  // ← 混乱的命名
int accumulate = result["seq_accum"].as<int>();
if (accumulate < nts) {

// 修改后
int nts = result["nts"].as<int>();
int gulp_size_ms = 40;  // ← 明确的常量，或从配置读取
int accumulate = result["seq_accum"].as<int>();
if (accumulate < gulp_size_ms) {
    return "Sequence accumulation time must be greater than " + 
           std::to_string(gulp_size_ms) + " ms";
}
```

**彻底修复**：
```cpp
// gulp_size 应该是固定值或者从带宽计算
// 不应该依赖 nts！

constexpr int DEFAULT_GULP_SIZE_MS = 40;
int gulp_size = DEFAULT_GULP_SIZE_MS;  // 或从带宽计算
int accumulate = result["seq_accum"].as<int>();

if (accumulate < gulp_size) {
    return "seq_accum (" + std::to_string(accumulate) + 
           " ms) must be >= gulp_size (" + std::to_string(gulp_size) + " ms)";
}
```

---

## Bug #2: RTX 4090通道数限制

**位置**：`src/ex/option_parser.hpp:168-170`

```cpp
if (channels > MAX_CHANNELS_4090) {
    return "RTX 4090 only supports output channels upto "s +
           std::to_string(MAX_CHANNELS_4090);
}
```

### 实际限制

```cpp
// 推测定义（需确认）
constexpr int MAX_CHANNELS_4090 = 128;
```

### 问题

1. **硬编码GPU型号**
   - 代码中写死了"RTX 4090"
   - 其他GPU可能有不同的限制

2. **通道数限制过严**
   - 250 MHz带宽，128通道 = 1.95 MHz/通道（太粗！）
   - 应该支持512+通道for精细频谱分辨率

3. **缺少权衡说明**
   - 为什么是128？内存限制？计算限制？
   - 能否通过减少imagesize来增加通道数？

### Linus评价

> "So we have 250 MHz bandwidth but can only use 128 channels?  
> That's 1.95 MHz per channel - totally inadequate for RFI flagging.  
> This is a GPU memory issue, not a fundamental limit.  
> Should be: warn user, let them decide."

### 当前Workaround

```bash
# 被迫使用128通道
--channels 128  # 而不是理想的512-1024
```

**影响**：
- 频谱分辨率：1.95 MHz/通道（vs 理想的 488 kHz 或 244 kHz）
- RFI处理能力：大幅降低
- 科学价值：受限

---

## Bug #3: imagesize硬编码

**位置**：`src/ex/option_parser.hpp:140-142`

```cpp
if (image_size != 64 && image_size != 128) {
    return "Invalid image size: " + std::to_string(image_size) +
           ". Image size can only be 64 or 128";
}
```

### Linus评价

> "Why 64 or 128? Is it CUDA warp size? FFT optimization?  
> Fine, but this check belongs in the GPU kernel, not the CLI parser.  
> Command-line interface should accept any size and let the backend decide."

### 解决方案

```cpp
// 改成警告而不是硬错误
if (image_size != 64 && image_size != 128) {
    LOG(WARNING) << "Non-standard image size: " << image_size 
                 << ". Performance may be suboptimal (optimized for 64/128).";
}

// 或者至少支持2的幂次
if ((image_size & (image_size - 1)) != 0 || image_size < 32 || image_size > 256) {
    return "Image size must be a power of 2 between 32 and 256";
}
```

---

## 当前Workaround配置

基于这些bug，当前推荐的9197数据成图参数：

```bash
./epic++ --offline \
    --npytbnfile data/epic_ready/9197_epic_512chan.npz \
    --imagesize 64 \           # ← 受限于bug #2
    --imageres 5.0 \
    --channels 512 \
    --nts 1000 \              # ← 受限于bug #1（只用0.1秒数据）
    --seq_accum 100 \         # ← 满足 100 > 1000×0.04
    --support 2 \
    --aeff 10
```

**使用的数据**：
- 总数据：4.287秒
- 实际使用：0.1秒（2.3%）
- 浪费：4.187秒（97.7%）

---

## 长期修复计划

### Phase 1：理解gulp_size含义
1. 查看 `src/raft_kernels/accumulator.hpp` 中的实现
2. 理解为什么是40ms
3. 是否可以从数据带宽动态计算？

### Phase 2：修改参数验证
1. 解耦 gulp_size 和 nts
2. 改进错误消息
3. 添加更多验证和提示

### Phase 3：支持分块处理
1. 添加 `--offset` 参数
2. 或者添加 `--auto_batch` 模式，自动分批处理长数据

### Phase 4：去除硬编码
1. imagesize支持任意2的幂次
2. 动态天线数支持
3. 动态通道数上限

---

## 总结

**Linus的最终评价**：

> "EPIC works, but it's full of hardcoded assumptions.  
> That's fine for a research prototype.  
> But if you want to adapt it to a new array, you need to:  
>   
> 1. Understand these assumptions  
> 2. Work around them (short term)  
> 3. Fix them properly (long term)  
>   
> For now, use nts=1000 and get your first image.  
> Then we'll fix this shit properly."

**当前状态**：
- ✅ Workaround已实现
- ✅ 参数已调整为 nts=1000, seq_accum=100
- 🔄 等待第一张图输出
- ⏭️ 下一步：修改源码支持长时间观测

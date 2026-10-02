# 最终解决方案：EPIC Offline数据处理

## 问题根源

EPIC的offline模式（`DummyPktGen`）**硬编码了128通道**：

```cpp
// src/raft_kernels/dummy_packet_gen.hpp:44
const int m_nchan_in{128};  // ← 硬编码！
```

且创建时不读取NPZ文件信息：
```cpp
// src/raft_kernels/kernel_types.hpp:120  
return ktype(1);  // 没有传递npytbnfile参数
```

---

## 解决方案对比

### 方案A：快速workaround（推荐先做）⏱️ 10分钟

生成128通道版本数据并成图：

```bash
cd /mnt/data_ssd/zdj/LWA_EPIC

# 1. 转换为128通道（如果还没有）
python3 scripts/convert_to_epic_format.py \
    data/processed/9197_file0_complete.npz \
    -o data/epic_ready/9197_epic_128chan.npz \
    --nchan 128 \
    --ntime-samples 25000

# 2. 运行EPIC（修复后的版本）
cd build
./epic++ --offline \
    --npytbnfile ../data/epic_ready/9197_epic_128chan.npz \
    --out_dir ../results/images \
    --imagesize 64 \
    --imageres 5.0 \
    --channels 128 \
    --seq_accum 100 \
    --nts 10000 \  # 可以用更多时间步了！
    --support 2 \
    --aeff 10 \
    --ngpus 1 \
    --gpu_ids=0
```

**优点**：
- ✅ 立即可用
- ✅ 能用10000+时间步（不再受gulp_size限制）
- ✅ 出第一张图（验证流程）

**缺点**：
- ❌ 频谱分辨率差（1.95 MHz/通道）
- ❌ 不是最终解决方案

---

### 方案B：正确修复（完整解决）⏱️ 2-4小时

修改EPIC源码支持动态通道数：

#### Step 1: 修改DummyPktGen读取NPZ元数据

**src/raft_kernels/dummy_packet_gen.hpp**

```cpp
// 修改前
class DummyPktGen : public raft::kernel {
 private:
  const int m_nchan_in{128};  // ← 硬编码

// 修改后
class DummyPktGen : public raft::kernel {
 private:
  int m_nchan_in{128};  // ← 去掉const，允许修改
  std::string m_npz_file;  // ← 添加NPZ文件路径
  
 public:
  DummyPktGen(size_t p_n_pkts = 1, 
              std::string npz_file = "",  // ← 新参数
              std::string utcstart = "2023_06_19T00_00_00")
      : raft::kernel(), 
        m_n_pkts(p_n_pkts),
        m_npz_file(npz_file) {
    
    // 如果提供了NPZ文件，读取实际通道数
    if (!m_npz_file.empty()) {
      m_nchan_in = ReadNpzChannels(m_npz_file);
      LOG(INFO) << "Loaded NPZ file with " << m_nchan_in << " channels";
    }
    // ... 其余代码
  }
  
 private:
  int ReadNpzChannels(const std::string& npz_file) {
    // 使用Python接口读取NPZ
    // 或者使用cnpy库
    // 返回data.shape[1]
  }
};
```

#### Step 2: 传递npytbnfile参数

**src/raft_kernels/kernel_types.hpp**

```cpp
// 修改前
template <>
struct Kernel<_DUMMY_PACK_GEN> : KernelTypeDefs {
  template <unsigned int _GpuId>
  static ktype get_kernel(const opt_t&) {
    return ktype(1);  // ← 没有传递参数
  }
};

// 修改后
template <>
struct Kernel<_DUMMY_PACK_GEN> : KernelTypeDefs {
  template <unsigned int _GpuId>
  static ktype get_kernel(const opt_t& opts) {
    std::string npz_file = "";
    if (opts.count("npytbnfile")) {
      npz_file = opts["npytbnfile"].as<std::string>();
    }
    return ktype(1, npz_file);  // ← 传递NPZ文件路径
  }
};
```

#### Step 3: 实现NPZ读取（可选：使用cnpy）

添加依赖或使用Python接口读取NPZ元数据。

#### Step 4: 重新编译

```bash
cd build
make -j$(nproc)
```

**优点**：
- ✅ 真正解决问题
- ✅ 支持任意通道数
- ✅ 未来可扩展

**缺点**：
- ❌ 需要更多开发时间
- ❌ 需要测试NPZ读取逻辑

---

## 推荐流程

### Phase 1: 立即行动（今天）

1. ✅ 源码已修复（gulp_size, MAX_CHANNELS_4090）
2. 🔄 用方案A：生成128通道数据
3. 🚀 运行EPIC，出第一张图
4. 📊 验证流程和图像质量

### Phase 2: 完整修复（明天）

5. 🔧 实现方案B：修改DummyPktGen
6. ✅ 测试512通道
7. 🎯 处理完整数据（42865时间步）
8. 📈 对比128 vs 512通道图像质量

### Phase 3: 生产级（本周）

9. 处理完整9197数据集（137文件）
10. 处理crab数据（12天线）
11. 自动化pipeline
12. 文档化最佳实践

---

## 立即执行命令

```bash
cd /mnt/data_ssd/zdj/LWA_EPIC

# Step 1: 检查128通道数据是否存在
ls -lh data/epic_ready/9197_epic_128chan.npz 2>/dev/null || \
python3 scripts/convert_to_epic_format.py \
    data/processed/9197_file0_complete.npz \
    -o data/epic_ready/9197_epic_128chan.npz \
    --nchan 128 \
    --ntime-samples 25000

# Step 2: 运行EPIC（可以用更多时间步了！）
cd build
./epic++ --offline \
    --npytbnfile ../data/epic_ready/9197_epic_128chan.npz \
    --out_dir ../results/images \
    --disable_metrics \
    --imagesize 64 \
    --imageres 5.0 \
    --channels 128 \
    --seq_accum 100 \
    --nts 20000 \
    --support 2 \
    --aeff 10 \
    --kernel_oversample 2 \
    --nstreams 4 \
    --ngpus 1 \
    --gpu_ids=0

# Step 3: 查看结果
cd ..
ls -lh results/images/EPIC_*.fits
```

---

## 数据利用率对比

| 方案 | 通道 | 时间步 | 利用率 | 图像质量 |
|------|------|--------|--------|----------|
| 初始workaround | 128 | 1000 | 0.6% | 很差 |
| 方案A | 128 | 20000 | 11.7% | 可用 |
| 方案B（目标） | 512 | 42865 | 100% | 最佳 |

---

## Linus的最终建议

```text
"Right now, you have TWO options:

Option 1: Use 128 channels, get your first image TODAY.
          It won't be great, but it WORKS.
          And you can use 20000 timesteps now (not just 1000).
          That's 20x better than before.

Option 2: Spend 2-4 hours fixing DummyPktGen properly.
          Then you can use 512 channels and ALL your data.

My advice?

Do BOTH.

Use Option 1 NOW to verify the pipeline works.
Then do Option 2 to get proper results.

Don't let perfect be the enemy of good.
Get the first image. Then make it better."
```

---

## 总结

**当前状态**：
- ✅ 源码Bug已修复（2个）
- ✅ 可以使用更多时间步（不受gulp_size限制）
- 🔴 仍需workaround 128通道限制
- ⏭️ 10分钟内可出第一张图

**下一步**：
1. 生成128通道数据（或使用现有512通道数据的前128通道）
2. 运行EPIC
3. 查看第一张图！
4. 然后修复DummyPktGen支持512通道

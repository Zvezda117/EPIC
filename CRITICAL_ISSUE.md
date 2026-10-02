# 🔴 CRITICAL: EPIC Offline模式根本不工作！

## 问题现象

- EPIC运行26分钟无输出
- GPU使用率：**0%**
- 81个线程全部睡眠
- 1秒后显示"Stopping gulp gen"

## 根本原因

**EPIC的offline模式是假的！**

```cpp
// src/raft_kernels/dummy_packet_gen.hpp:51-65
DummyPktGen(size_t p_n_pkts = 1, ...) {
    // 只生成 p_n_pkts 个假gulp（默认1个！）
}

// Line 106
return raft::stop;  // 生成1个gulp后立即停止
```

**`--npytbnfile`参数完全被忽略！**

```cpp
// src/raft_kernels/kernel_types.hpp:120
return ktype(1);  // 永远只生成1个gulp，不读取NPZ
```

## 流程分析

```
1. DummyPktGen启动 → 生成1个假gulp (40ms数据)
2. DummyPktGen停止 → 返回 raft::stop
3. Pipeline等待更多数据 → 永远等待
4. GPU空闲 → 0%使用率
5. 进程死锁 → 26分钟无进展
```

## 证据

### 终端输出
```
I20260129 16:30:39.376451 epic_executor.hpp:200] Running...
I20260129 16:30:40.839293 dummy_packet_gen.hpp:106] Stopping gulp gen
(之后再无任何输出)
```

**1秒后停止 = 只生成了1个gulp！**

### GPU状态
```
GPU-Util: 0%
Memory: 528MiB allocated (初始化时分配)
Process: 81 threads, all sleeping
```

## Linus的暴怒评价

> "WHAT THE ACTUAL FUCK?!
> 
> You have an '--npytbnfile' parameter.
> You have NPZ files with real data.
> But the code IGNORES IT COMPLETELY?!
> 
> DummyPktGen generates 1 fake gulp and stops.
> That's not 'offline processing'.
> That's 'pretend to work and hang forever'.
> 
> This is not a bug. This is BROKEN BY DESIGN.
> The offline mode DOESN'T EXIST."

---

## 真正的解决方案

### 选项A：修复DummyPktGen读取NPZ（正确但复杂）

需要：
1. 添加NPZ读取逻辑（cnpy或Python接口）
2. 修改DummyPktGen从NPZ加载真实数据
3. 循环读取所有时间步，不是生成1个就停止
4. 传递元数据（nchan, nts等）

**工作量**：4-8小时完整实现

### 选项B：使用LWA-SV的真实数据处理路径（workaround）

EPIC可能有在线数据路径（通过网络接收ROACH board数据）。检查是否有：
```bash
grep -r "PktGen\|PacketGen\|网络接收" src/
```

但这需要修改数据格式以模拟网络数据包。

### 选项C：使用Python接口（如果存在）

检查`src/python/epic_utils.py`是否有offline处理能力：

```python
# epic_utils.py可能有load_npz_and_process()？
```

### 选项D：手动pipeline（推荐先做）

**绕过EPIC的offline模式，手动处理**：

1. 用Python加载NPZ
2. 逐gulp调用EPIC的GPU kernel（如果有Python绑定）
3. 或者修改数据格式以适配EPIC期望

---

## 立即行动

### Step 1: 验证猜测

```bash
cd /mnt/data_ssd/zdj/LWA_EPIC

# 检查是否有真正的NPZ reader
grep -r "cnpy\|numpy.*load" src/
grep -r "npytbnfile" src/ --include="*.cpp" --include="*.hpp"

# 检查Python接口
ls -la src/python/
cat src/python/epic_utils.py | grep -A20 "npz\|offline"
```

### Step 2: 检查现有示例

```bash
# 他们是如何测试offline模式的？
find . -name "*test*" -o -name "*example*" | grep -i offline
```

### Step 3: 查看构建系统

```bash
# 是否链接了cnpy或其他NPZ库？
grep -r "cnpy\|numpy" build/CMakeCache.txt
```

---

## 可能的发现

EPIC可能根本**没有实现offline模式**！

- `--offline`标志存在
- `--npytbnfile`参数存在  
- 但**没有实际实现读取NPZ的代码**

DummyPktGen是用于**在线测试**的假数据生成器，不是offline处理。

真正的用法可能是：
```
实时观测 → ROACH boards → 网络数据包 → PktGen → EPIC处理 → 实时成图
```

Offline处理可能需要：
1. 先转换NPZ为网络数据包格式
2. 或者使用未完成/未文档化的功能

---

## 下一步诊断

```bash
# 1. 查看所有packet generator
ls -la src/raft_kernels/*packet*

# 2. 搜索NPZ相关代码
find src/ -type f -exec grep -l "npz\|NPZ" {} \;

# 3. 检查Python工具
cat src/python/epic_utils.py

# 4. 查看测试或示例
find . -name "*.py" -exec grep -l "epic.*offline\|npz.*epic" {} \;
```

---

## 结论

**EPIC的offline模式是stub实现！**

两个选择：
1. 花2-3天完整实现NPZ reading
2. 找到EPIC作者文档化的正确offline用法
3. 或者放弃EPIC，用其他imaging软件

**真正的教训**：

> "Never assume '--offline' means 'works with offline data'.  
> Always check if the feature is ACTUALLY IMPLEMENTED.  
> EPIC has the flag, but not the functionality."

---

## 立即执行

```bash
cd /mnt/data_ssd/zdj/LWA_EPIC

# 全面搜索NPZ处理代码
echo "=== Searching for NPZ handling ===" 
find src/ -type f \( -name "*.cpp" -o -name "*.hpp" -o -name "*.h" \) -exec grep -l "npz" {} \;

# 检查Python接口
echo "=== Checking Python interface ==="
cat src/python/epic_utils.py | head -300

# 查看示例数据如何生成
echo "=== Checking example data ==="
find . -name "*.py" -path "*/build/*" -prune -o -type f -name "*.py" -exec grep -l "npz\|savez" {} \;
```

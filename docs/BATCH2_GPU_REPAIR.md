# 第二批：双 RTX4090 数值修复与可运行入口

本批在保留用户原有改造的基础上修复核心执行问题，并新增适合当前小阵列的 FX 后端。源码先在独立 worktree 修改，按修改前 SHA256 校验后应用到 `/mnt/data_ssd/zdj/LWA_EPIC`。旧观测结果和旧 `build/epic++` 保留。

## 已修复

- **同步与越界**：shared grid 清零、上下半网格交换加入必要 barrier；Y 方向与 X 一样检查边界；GCF 使用完整频道索引，修正多 stream 的频道串用。
- **输入生命周期**：每个 stream 上传自身实际处理的频道；非最后一 gulp 也等待上传/计算完成，避免输入缓冲过早被复用。
- **自相关**：默认保留自相关；`--remove_autocorr` 显式启用。预计算每根天线实际 gridding footprint 的 Fourier 功率响应，按增益正确扣除 XX、YY、ReXY、ImXY。扣除路径使用 float FFT，避免 half FFT 残差掩盖小信号；普通路径保留 packed half FFT。
- **GCF**：所有线程统一参加 block barrier；16 个 antenna slots 与不同 support 下均验证。拒绝 oversampled support 大于 32，避免零线程 launch。
- **双卡分工**：离线 `--channels` 表示总频道数，按 GPU slot 切成相邻、互不重叠的频段，GPU IDs 可倒序指定。每卡频道数必须可被 bin 和 stream 数整除。
- **离线输入与时间**：`--npytbnfile` 真正接通且优先于环境变量；UTC 取 NPZ `meta[0]`，不再拼接固定日期/虚构序号；每序列固定 40 μs。未知/零 UTC 在 MOFF 中明确失败，工程相对时间可使用 FX NPZ 输出。
- **积累和内存**：单 NPZ 输出一幅完整积分图，`--seq_accum` 仅用于在线输入；离线要求 `--nimg_accum=1`。输入池从每卡约 2.88 GB 改为 2 个实际大小缓冲，频道归并池同样缩小；检查整数和容量边界。移除人为 40 ms 等待。
- **CLI/构建**：修复零 bin、零 oversampling、GPU IDs、GPU 查询失败、help、默认 metrics IDs；CUDA 编译与 device link 统一 sm_89，释放 streams、auto response 和 GCF texture。
- **FX**：持久 CUDA/cuBLAS 上下文，三维相位、任意显式 RF 频率、复增益、四个实相关量和物理天线自相关开关。分频道/时间块运行器支持双 GPU 并行、尾窗口正确归一化、拒绝覆盖输出。见 [FX 使用文档](FX_BACKEND.md)。

## 验证证据

| 验证 | 最终结果 |
|---|---|
| CPU + native FX + 双卡并行 CLI 综合回归 | 46/46 通过，无跳过 |
| 实际 MOFF CUDA 核：22 案例 × 3 次 | 66/66 通过 |
| CUDA memcheck / racecheck / synccheck | 检测到的错误均为 0 |
| GCF support 1/3/5/9/10 | 全部通过；权重和最大误差 2.24e-8 |
| 单天线 128² 自相关扣除 YY 残差 | 2.66e-3 降至 2.09e-7；未放宽容差 |
| 单卡、多 stream、双卡 NPZ→FITS | 合并像素最大差为 0；频率、UTC、时长一致 |
| 新 native FX memcheck | 0 错误，0 泄漏 |

MOFF 回归与独立 double DFT 比较，原先固定门槛为 `2e-5 + 0.0075 × 未扣除图对应平面峰值`，保留 half gridding 的精度边界；这不是所有输入均有 1e-7 精度的保证。零输入和越界零贡献为精确零。新 FX 为 FP32 PEDANTIC，禁用 TF32，使用独立电场求和 oracle 验证。

当前代表性新 FX API（合成 H5/V7、16 slots、128 通道、1000 时间样本、4096 像素）单卡调用中位数 **16.014 ms**，包括检查、增益、H2D、相关成像和 D2H；初始化 413 ms 另计，读盘/F-engine 不计。算法选择与公平比较限制见 [LBAND_ALGORITHM_CHOICE.md](LBAND_ALGORITHM_CHOICE.md)。这不是双卡实测加速比，也不是实际观测全链路吞吐保证。

原始日志、源码快照、微基准及合成 FITS 在 `/mnt/data_ssd/zdj/lwa_epic_review_20261002/`；CUDA 详细日志在 `LWA_EPIC_fixwork/repair-build/`。本次重新编译了三个业务 translation units，复用了已有第三方依赖，不声称完成 clean-room 构建。

## 当前可运行方式

小阵列优先使用 `scripts/run_lband_fx.py`，显式提供 geometry NPZ；完整参数和可复制示例见 [FX_BACKEND.md](FX_BACKEND.md)。不要把任意目标坐标、文件名或全 1 增益当作已完成定标。

旧 MOFF 管线通过独立入口调用本批验证的 executable：

```bash
cd /mnt/data_ssd/zdj/LWA_EPIC
EPIC_ARRAY_CONFIG="$PWD/antenna_mapping_9178_full16.yaml" \
EPIC_ANTENNA_COORDS="$PWD/antenna_coordinates.csv" \
bash scripts/run_moff_offline.sh \
  --npytbnfile /absolute/path/to/known_UTC_gulp.npz \
  --out_dir /absolute/path/to/new_output \
  --disable_metrics --ngpus 2 --gpu_ids 0,1 --aeff 25,25 \
  --channels 128 --nts 1000 --imagesize 64 --imageres 0.02 \
  --chan_nbin 4 --nstreams 1 --support 5 --kernel_oversample 2
```

这是接口示例；输入必须满足 `(T,C,16,2)`、整数 CNib 分量 `[-8,7]`、真实正 UTC 和 25 kHz 频道契约。`--aeff`、映射和成像网格必须符合实际仪器。启用扣除时加 `--remove_autocorr`，保持默认 FP32 积累；本批没有验证 BF16 的科学精度。

入口默认调用 `build/epic-repaired++`，使用本机 epic310 Python 环境，并仅为当前进程加载与系统编译器匹配的 libstdc++，解决现有 Conda 库缺少 `GLIBCXX_3.4.32` 的启动故障。可用 `EPIC_MOFF_BINARY`、`EPIC_PYTHON`、`EPIC_CPP_RUNTIME` 覆盖路径。它不修改 Conda 环境。旧脚本直接运行 `build/epic++` 仍会调用旧程序，应改用此入口或重新构建正常目标。

## 仍未关闭的科学与规模验证

工程路径可运行，**真实观测的标定验收尚未完成**。MOFF 的自定义阵列增益仍为 unity，其二维近似、相位中心、天空 WCS 和绝对流量归一化不能由这些回归证明正确。新 FX 支持明确三维模型，但仍需要真实天线顺序、时间对齐、RF/IQ 约定和仪器校准。

NPZ 仍会整块解压进 CPU 内存；FX 只保证 device 和输出按块有界。完整 250 MHz、F-engine、持续输入和长期稳定性仍需端到端压测。已有 FITS 和视频不会因源码修复而自动变正确。

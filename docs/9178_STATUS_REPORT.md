# 9178 L-band 阵列处理状态报告

更新时间：2026-04-26

## 1. 9178 使用的天线

9178 数据不是标准 LWA 阵列，而是当前本地 L-band 阵列。按接线记录，6 个终端与输入如下：

| 终端 | 文件前缀 | 逻辑名 | 通道 0 | 通道 1 |
| --- | --- | --- | --- | --- |
| 1 | `HBaseband0Spec0` | H00 | 北4 垂直 | 北3 垂直 |
| 2 | `HBaseband1Spec1` | H11 | 东4 水平 | 东3 垂直 |
| 3 | `HBaseband2Spec2` | H22 | 东5 水平 | 东6 水平 |
| 4 | `VBaseband0Spec0` | V00 | 东4 垂直 | 东5 垂直 |
| 5 | `VBaseband1Spec1` | V11 | 东9 水平 | 东10 垂直 |
| 6 | `VBaseband2Spec2` | V22 | 东7 水平 | 东8 垂直 |

物理天线共 10 根：

`北4、北3、东4、东3、东5、东6、东9、东10、东7、东8`

其中 `东4` 和 `东5` 是当前数据中真正同时有 H/V 的双极化天线；其他天线目前只接入一个极化。

## 2. 解交织结论

9178 数据包格式：

- 每包 `262400` 字节。
- 包头 `256` 字节。
- payload `262144` 字节。
- payload 按 `[32 bytes ch0][32 bytes ch1]` 重复交织。
- 包头标识为 `0x18EFDC01`，datatype 为 `13`。

`scripts/check_9178_interleave.py` 已对 6 个 9178 文件的两个通道首包做过验证。快速 `memmap + reshape` 解交织与 `UnpackCustom.py` 输出逐点一致：

- `equal=True`
- `max_abs_diff=0`

所以成图异常不是由解交织方式错误造成的。

## 3. EPIC 输入格式适配

9178 原始数据是 int8 时域基带，不是 EPIC 标准库可直接读取的 CHIPS/F-engine channelized CNib 输入。

当前适配流程在 `scripts/convert_9178_to_epic.py` 中完成：

1. 按终端文件和通道读取 int8 IQ。
2. 每 `40 us` 切成一个 EPIC sequence。
3. 对每个 sequence 做 FFT/channelization。
4. 取 128 个 25 kHz 频点，中心频率默认按 1355 MHz 设置。
5. 在进入 EPIC 前缩放量化为 signed 4-bit CNib。
6. 输出 EPIC offline NPZ，形状为 `(ntime, nchan, nstand, npol)`。

现在支持：

- `--array-mode epic4`：4 输入 smoke test。
- `--array-mode full10`：10 根真实天线的数据形状，但当前 EPIC CUDA 路径会触发 misaligned address。
- `--array-mode full16`：10 根真实天线 + 6 个 padding stand，是当前可运行的全量路径。
- `--start-seq`：按 40us sequence 精确切块，避免全文件分块时 packet 边界错位。
- `--cnib-scale`：对多个分块使用统一 CNib 缩放，便于分块 FITS 后续叠加。

## 4. 已修复的问题

full16 最初能运行但 FITS 全零。原因不是 padding 数据为零，而是 padding stand 的坐标曾设为 `(0, 0, 0)`；当前 `antenna_coordinates.csv` 的坐标是大数值坐标，`gen_loc_lwasv()` 做 min/max 居中时把 padding 也算入阵列跨度，导致 10 根真实天线被推到 64x64 grid 外。

修复位置：

- `src/python/epic_utils.py`

修复方式：

- padding stand 仍作为零数据输入槽位存在。
- padding stand 的坐标放在真实天线中心附近。
- padding 不再扩大阵列 min/max 范围。

修复后，full16 在 `imageres=0.02`、64x64 grid 下，真实天线像素位置约为：

- x: `11.6 .. 52.4`
- y: `18.6 .. 45.4`

全量 full16 已能生成非零 FITS。

## 5. 当前全量处理结果

9178 每个文件：

- `8184` 个 packet。
- `53634` 个完整 40us sequence。
- 可用时长约 `2145.36 ms`。

已按 6 个分块处理：

- `0 .. 9999`
- `10000 .. 19999`
- `20000 .. 29999`
- `30000 .. 39999`
- `40000 .. 49999`
- `50000 .. 53633`

统一 CNib 缩放：

- `--cnib-scale 0.0156822`

全量叠加输出：

- FITS: `results/9178_crab_full16_fullfile_fixedscale_res002/9178_crab_full16_fullfile_sum_2145ms.fits`
- PNG: `results/9178_crab_full16_fullfile_fixedscale_res002/9178_crab_full16_fullfile_sum_2145ms_XXYY_sum.png`

总 FITS 统计：

- shape: `(4, 32, 64, 64)`
- nonzero: `524288 / 524288`
- min: `-1733913344.0`
- max: `216003760128.0`
- mean: `76226692992.41559`

这些图仍是未校准图，不能直接作为科学成图结果解释。

## 6. 当前校准状态

当前 9178 自定义阵列路径中，`epic_utils.gen_phases_lwasv()` 对 custom array 返回 unity phase。

也就是说，当前只做了：

- 使用 `antenna_coordinates.csv` 作为阵列坐标。
- 使用 EPIC gridder 成图。

尚未做：

- Crab 几何延迟修正。
- 线缆/接收机延迟修正。
- 每输入复增益校准。
- bandpass 校准。
- fringe stopping。
- 真实观测 UTC 写入 FITS。

因此，当前成图异常或源位置不对，主要不是解包问题，而是校准和时间模型尚未补齐。

## 7. 9178 应补充的校准流程

要把当前结果变成科学可用图，建议按这个顺序补：

1. 确认观测 UTC。
   文件名有 `26-01-11_19-49-55`，但 packet header 的时间字段需要确认编码和时区。没有真实 UTC，无法正确计算 Crab 几何相位。

2. 确认坐标系。
   `antenna_coordinates.csv` 当前作为坐标基准使用，但仍需明确它是 ENU、本地平面坐标还是绝对坐标，以及单位和原点。

3. 计算 Crab 几何延迟。
   用 Crab 的 J2000 坐标和观测 UTC，计算每根天线相对参考天线的 topocentric delay，在 channelization 后、CNib 量化前乘掉相位。

4. 拟合残余仪器延迟。
   对 Crab 长积分数据做 baseline phase vs frequency 拟合，求每输入相对参考输入的 residual delay。

5. 求每输入复增益和 bandpass。
   在几何延迟和 residual delay 修正后，解每路输入的幅度和相位频率响应。

6. 固化校准表。
   推荐表字段：
   `antenna, pol, delay_ns, phase0_rad, gain_amp, bandpass_re, bandpass_im`

7. 在 `convert_9178_to_epic.py` 中增加 `--calibration`，在 CNib 量化前应用校准。

## 8. 当前应保留的文件

核心脚本：

- `scripts/convert_9178_to_epic.py`
- `scripts/check_9178_interleave.py`
- `scripts/combine_9178_fits.py`

核心配置：

- `antenna_coordinates.csv`
- `antenna_mapping_9178.yaml`
- `antenna_mapping_9178_epic4.yaml`
- `antenna_mapping_9178_full16.yaml`

核心结果：

- `data/epic_ready/9178_crab_full16_*fixedscale.npz`
- `results/9178_crab_full16_*fixedscale_res002/`
- `results/9178_crab_full16_fullfile_fixedscale_res002/`

参考脚本：

- `main_my_base2fil_for2_npol_nfile_pyfftw_hightime2_8bits_argv_norm_DC_argv_freq_size.py`

9197/LWA 相关脚本目前建议先保留为 reference，不直接删除。等 9178 校准表和全流程稳定后，再统一移动到 `legacy/`。


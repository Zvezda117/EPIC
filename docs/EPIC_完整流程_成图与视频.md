# EPIC 从 CMake 到成图与视频（完整数据）

目标：使用 EPIC 从源码构建开始，到输出**利用完整数据的图片和视频**（1024 通道、全 250 MHz 带宽、中心 1355 MHz）。

---

## 0. 前提

- 9197 原始数据目录存在，例如：`/mnt/DATA/存储阵列/9197_Base2_ADC16_26-01-18_19-16-51/`
- 环境：CUDA、Python 3（numpy、matplotlib、PIL、tqdm 等），可选 conda 环境 `epic310`
- 若用 EPIC C++ 成图：需 NVIDIA GPU（支持 CUDA）、CMake ≥ 3.10

---

## 1. 构建 EPIC（CMake → 可执行文件）

在项目根目录执行：

```bash
cd /mnt/data_ssd/zdj/LWA_EPIC
mkdir -p build && cd build
cmake ..
make -j$(nproc)
```

生成的可执行文件：`build/epic++`。  
若需在别处运行，请从**项目根目录**进入 `build` 再执行 `./epic++`（Python 与 `data/` 路径相对项目根）。

---

## 2. 准备 EPIC 用 npz（1s、1024 通道、全 250 MHz）

EPIC 离线模式从 **Python 的 `get_40ms_gulp()`** 读入 npz，该函数会按固定顺序查找 npz 路径（如 `data/...`、`build/data/...`）。因此需要先把「1s、1024 通道、全带宽、校准后」的 npz 生成好并放到 EPIC 会读的位置。

在项目根目录：

```bash
cd /mnt/data_ssd/zdj/LWA_EPIC
source /opt/miniconda3/bin/activate epic310   # 或你的 Python 环境
# 指定输出路径；不写则默认 data/epic_ready/9197_1s_calib.npz
python3 scripts/convert_9197_1s_epic.py data/epic_ready/9197_1s_1024ch.npz
```

该脚本会：

- 从 9197 某个 `.data` 文件读取 **1s** 数据（25 × 40 ms），
- 做 **CasA 几何时延校准** 和 **DC mask**，
- 输出 **1024 通道、全 250 MHz** 的 npz，shape 为 `(25, 1024, 4, 2)`，
- 若存在 `build/data`，会**自动复制**到 `build/data/40ms_128chan_600offset_gulp_c64_virtransit.npz`（EPIC 默认读取的文件名）。

这样 EPIC 在 `build` 下运行时会自动用到这份 1024 通道、全带宽的 npz。

---

## 3. 用 EPIC 成图（单帧/多帧，基于上述 npz）

在 **build** 目录下运行（使用完整数据的 1s 段 → 一张或数张图）：

```bash
cd /mnt/data_ssd/zdj/LWA_EPIC/build
export MKL_THREADING_LAYER=GNU
./epic++ --offline \
  --out_dir ../results/epic_full \
  --disable_metrics \
  --imagesize 64 \
  --channels 1024 \
  --chan_nbin 1 \
  --seq_accum 40 \
  --nts 1000 \
  --imageres 1.056 \
  --accum_16bit \
  --nstreams 8 \
  --support 5 \
  --aeff 8 \
  --kernel_oversample 4 \
  --ngpus 1 \
  --gpu_ids 0
```

说明：

- `--offline`：使用上面准备的 npz（`get_40ms_gulp()` 会读到 1024 通道、1s 数据）。
- `--channels 1024`：与 npz 的 1024 通道一致。
- 离线模式下 EPIC 会按内部逻辑处理多个 gulp（例如 3 个），每个 gulp 对应一帧，输出到 `--out_dir`（FITS/PNG 等，取决于 EPIC 配置）。

**输出**：`results/epic_full/` 下的单帧或多帧图像，对应「完整 1s 数据、1024 通道、全 250 MHz」的成像结果。

---

## 4. 完整数据 → 多帧图 + 视频（Python 脚本，推荐）

EPIC 离线模式**每次运行只处理有限个 gulp**（例如 3 个），无法直接对「每个文件整段 4s+、多文件」逐帧成图再拼成视频。  
要对**完整数据**做多帧成图并合成视频，需用 Python 脚本（同一套校准与 1024 通道、250 MHz 设置）：

```bash
cd /mnt/data_ssd/zdj/LWA_EPIC
python3 scripts/convert_9197_multi_full_video.py 0
```

- 单文件示例：`0` 表示使用第 0 号 `.data` 文件，**整文件 ~4.28s** 不截断。
- 多文件示例（多段完整数据拼成更长视频）：
  ```bash
  python3 scripts/convert_9197_multi_full_video.py 0 1 2
  ```

该脚本会：

- 读取每个文件的**全部采样**（完整数据），
- **CasA 几何时延校准** + DC mask，
- **1024 通道、全 250 MHz**，40 ms/帧，
- 逐帧做脏图并写出帧图，再合成 **GIF 视频**。

**输出**：

- 帧图：`results/multi_full_video/frames/`
- 视频：`results/multi_full_video/9197_multi_full_calib.gif`

若已安装 ffmpeg，可再生成 MP4（脚本末尾会提示命令，例如）：

```bash
ffmpeg -framerate 25 -i results/multi_full_video/frames/frame_%05d.png \
  -c:v libx264 -pix_fmt yuv420p results/multi_full_video/9197_multi_full_calib.mp4
```

这样得到的是**真正完整数据**下的多帧图与视频。

---

## 4B. Python 只做「转换 + 拼接」，EPIC 多次成图出连续视频（推荐）

若希望 **Python 仅负责数据转换和视频拼接，所有成图由 EPIC 完成**，可以**多次调用 EPIC**，每帧一次（**每帧 1s 积分**），再拼接成视频。EPIC 侧读取 npz 的入口仍叫 `get_40ms_gulp()`（历史命名），实际支持任意 nseqs，当前约定每帧 **1s = 25×40ms**，shape `(25, 1024, 4, 2)`。

### 步骤 1：Python 仅做数据转换（不成图）

将 9197 完整数据转为「每 1s 一帧」的 npz，每帧 shape `(25, 1024, 4, 2)`（25 个 40ms sequence），供 EPIC 单次调用读入：

```bash
cd /mnt/data_ssd/zdj/LWA_EPIC
python3 scripts/convert_9197_to_gulp_npz.py 0
# 多文件：python3 scripts/convert_9197_to_gulp_npz.py 0 1 2
```

- 输出目录默认：`data/epic_ready/gulps_9197/`
- 文件命名：`frame_00000.npz`, `frame_00001.npz`, ...（每文件 4.28s ≈ 4 帧/文件，1s/帧）

### 步骤 2：多次调用 EPIC 成图（串行或并行），收集每帧

- **串行**（每帧依次跑 EPIC，每帧用环境变量 `EPIC_GULP_NPZ` 指向对应 npz、独立 `--out_dir`）：
  ```bash
  chmod +x scripts/run_epic_video.sh
  ./scripts/run_epic_video.sh
  ```
- **并行**（多进程同时跑多帧，提升效率；单 GPU 时建议 `--jobs 1` 或 `2` 以免 OOM）：
  ```bash
  python3 scripts/run_epic_video_parallel.py --jobs 4
  ```

两种方式都会把每帧的 `*_QL.png` 收为 `results/epic_video_frames/frame_00000.png`, `frame_00001.png`, ...；并行版通过 `EPIC_GULP_NPZ` 与每帧独立 `--out_dir` 避免冲突。

### 步骤 3：Python 仅做视频拼接（不成图、不读原始数据）

```bash
python3 scripts/stitch_video.py results/epic_video_frames results/epic_video_frames/9197_epic_calib.gif
```

- `stitch_video.py` 只读 `frame_*.png`，按序号排序后生成 GIF（或指定 `.mp4` 且本机有 ffmpeg 时生成 MP4）。串行/并行脚本在收完帧后会自动调用本步。

**小结**：Python 只做「转换数据 → 1s 帧 npz」和「帧图 → 视频」；成图全部由**多次调用 EPIC** 完成；每帧 **1s** 积分；可并行执行以提升效率。

---

## 5. 可选：完整数据积分成一张图（长时间积分）

若希望用「整段 4s+ 数据」积分成**一张**高 SNR 图（而不用 EPIC 多帧），可用长时间积分脚本。需要先有「多 gulp、1024 通道」的 npz（可由 `convert_9197_multi_full_video.py` 的中间逻辑或类似脚本写出），再调用：

```bash
python3 scripts/image_9197_long_integration.py
```

该脚本会对多帧可见度做 incoherent 平均后成图，输出到 `results/long_integration/`。若当前仓库里该脚本仍读 128 通道 npz，需把其输入改为 1024 通道、全带宽的 npz 以与当前设置一致。

---

## 6. 流程小结（完整数据 → 图 + 视频）

| 步骤 | 做什么 | 输出 |
|------|--------|------|
| 1 | `mkdir build && cd build && cmake .. && make -j` | `build/epic++` |
| 2 | `python3 scripts/convert_9197_1s_epic.py data/epic_ready/9197_1s_1024ch.npz` | 1s、1024ch、250MHz npz，并复制到 `build/data/` |
| 3 | `cd build && ./epic++ --offline ... --channels 1024` | 基于 1s 完整段的单/多帧图 → `results/epic_full/` |
| 4 | `python3 scripts/convert_9197_multi_full_video.py 0 [1 2 ...]` | 完整数据多帧图 + GIF 视频 → `results/multi_full_video/`（Python 成图） |
| **4B** | **Python 只做转换+拼接、EPIC 多次成图**：`convert_9197_to_gulp_npz.py` → `run_epic_video.sh` → `stitch_video.py` | gulp npz → EPIC 逐帧成图 → 帧图 + 视频 → `results/epic_video_frames/` |
| 5（可选） | `python3 scripts/image_9197_long_integration.py` | 长时间积分单张图 → `results/long_integration/` |

**「完整数据」的含义**：

- **成图（EPIC）**：使用 1s 完整段、1024 通道、全 250 MHz 的 npz，EPIC 输出对应这一段的图像。
- **视频与多帧图**：由 `convert_9197_multi_full_video.py` 处理每个文件的**整段 4.28s**（及多文件），不截断，得到完整数据下的帧序列与视频。

数据与带宽约定：250 MHz 带宽、中心 1355 MHz、I/Q 复数 250 Msps；校准为 CasA 几何时延 + DC mask；通道数 1024，全基带 0~250 MHz。

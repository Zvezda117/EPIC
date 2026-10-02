#!/bin/bash
# 多次调用 EPIC 成图（串行），再拼接视频。
# 前置：已用 convert_9197_to_gulp_npz.py 生成 frame_00000.npz, frame_00001.npz, ...（每帧 1s）
# 并行版请用: python3 scripts/run_epic_video_parallel.py [--jobs 4]
set -e
cd "$(dirname "$0")/.."
PROJECT_ROOT="$PWD"

GULP_DIR="${GULP_NPZ_DIR:-data/epic_ready/gulps_9197}"
FRAMES_DIR="${FRAMES_DIR:-results/epic_video_frames}"
TMP_OUT="${TMP_OUT:-results/epic_video_tmp}"
VIDEO_OUT="${VIDEO_OUT:-results/epic_video_frames/9197_epic_calib.gif}"

# 优先 frame_*.npz（1s/帧），兼容旧 gulp_*.npz
NPZ_PATTERN="$GULP_DIR/frame_*.npz"
if ! compgen -G "$NPZ_PATTERN" > /dev/null 2>&1; then
  NPZ_PATTERN="$GULP_DIR/gulp_*.npz"
fi

if [ ! -d "$GULP_DIR" ]; then
  echo "错误: 未找到目录 $GULP_DIR"
  echo "请先运行: python3 scripts/convert_9197_to_gulp_npz.py 0 [1 2 ...]"
  exit 1
fi

N=$(ls -1 $NPZ_PATTERN 2>/dev/null | sort -V | wc -l)
if [ "$N" -eq 0 ]; then
  echo "错误: $GULP_DIR 下没有 frame_*.npz 或 gulp_*.npz"
  exit 1
fi

mkdir -p "$FRAMES_DIR"
mkdir -p "$TMP_OUT"

echo "=============================================="
echo "  多次调用 EPIC 成图（串行）→ 收集帧 → 拼接视频"
echo "=============================================="
echo "  帧 npz 目录: $GULP_DIR  共 $N 帧 (1s/帧)"
echo "  帧输出:     $FRAMES_DIR"
echo "  视频:       $VIDEO_OUT"
echo "  并行版:     python3 scripts/run_epic_video_parallel.py --jobs 4"
echo "=============================================="

export MKL_THREADING_LAYER=GNU
export EPIC_ARRAY_CONFIG="$PROJECT_ROOT/antenna_mapping_9197.yaml"
export EPIC_ANTENNA_COORDS="$PROJECT_ROOT/antenna_coordinates.csv"
export EPIC_CENTER_FREQ_MHZ=1355

i=0
for npz in $(ls -1 $NPZ_PATTERN 2>/dev/null | sort -V); do
  out_frame_dir="$TMP_OUT/frame_$(printf '%05d' "$i")"
  mkdir -p "$out_frame_dir"
  # 使用绝对路径，以便 EPIC 从 build/ 运行时 Python 能正确读 npz
  case "$npz" in /*) EPIC_GULP_NPZ="$npz" ;; *) EPIC_GULP_NPZ="$PROJECT_ROOT/$npz" ;; esac
  export EPIC_GULP_NPZ
  cd "$PROJECT_ROOT/build"
  ./epic++ --offline --out_dir "$PROJECT_ROOT/$out_frame_dir" --disable_metrics \
    --imagesize 64 --channels 1024 --chan_nbin 8 --seq_accum 40 --nts 1000 \
    --imageres 0.05 --accum_16bit --nstreams 1 --support 2 --aeff 8 \
    --kernel_oversample 2 --ngpus 1 --gpu_ids 0 > /dev/null 2>&1 || true
  cd "$PROJECT_ROOT"
  latest=$(ls -t "$out_frame_dir"/*_QL.png 2>/dev/null | head -1)
  if [ -n "$latest" ]; then
    cp -f "$latest" "$FRAMES_DIR/frame_$(printf '%05d' "$i").png"
  fi
  printf "\r  EPIC 成图 %d / %d" "$((i + 1))" "$N"
  i=$((i + 1))
done
echo ""

echo ""
echo "【拼接视频】"
python3 scripts/stitch_video.py "$FRAMES_DIR" "$VIDEO_OUT"

echo ""
echo "完成。视频: $VIDEO_OUT  帧: $FRAMES_DIR"

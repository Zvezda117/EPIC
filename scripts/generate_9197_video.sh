#!/bin/bash
# 9197视频生成脚本
# 使用完整4秒数据，106帧，40ms/帧

set -e

cd /mnt/data_ssd/zdj/LWA_EPIC

# 创建输出目录
mkdir -p results/video_frames

# 源文件
SOURCE_NPZ="data/epic_ready/9197_full_106gulps_128chan_dcmask.npz"
EPIC_INPUT="build/data/40ms_128chan_600offset_gulp_c64_virtransit.npz"

echo "=================================================="
echo "  9197 视频生成 (106帧, 40ms/帧)"
echo "=================================================="
echo ""
echo "源文件: $SOURCE_NPZ"
echo "输出目录: results/video_frames"
echo ""

# 复制数据文件
echo "[1/3] 复制数据到EPIC输入路径..."
cp "$SOURCE_NPZ" "$EPIC_INPUT"

# 生成图像（一次处理所有gulps）
echo "[2/3] 运行EPIC生成所有帧..."
cd build
./epic++ --offline \
    --out_dir ../results/video_frames \
    --disable_metrics \
    --imagesize 128 \
    --channels 128 \
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
    --gpu_ids=0

cd ..

# 检查生成的帧数
echo ""
echo "[3/3] 检查生成的帧..."
FRAME_COUNT=$(ls -1 results/video_frames/*.png 2>/dev/null | wc -l)
echo "生成的PNG帧数: $FRAME_COUNT"

if [ "$FRAME_COUNT" -gt 0 ]; then
    echo ""
    echo "生成视频..."
    ffmpeg -y -framerate 10 -pattern_type glob \
        -i 'results/video_frames/*.png' \
        -c:v libx264 -pix_fmt yuv420p \
        results/9197_video.mp4
    
    echo ""
    echo "=================================================="
    echo "  完成！"
    echo "  视频: results/9197_video.mp4"
    echo "=================================================="
else
    echo "没有生成帧，检查EPIC输出..."
    ls -la results/video_frames/
fi

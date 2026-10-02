#!/bin/bash
# 批量处理9197数据（分批1000步）

set -e

cd /mnt/data_ssd/zdj/LWA_EPIC
source /opt/miniconda3/bin/activate epic310

echo "══════════════════════════════════════════════"
echo "  9197数据批量成图"
echo "══════════════════════════════════════════════"
echo ""

# 参数
TOTAL_STEPS=42865
BATCH_SIZE=1000
NBATCHES=$((TOTAL_STEPS / BATCH_SIZE + 1))

echo "总时间步: $TOTAL_STEPS"
echo "批次大小: $BATCH_SIZE"
echo "批次数量: $NBATCHES"
echo ""

for batch in $(seq 0 $((NBATCHES-1))); do
    START=$((batch * BATCH_SIZE))
    END=$(( (batch+1) * BATCH_SIZE ))
    if [ $END -gt $TOTAL_STEPS ]; then
        END=$TOTAL_STEPS
    fi
    
    NSTEPS=$((END - START))
    
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "批次 $((batch+1))/$NBATCHES: 步骤 $START-$END ($NSTEPS步)"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    
    # 生成这批数据
    python3 << PYEOF
import numpy as np

print("加载时域数据...")
time_data = np.load('data/processed/9197_file0_complete.npz')
voltages = time_data['voltages']

# 提取这批
start_sample = $START * 25000
end_sample = $END * 25000
voltages_batch = voltages[start_sample:end_sample, 0, :, 0]

print(f"FFT转换 {$NSTEPS} 步...")
fft_result = np.zeros(($NSTEPS, 128, 4), dtype=np.complex64)
voltages_reshaped = voltages_batch.reshape($NSTEPS, 25000, 4)

for itime in range($NSTEPS):
    for iant in range(4):
        fft = np.fft.fft(voltages_reshaped[itime, :, iant])
        fft_result[itime, :, iant] = fft[:128]

print("填充到256天线...")
epic_data = np.zeros(($NSTEPS, 128, 256, 2), dtype=np.complex64)
for iant in range(4):
    epic_data[:, :, iant, 0] = fft_result[:, :, iant]
    epic_data[:, :, iant, 1] = fft_result[:, :, iant]

meta = np.array([$NSTEPS, 0], dtype=np.int64)

print("保存...")
np.savez_compressed('build/data/40ms_128chan_600offset_gulp_c64_virtransit.npz',
                   data=epic_data, meta=meta)
print("✓ 完成")
PYEOF

    # 运行EPIC
    cd build
    ./epic++ --offline \
        --out_dir ../results/images_batch${batch} \
        --disable_metrics \
        --imagesize 128 \
        --channels 128 \
        --chan_nbin 1 \
        --seq_accum 40 \
        --nts $NSTEPS \
        --imageres 1.056 \
        --accum_16bit \
        --nstreams 8 \
        --support 5 \
        --aeff 25 \
        --kernel_oversample 4 \
        --ngpus 1 \
        --gpu_ids=0
    
    cd ..
    
    echo "✓ 批次 $((batch+1)) 完成"
    ls -lh results/images_batch${batch}/EPIC_*.fits
    echo ""
done

echo "══════════════════════════════════════════════"
echo "  全部完成！"
echo "══════════════════════════════════════════════"
echo ""
echo "合并所有批次图像（待实现）"
ls -lh results/images_batch*/EPIC_*.fits

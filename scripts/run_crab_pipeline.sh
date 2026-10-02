#!/bin/bash
# Crab数据处理完整流程
# 作者：基于Linus风格的实用主义设计

set -e  # 遇到错误立即退出

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo_error() { echo -e "${RED}[ERROR]${NC} $1"; }
echo_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
echo_info() { echo -e "${YELLOW}[INFO]${NC} $1"; }

# 配置
DATA_DIR="/mnt/data_ssd/zdj/data/crab"
WORK_DIR="/mnt/data_ssd/zdj/LWA_EPIC"
OUTPUT_DIR="${WORK_DIR}/crab_output"

cd "$WORK_DIR" || { echo_error "Cannot cd to $WORK_DIR"; exit 1; }

echo "=================================================="
echo "  Crab Data Processing Pipeline"
echo "=================================================="
echo ""

# ============================================================
# 步骤1：验证接线（必须先做）
# ============================================================
echo_info "Step 1: Verifying antenna wiring..."
echo "This will analyze 100 packets from each file to check consistency"
echo ""

if [ ! -f "verify_antenna_wiring.py" ]; then
    echo_error "verify_antenna_wiring.py not found!"
    exit 1
fi

python3 verify_antenna_wiring.py \
    --data-dir "$DATA_DIR" \
    --npackets 100 \
    --check-consistency | tee verify_wiring.log

echo ""
read -p "请检查上面的结果。如果相关性异常，请按Ctrl+C退出并修正接线图。否则按Enter继续..." dummy

# ============================================================
# 步骤2：处理测试数据（快速验证）
# ============================================================
echo ""
echo_info "Step 2: Processing test data (first 100 packets)..."

python3 process_crab_data.py \
    "$DATA_DIR" \
    -o "${OUTPUT_DIR}/crab_test_100packs.npz" \
    --pack-end 100 || {
    echo_error "Test processing failed!"
    exit 1
}

# 验证输出
python3 << 'EOF'
import numpy as np
import sys

try:
    data = np.load('crab_output/crab_test_100packs.npz', allow_pickle=True)
    print("\n=== Test Output Verification ===")
    print("Keys:", list(data.keys()))
    print("Voltages shape:", data['voltages'].shape)
    print("Expected: (nsamples, 1, nstands, 2)")
    
    v = data['voltages']
    print(f"\nData range:")
    print(f"  Real: [{v.real.min():.2e}, {v.real.max():.2e}]")
    print(f"  Imag: [{v.imag.min():.2e}, {v.imag.max():.2e}]")
    print(f"  Power: [{np.abs(v).min():.2e}, {np.abs(v).max():.2e}]")
    
    # 检查是否有数据
    if np.abs(v).max() < 1e-10:
        print("\n⚠️  WARNING: All voltages near zero!")
        sys.exit(1)
    
    print("\n✓ Test data looks good!")
except Exception as e:
    print(f"\n✗ Error: {e}")
    sys.exit(1)
EOF

if [ $? -ne 0 ]; then
    echo_error "Test data verification failed!"
    exit 1
fi

echo_success "Test processing complete!"

# ============================================================
# 步骤3：询问是否处理全量数据
# ============================================================
echo ""
echo_info "Step 3: Process full dataset?"
echo "This will take ~20-30 minutes and use ~500MB disk space"
echo ""
read -p "Process full data? (y/N): " answer

if [[ "$answer" =~ ^[Yy]$ ]]; then
    echo_info "Processing full dataset..."
    
    python3 process_crab_data.py \
        "$DATA_DIR" \
        -o "${OUTPUT_DIR}/crab_full_data.npz" || {
        echo_error "Full processing failed!"
        exit 1
    }
    
    echo_success "Full data processing complete!"
else
    echo_info "Skipping full processing. Using test data for now."
fi

# ============================================================
# 步骤4：生成EPIC输入文件
# ============================================================
echo ""
echo_info "Step 4: Generating EPIC-compatible input..."

# 选择使用哪个数据文件
if [ -f "${OUTPUT_DIR}/crab_full_data.npz" ]; then
    INPUT_DATA="${OUTPUT_DIR}/crab_full_data.npz"
else
    INPUT_DATA="${OUTPUT_DIR}/crab_test_100packs.npz"
fi

echo "Using: $INPUT_DATA"

# 转换为EPIC离线格式
python3 << EOF
import numpy as np

# 读取处理后的数据
data = np.load('$INPUT_DATA', allow_pickle=True)

# EPIC离线模式期望的格式
# 参考：build/data/40ms_128chan_600offset_gulp_c64_virtransit.npz

voltages = data['voltages']
positions = data['antenna_positions']
phases = data['phases']

print(f"Converting to EPIC format...")
print(f"  Input shape: {voltages.shape}")

# EPIC格式：可能需要重塑
# 检查build/data下的示例文件格式
epic_data = {
    'voltages': voltages,
    'antenna_positions': positions,
    'phases': phases,
    'freq': ${data['config']['observation']['center_freq_mhz']},
    'bandwidth': ${data['config']['observation']['bandwidth_mhz']}
}

output_file = '${OUTPUT_DIR}/crab_epic_input.npz'
np.savez_compressed(output_file, **epic_data)
print(f"Saved to: {output_file}")
print(f"File size: {os.path.getsize(output_file) / 1e6:.2f} MB")
EOF

echo_success "EPIC input file generated!"

# ============================================================
# 步骤5：运行EPIC（如果已编译）
# ============================================================
echo ""
echo_info "Step 5: Running EPIC imager..."

if [ ! -f "./build/epic++" ]; then
    echo_error "EPIC not built yet!"
    echo "Please run:"
    echo "  cd build && cmake .. && make"
    exit 1
fi

echo "Running EPIC in offline mode..."

mkdir -p "${OUTPUT_DIR}/images"

./build/epic++ \
    --offline \
    --out_dir "${OUTPUT_DIR}/images" \
    --disable_metrics \
    --imagesize 64 \
    --channels 64 \
    --chan_nbin 1 \
    --seq_accum 40 \
    --nts 1000 \
    --imageres 2.0 \
    --support 3 \
    --aeff 12.5 \
    --kernel_oversample 2 \
    --nstreams 4 \
    --ngpus 1 \
    --gpu_ids=0 || {
    echo_error "EPIC failed! Check logs above."
    exit 1
}

echo_success "EPIC processing complete!"

# ============================================================
# 总结
# ============================================================
echo ""
echo "=================================================="
echo "  Pipeline Complete!"
echo "=================================================="
echo ""
echo "Generated files:"
echo "  - Wiring verification: verify_wiring.log"
echo "  - Test data: ${OUTPUT_DIR}/crab_test_100packs.npz"
if [ -f "${OUTPUT_DIR}/crab_full_data.npz" ]; then
    echo "  - Full data: ${OUTPUT_DIR}/crab_full_data.npz"
fi
echo "  - EPIC input: ${OUTPUT_DIR}/crab_epic_input.npz"
echo "  - Images: ${OUTPUT_DIR}/images/*.fits"
echo ""
echo "Next steps:"
echo "  1. Check images with ds9: ds9 ${OUTPUT_DIR}/images/*.fits"
echo "  2. If images look bad, adjust imaging parameters"
echo "  3. Compare with known Crab position: RA=05:34:30, Dec=+22:00:52"
echo ""

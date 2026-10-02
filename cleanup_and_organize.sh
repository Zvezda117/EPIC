#!/bin/bash
# 项目清理和整理脚本

echo "============================================"
echo "  LWA_EPIC 项目整理"
echo "============================================"
echo ""

# 1. 创建目录结构
echo "1. 创建标准目录结构..."
mkdir -p {data/{raw,processed,epic_ready},results/{images,logs},scripts,docs}

# 2. 清理Python缓存
echo "2. 清理Python缓存..."
rm -rf __pycache__
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
echo "   ✓ 已删除 __pycache__"

# 3. 整理数据文件
echo "3. 整理数据文件..."

# 删除测试文件
rm -f test_9197_file0.npz
echo "   ✓ 删除测试文件: test_9197_file0.npz"

# 删除旧版本（128通道）
rm -f 9197_epic_ready.npz
echo "   ✓ 删除旧版本: 9197_epic_ready.npz (128通道)"

# 移动数据文件到对应目录
if [ -f "9197_file0_complete.npz" ]; then
    mv 9197_file0_complete.npz data/processed/
    echo "   ✓ 移动: 9197_file0_complete.npz → data/processed/"
fi

if [ -f "9197_epic_512chan.npz" ]; then
    mv 9197_epic_512chan.npz data/epic_ready/
    echo "   ✓ 移动: 9197_epic_512chan.npz → data/epic_ready/"
fi

# 4. 整理日志文件
echo "4. 整理日志文件..."
mv *.log results/logs/ 2>/dev/null && echo "   ✓ 移动所有 .log 文件" || echo "   - 无日志文件"

# 5. 整理脚本文件
echo "5. 整理脚本文件..."
for script in process_*.py verify_*.py convert_*.py run_*.sh; do
    if [ -f "$script" ]; then
        mv "$script" scripts/
        echo "   ✓ 移动: $script → scripts/"
    fi
done

# 6. 整理文档
echo "6. 整理文档..."
mv *.md docs/ 2>/dev/null && echo "   ✓ 移动所有 .md 文件" || echo "   - 无文档文件"

# 7. 整理配置文件
echo "7. 保留配置文件在根目录..."
# antenna_mapping*.yaml 保留在根目录
ls -1 antenna_mapping*.yaml 2>/dev/null | while read f; do
    echo "   ✓ 保留: $f"
done

# 8. 清理build目录的临时文件
echo "8. 清理build目录..."
if [ -d "build" ]; then
    rm -f build/*.log 2>/dev/null
    echo "   ✓ 清理 build/*.log"
fi

# 9. 显示最终结构
echo ""
echo "============================================"
echo "  清理完成！当前结构："
echo "============================================"
tree -L 2 -h --du 2>/dev/null || ls -lh

echo ""
echo "磁盘使用："
du -sh data/* results/* 2>/dev/null

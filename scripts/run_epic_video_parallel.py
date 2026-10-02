#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
并行多次调用 EPIC 成图，再拼接视频。

- 每帧 npz 为 1s（25×40ms），shape (25, 1024, 4, 2)
- 通过环境变量 EPIC_GULP_NPZ 指定每帧 npz，避免多进程写同一路径
- 每帧使用独立 --out_dir，再收集 *_QL.png 为 frame_XXXXX.png
- 并行度默认 4；单 GPU 时建议 --jobs 1 或 2 以免 OOM
"""
import os
import sys
import glob
import subprocess
import shutil
from pathlib import Path
from multiprocessing import Pool

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BUILD_DIR = PROJECT_ROOT / "build"
EPIC_BIN = BUILD_DIR / "epic++"


def run_one_frame(args):
    """处理单帧：设置 EPIC_GULP_NPZ，运行 EPIC，收集 *_QL.png 到 frame_XXXXX.png"""
    frame_idx, npz_path, frames_dir, tmp_base, env_extra = args
    npz_path = Path(npz_path)
    frames_dir = Path(frames_dir)
    tmp_base = Path(tmp_base)
    out_frame_dir = tmp_base / f"frame_{frame_idx:05d}"
    out_frame_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["EPIC_GULP_NPZ"] = str(npz_path.resolve())
    env["MKL_THREADING_LAYER"] = "GNU"
    env["EPIC_ARRAY_CONFIG"] = str(PROJECT_ROOT / "antenna_mapping_9197.yaml")
    env["EPIC_ANTENNA_COORDS"] = str(PROJECT_ROOT / "antenna_coordinates.csv")
    env["EPIC_CENTER_FREQ_MHZ"] = "1355"

    cmd = [
        str(EPIC_BIN),
        "--offline",
        "--out_dir", str(out_frame_dir.resolve()),
        "--disable_metrics",
        "--imagesize", "64",
        "--channels", "1024",
        "--chan_nbin", "8",
        "--seq_accum", "40",
        "--nts", "1000",
        "--imageres", "0.05",
        "--accum_16bit",
        "--nstreams", "1",
        "--support", "2",
        "--aeff", "8",
        "--kernel_oversample", "2",
        "--ngpus", "1",
        "--gpu_ids", "0",
    ]
    try:
        subprocess.run(
            cmd,
            cwd=str(BUILD_DIR),
            env=env,
            capture_output=True,
            timeout=120,
        )
    except Exception as e:
        return frame_idx, str(e)

    # 取该帧输出目录中最新一张 *_QL.png 作为本帧
    ql_list = sorted(out_frame_dir.glob("*_QL.png"), key=lambda p: p.stat().st_mtime, reverse=True)
    dst = frames_dir / f"frame_{frame_idx:05d}.png"
    if ql_list:
        shutil.copy2(ql_list[0], dst)
    return frame_idx, None


def main():
    gulp_dir = os.environ.get("GULP_NPZ_DIR", str(PROJECT_ROOT / "data" / "epic_ready" / "gulps_9197"))
    frames_dir = os.environ.get("FRAMES_DIR", str(PROJECT_ROOT / "results" / "epic_video_frames"))
    tmp_base = os.environ.get("TMP_OUT", str(PROJECT_ROOT / "results" / "epic_video_tmp"))
    video_out = os.environ.get("VIDEO_OUT", str(Path(frames_dir) / "9197_epic_calib.gif"))
    jobs = int(os.environ.get("EPIC_VIDEO_JOBS", "4"))

    if "--jobs" in sys.argv:
        i = sys.argv.index("--jobs")
        if i + 1 < len(sys.argv):
            jobs = int(sys.argv[i + 1])

    npz_pattern = str(Path(gulp_dir) / "frame_*.npz")
    npz_files = sorted(glob.glob(npz_pattern), key=lambda p: int(Path(p).stem.replace("frame_", "")))
    if not npz_files:
        # 兼容旧命名 gulp_*.npz
        npz_pattern = str(Path(gulp_dir) / "gulp_*.npz")
        npz_files = sorted(glob.glob(npz_pattern), key=lambda p: int(Path(p).stem.replace("gulp_", "")))
    if not npz_files:
        print(f"错误: 未找到帧 npz，目录 {gulp_dir}")
        print("请先运行: python3 scripts/convert_9197_to_gulp_npz.py 0 [1 2 ...]")
        sys.exit(1)

    if not EPIC_BIN.exists():
        print(f"错误: 未找到 EPIC 可执行文件 {EPIC_BIN}")
        print("请先构建: cd build && cmake .. && make -j")
        sys.exit(1)

    Path(frames_dir).mkdir(parents=True, exist_ok=True)
    Path(tmp_base).mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  并行多次调用 EPIC 成图 → 收集帧 → 拼接视频")
    print("=" * 60)
    print(f"  帧 npz 目录: {gulp_dir}  共 {len(npz_files)} 帧 (1s/帧)")
    print(f"  并行数: {jobs}  帧输出: {frames_dir}")
    print("=" * 60)

    args_list = [
        (i, npz_files[i], frames_dir, tmp_base, None)
        for i in range(len(npz_files))
    ]
    failed = []
    if jobs <= 1:
        for a in args_list:
            idx, err = run_one_frame(a)
            if err:
                failed.append((idx, err))
            else:
                print(f"  帧 {idx + 1}/{len(npz_files)} 完成")
    else:
        with Pool(jobs) as pool:
            for idx, err in pool.imap_unordered(run_one_frame, args_list):
                if err:
                    failed.append((idx, err))
                else:
                    print(f"  帧 {idx + 1}/{len(npz_files)} 完成")

    if failed:
        print(f"  失败 {len(failed)} 帧: {failed[:5]}{'...' if len(failed) > 5 else ''}")

    print("\n【拼接视频】")
    subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "scripts" / "stitch_video.py"), frames_dir, video_out],
        cwd=str(PROJECT_ROOT),
        check=False,
    )
    print(f"\n完成。视频: {video_out}  帧: {frames_dir}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
仅做「帧图 → 视频」拼接，不成图、不读原始数据。

用法:
  python3 scripts/stitch_video.py <帧目录> [输出路径]
  帧目录下需有 frame_00000.png, frame_00001.png, ...（按数字顺序）
  输出默认 <帧目录>/video.gif；若指定 .mp4 且本机有 ffmpeg 则出 mp4。
"""
import os
import sys
import subprocess
import glob
from pathlib import Path

def main():
    if len(sys.argv) < 2:
        print("用法: python3 stitch_video.py <帧目录> [输出路径]")
        sys.exit(1)
    frames_dir = sys.argv[1].rstrip("/")
    out_path = sys.argv[2] if len(sys.argv) > 2 else os.path.join(frames_dir, "video.gif")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    # 按数字排序的 frame_*.png
    pattern = os.path.join(frames_dir, "frame_*.png")
    files = sorted(glob.glob(pattern), key=lambda p: int(os.path.basename(p).replace("frame_", "").replace(".png", "")))
    if not files:
        print(f"未找到帧: {pattern}")
        sys.exit(1)

    try:
        import shutil
    except ImportError:
        shutil = None
    ext = os.path.splitext(out_path)[1].lower()
    if ext == ".mp4":
        ffmpeg = shutil.which("ffmpeg") if shutil else None
        if ffmpeg:
            # 生成 filelist 或直接用 -i pattern（部分 ffmpeg 支持）
            list_file = os.path.join(frames_dir, "list.txt")
            with open(list_file, "w") as f:
                for p in files:
                    f.write(f"file '{os.path.abspath(p)}'\n")
                f.write("")
            subprocess.run([
                ffmpeg, "-y", "-f", "concat", "-safe", "0", "-i", list_file,
                "-c:v", "libx264", "-pix_fmt", "yuv420p", "-r", "25", out_path
            ], check=True)
            try:
                os.remove(list_file)
            except Exception:
                pass
            print(f"  已生成 MP4: {out_path}")
            return
        # 无 ffmpeg 则改输出 gif
        out_path = out_path.replace(".mp4", ".gif")
        print("  未找到 ffmpeg，改为输出 GIF")

    # GIF
    try:
        from PIL import Image
    except ImportError:
        print("需要 PIL/Pillow 生成 GIF: pip install Pillow")
        sys.exit(1)
    ims = [Image.open(p).convert("P", palette=Image.ADAPTIVE, colors=256) for p in files]
    ims[0].save(out_path, save_all=True, append_images=ims[1:], duration=80, loop=0)
    print(f"  已生成 GIF: {out_path}  ({len(files)} 帧)")


if __name__ == "__main__":
    main()

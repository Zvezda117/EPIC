#!/usr/bin/env python3
"""
9197 长时间积分成图：用 ≥1s 积分提升信噪比

策略：
1. Incoherent 平均：对多帧脏图做强度平均 → SNR ∝ sqrt(N_frame)，不受条纹旋转影响
2. Coherent 平均：先对可见度在时间上相干平均再成图 → 1s 内条纹旋转有限，可尝试

数据：data/epic_ready/9197_4station_106gulps_128chan.npz
      (106 gulps × 40ms = 4.24s)，每 gulp (128 chan, 4 stands, 2 pol)
"""
import numpy as np
import matplotlib.pyplot as plt
import os

# 天线 ENU (米)，相对 N3
ANT_ENU = {
    'N3': (0.0, 0.0),
    'N4': (21.06, 0.035),
    'N5': (-2.661, -49.898),
    'N6': (23.804, -49.902),
}
WAVELENGTH = 3e8 / 1355e6  # 1355 MHz

def get_baselines():
    pos = np.array([ANT_ENU['N3'], ANT_ENU['N4'], ANT_ENU['N5'], ANT_ENU['N6']])
    baselines = []
    for i in range(4):
        for j in range(i + 1, 4):
            u = (pos[j, 0] - pos[i, 0]) / WAVELENGTH
            v = (pos[j, 1] - pos[i, 1]) / WAVELENGTH
            baselines.append((i, j, u, v))
    return baselines

def dirty_image_from_visibilities(vis_list, baselines, image_size=64, fov_deg=30):
    """从可见度列表 (每帧一组可见度) 做 incoherent 平均脏图。"""
    l = np.linspace(-np.sin(np.deg2rad(fov_deg/2)), np.sin(np.deg2rad(fov_deg/2)), image_size)
    m = np.linspace(-np.sin(np.deg2rad(fov_deg/2)), np.sin(np.deg2rad(fov_deg/2)), image_size)
    L, M = np.meshgrid(l, m)
    accum = np.zeros((image_size, image_size), dtype=np.float64)
    for vis in vis_list:
        dirty = np.zeros((image_size, image_size), dtype=np.complex128)
        for (i, j, u, v), v_ij in zip(baselines, vis):
            phase = -2j * np.pi * (u * L + v * M)
            dirty += v_ij * np.exp(phase) + np.conj(v_ij) * np.exp(-phase)
        accum += np.real(dirty)
    return accum / len(vis_list)

def visibility_per_gulp(gulp_data, baselines):
    """gulp_data: (nchan, nstand, npol) → 每基线一个复数 (跨通道平均)。"""
    vis = []
    for i, j, u, v in baselines:
        v_ij = np.mean(gulp_data[:, i, 0] * np.conj(gulp_data[:, j, 0]))
        vis.append(v_ij)
    return vis

def main():
    data_path = 'data/epic_ready/9197_4station_106gulps_128chan.npz'
    if not os.path.exists(data_path):
        data_path = os.path.join(os.path.dirname(__file__), '..', data_path)
    npz = np.load(data_path)
    data = npz['data']   # (ngulp, nchan, nstand, npol)
    ngulps, nchan, nstand, npol = data.shape
    baselines = get_baselines()

    # 积分时间选项 (gulp 数 × 40ms)
    integrations = [
        (25, '1s'),
        (50, '2s'),
        (75, '3s'),
        (106, '4.24s'),
    ]
    os.makedirs('results/long_integration', exist_ok=True)

    for n_gulp, label in integrations:
        if n_gulp > ngulps:
            continue
        # 使用前 n_gulp 帧
        vis_list = [visibility_per_gulp(data[t], baselines) for t in range(n_gulp)]
        img = dirty_image_from_visibilities(vis_list, baselines)
        int_time_ms = n_gulp * 40
        vmax = np.percentile(np.abs(img), 99)
        vmin = -vmax * 0.3

        plt.style.use('dark_background')
        fig, ax = plt.subplots(figsize=(8, 8))
        im = ax.imshow(img, origin='lower', cmap='viridis',
                       extent=[-30/2, 30/2, -30/2, 30/2],
                       vmin=vmin, vmax=vmax)
        ax.set_xlabel('L (deg)')
        ax.set_ylabel('M (deg)')
        ax.set_title(f'9197 CasA | 4 stations | Incoherent {label} ({int_time_ms} ms, {n_gulp} frames)')
        fig.colorbar(im, ax=ax, shrink=0.8, label='Flux')
        out = f'results/long_integration/9197_incoherent_{label.replace(".", "p")}.png'
        plt.savefig(out, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f'Saved {out}')

    # Coherent 平均：先对可见度在时间上取平均，再成一张图
    for n_gulp, label in integrations:
        if n_gulp > ngulps:
            continue
        vis_per_frame = [visibility_per_gulp(data[t], baselines) for t in range(n_gulp)]
        # 每条基线：复数平均
        vis_avg = [np.mean([v[b] for v in vis_per_frame]) for b in range(len(baselines))]
        l = np.linspace(-np.sin(np.deg2rad(15)), np.sin(np.deg2rad(15)), 64)
        m = np.linspace(-np.sin(np.deg2rad(15)), np.sin(np.deg2rad(15)), 64)
        L, M = np.meshgrid(l, m)
        dirty = np.zeros((64, 64), dtype=np.complex128)
        for (i, j, u, v), v_ij in zip(baselines, vis_avg):
            phase = -2j * np.pi * (u * L + v * M)
            dirty += v_ij * np.exp(phase) + np.conj(v_ij) * np.exp(-phase)
        img = np.real(dirty)
        int_time_ms = n_gulp * 40
        vmax = np.percentile(np.abs(img), 99)
        vmin = -vmax * 0.3
        plt.style.use('dark_background')
        fig, ax = plt.subplots(figsize=(8, 8))
        im = ax.imshow(img, origin='lower', cmap='viridis',
                       extent=[-15, 15, -15, 15], vmin=vmin, vmax=vmax)
        ax.set_xlabel('L (deg)')
        ax.set_ylabel('M (deg)')
        ax.set_title(f'9197 CasA | 4 stations | Coherent {label} ({int_time_ms} ms)')
        fig.colorbar(im, ax=ax, shrink=0.8, label='Flux')
        out = f'results/long_integration/9197_coherent_{label.replace(".", "p")}.png'
        plt.savefig(out, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f'Saved {out}')

    print('Done. Check results/long_integration/')

if __name__ == '__main__':
    main()

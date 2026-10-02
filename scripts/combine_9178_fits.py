#!/usr/bin/env python3
"""Combine 9178 full16 EPIC chunk FITS files into one integrated FITS."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pattern",
        default="results/9178_crab_full16_seq*_fixedscale_res002/*.fits",
        help="Glob pattern for per-chunk FITS files",
    )
    parser.add_argument(
        "--output-dir",
        default="results/9178_crab_full16_fullfile_fixedscale_res002",
    )
    parser.add_argument("--label", default="9178_crab_full16_fullfile_sum_2145ms")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = sorted(Path(".").glob(args.pattern))
    if not paths:
        raise SystemExit(f"no FITS files matched {args.pattern!r}")

    total = None
    first_hdul = None
    for path in paths:
        hdul = fits.open(path)
        image_hdu = next(hdu for hdu in hdul if hdu.data is not None)
        data = image_hdu.data.astype(np.float64)
        total = data if total is None else total + data
        if first_hdul is None:
            first_hdul = hdul
        else:
            hdul.close()

    assert total is not None
    assert first_hdul is not None

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fits_path = out_dir / f"{args.label}.fits"
    png_path = out_dir / f"{args.label}_XXYY_sum.png"

    primary = fits.PrimaryHDU(header=first_hdul[0].header)
    image_header = next(hdu.header for hdu in first_hdul if hdu.data is not None).copy()
    image_header["NCHUNKS"] = len(paths)
    image_header["INTTIM"] = 2.14536
    image_header["BUNITS"] = "UNCALIB"
    image_header.add_history("Summed from fixed-scale 9178 full16 EPIC chunk FITS files.")
    image = fits.ImageHDU(total.astype(np.float32), header=image_header)
    fits.HDUList([primary, image]).writeto(fits_path, overwrite=True)

    xx_yy = total[0].sum(axis=0) + total[1].sum(axis=0)
    lo, hi = np.nanpercentile(xx_yy, [1, 99.5])
    fig, ax = plt.subplots(figsize=(7, 6), dpi=150)
    im = ax.imshow(xx_yy, origin="lower", cmap="inferno", vmin=lo, vmax=hi)
    ax.set_title("9178 full16 fixed-scale integrated XX+YY")
    ax.set_xlabel("x pixel")
    ax.set_ylabel("y pixel")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(png_path)
    plt.close(fig)

    print(f"combined {len(paths)} chunks")
    print(f"wrote {fits_path}")
    print(f"wrote {png_path}")
    print(
        "stats",
        "shape", total.shape,
        "nonzero", int(np.count_nonzero(total)),
        "min", float(np.nanmin(total)),
        "max", float(np.nanmax(total)),
        "mean", float(np.nanmean(total)),
    )

    first_hdul.close()


if __name__ == "__main__":
    main()

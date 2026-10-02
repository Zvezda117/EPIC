"""Pure numerical conventions shared by EPIC's Python bridge and tests."""
import numpy as np


def correction_grid(kernels, grid_size, support, nchan, oversample=1,
                    relative_power_floor=1e-6):
    """Inverse gridding power response in the CUDA image's pixel order.

    Retains upstream NumPy IFFT normalization (a delta has correction N**4,
    N=grid_size*oversample). Pixels below the per-channel relative response
    floor are masked to zero; their response cannot be safely inverted.
    This is gridding correction, not antenna/flux calibration.
    """
    dimensions = (grid_size, support, nchan, oversample)
    if any(not isinstance(v, (int, np.integer)) or v <= 0 for v in dimensions):
        raise ValueError("Grid, support, channels and oversampling must be positive integers")
    if grid_size % 2 or not 0 < relative_power_floor < 1:
        raise ValueError("Require an even image grid and a relative power floor in (0, 1)")
    size = grid_size * oversample
    if support > size // 2:
        raise ValueError("Kernel support does not fit in the padded grid")
    kernel = np.asarray(kernels, dtype=np.float64).reshape(nchan, support, support).copy()
    total = kernel.sum(axis=(1, 2), keepdims=True)
    if not np.isfinite(kernel).all() or not np.isfinite(total).all() or np.any(total <= 0):
        raise ValueError("Gridding kernel must be finite with positive channel sums")
    kernel /= total
    padded = np.zeros((nchan, size, size), dtype=np.float64)
    center = size // 2
    padded[:, center:center+support, center:center+support] = kernel
    power = np.abs(np.fft.ifft2(padded, axes=(1, 2)))**2
    power = np.fft.fftshift(power, axes=(1, 2))
    offset = (size-grid_size)//2
    power = power[:, offset:offset+grid_size, offset:offset+grid_size]
    power = np.fft.fftshift(power.transpose(0, 2, 1), axes=(1, 2))
    threshold = power.max(axis=(1, 2), keepdims=True) * relative_power_floor
    result = np.zeros_like(power)
    np.divide(1.0, power, out=result, where=(power >= threshold) & (power > 0))
    if not np.isfinite(result).all():
        raise ValueError("Non-finite gridding correction")
    return result


def output_frequency_axis(chan0, nchan, chan_width, input_chan_width=25000.0):
    """First/central frequency after contiguous, equally weighted binning.

    chan0 names the first input channel's center in input_chan_width units.
    """
    values = np.asarray([chan0, nchan, chan_width, input_chan_width], dtype=float)
    if not np.isfinite(values).all() or np.any(values <= 0):
        raise ValueError("Invalid frequency axis")
    ratio = chan_width / input_chan_width
    if chan0 != int(chan0) or nchan != int(nchan) or ratio != int(ratio):
        raise ValueError("Frequency binning requires integral channel indices/counts")
    first = chan0 * input_chan_width + (chan_width-input_chan_width)/2
    center = first + (nchan-1)*chan_width/2
    return first, center, float(chan_width)

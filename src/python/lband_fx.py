"""Native FP32 dual-polarization covariance + exact 3-D steering imager.

Geometry uses a shared caller-defined Cartesian frame: metres, hertz and unit
direction vectors. Gains multiply voltages, not powers. No station/time inference.
"""
import ctypes as ct
import os
from pathlib import Path
import threading

import numpy as np


def _integer(value, name, minimum):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f'{name} must be an integer')
    if value < minimum or value > np.iinfo(np.int32).max:
        raise ValueError(f'{name} must be in [{minimum}, 2147483647]')
    return int(value)


def _array(value, dtype, name):
    with np.errstate(over='ignore', invalid='ignore'):
        result = np.ascontiguousarray(value, dtype=dtype)
    if not np.isfinite(result).all():
        raise ValueError(f'{name} must be finite after dtype conversion')
    return result


def _library():
    explicit = os.environ.get('EPIC_FX_LIBRARY')
    path = Path(explicit).expanduser() if explicit else Path(__file__).resolve().parents[1] / 'fx/libepic_fx.so'
    try:
        lib = ct.CDLL(str(path))
    except OSError as exc:
        source = 'EPIC_FX_LIBRARY' if explicit else 'repo-relative native library'
        raise RuntimeError(f'{source} could not load {path}; build with src/fx/build.sh: {exc}') from exc
    lib.epic_fx_create.argtypes = [ct.c_int]*6 + [ct.c_void_p]*5
    lib.epic_fx_create.restype = ct.c_void_p
    lib.epic_fx_image.argtypes = [ct.c_void_p]*3
    lib.epic_fx_image.restype = ct.c_int
    lib.epic_fx_destroy.argtypes = [ct.c_void_p]
    lib.epic_fx_destroy.restype = None
    lib.epic_fx_error.argtypes = []
    lib.epic_fx_error.restype = ct.c_char_p
    return lib


class FXImager:
    """Persistent device buffers/steering/handle; image() is synchronous.

    Input complex64[T,C,N,2]; result float32[C,P,4] = mean_t of
    XX, YY, Re(X*conj(Y)), Im(X*conj(Y)). N counts physical antenna slots.
    Missing individual inputs use zero voltages or zero voltage gains.
    remove_autocorr deletes same-antenna terms from all XX/YY/XY products.
    """
    def __init__(self, positions_m, frequencies_hz, directions, phase_center,
                 gains=None, ntime=1, device=1, remove_autocorr=False):
        self._handle = None
        self._lock = threading.Lock()
        ntime = _integer(ntime, 'ntime', 1)
        device = _integer(device, 'device', 0)
        if not isinstance(remove_autocorr, (bool, np.bool_)):
            raise ValueError('remove_autocorr must be bool')
        pos = _array(positions_m, np.float64, 'positions_m')
        freq = _array(frequencies_hz, np.float64, 'frequencies_hz')
        dirs = _array(directions, np.float64, 'directions')
        center = _array(phase_center, np.float64, 'phase_center')
        if pos.ndim != 2 or pos.shape[1] != 3 or pos.shape[0] < 1:
            raise ValueError('positions_m must have shape [N,3], N>0')
        if freq.ndim != 1 or freq.size < 1 or (freq <= 0).any():
            raise ValueError('frequencies_hz must have shape [C] and be positive')
        if dirs.ndim != 2 or dirs.shape[1] != 3 or dirs.shape[0] < 1:
            raise ValueError('directions must have shape [P,3], P>0')
        if center.shape != (3,):
            raise ValueError('phase_center must have shape [3]')
        if (not np.allclose(np.linalg.norm(dirs, axis=1), 1., rtol=0., atol=1e-6)
                or not np.isclose(np.linalg.norm(center), 1., rtol=0., atol=1e-6)):
            raise ValueError('directions and phase_center must be unit vectors')
        n, c, p = pos.shape[0], freq.size, dirs.shape[0]
        for value, name in ((n, 'N'), (c, 'C'), (p, 'P')):
            _integer(value, name, 1)
        # Python integers do not wrap; native repeats this with checked size_t.
        peak_bytes = (4*c*n*ntime + 2*c*n*p + 3*c*n*n + 2*c*n)*8
        peak_bytes += 4*c*p*4 + (3*n+c+3*p+3)*8
        if peak_bytes > 900*1024*1024:
            raise ValueError('FX peak buffers exceed 900 MiB; reduce channel tile or window size')
        gain = np.ones((c, n, 2), np.complex64) if gains is None else _array(gains, np.complex64, 'gains')
        if gain.shape != (c, n, 2):
            raise ValueError('gains must have shape [C,N,2]')
        self.input_shape = (ntime, c, n, 2)
        self.output_shape = (c, p, 4)
        self.device = device
        self.remove_autocorr = bool(remove_autocorr)
        self._lib = _library()
        self._handle = self._lib.epic_fx_create(n, c, p, ntime, device,
                         int(remove_autocorr), *[ct.c_void_p(a.ctypes.data)
                                               for a in (pos, freq, dirs, center, gain)])
        if not self._handle:
            raise RuntimeError(self._error())

    def _error(self):
        return self._lib.epic_fx_error().decode('utf-8', errors='replace')

    def image(self, voltages):
        with self._lock:
            if not self._handle:
                raise RuntimeError('FXImager is closed')
            value = _array(voltages, np.complex64, 'voltages')
            if value.shape != self.input_shape:
                raise ValueError(f'voltages must have shape {self.input_shape}, got {value.shape}')
            result = np.empty(self.output_shape, np.float32)
            status = self._lib.epic_fx_image(self._handle, ct.c_void_p(value.ctypes.data),
                                              ct.c_void_p(result.ctypes.data))
            if status:
                raise RuntimeError(self._error())
            return result

    def close(self):
        with self._lock:
            if self._handle:
                self._lib.epic_fx_destroy(self._handle)
                self._handle = None

    def __enter__(self):
        if not self._handle:
            raise RuntimeError('FXImager is closed')
        return self

    def __exit__(self, *_):
        self.close()

    def __del__(self):
        if getattr(self, '_handle', None):
            self.close()

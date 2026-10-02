"""Representative synthetic NEW API wall timing, exclusively GPU1."""
import importlib.util
import json
from pathlib import Path
import sys
import time
import numpy as np

root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(root/'src/python'))
from lband_fx import FXImager

rng = np.random.default_rng(20261002)
t, c, n, p = 1000, 128, 16, 4096
pos = rng.uniform(-15, 15, (n, 3))
pos[10:] = pos[:10].mean(axis=0)
freq = np.linspace(1.230e9, 1.480e9, c)
l, m = np.meshgrid(np.linspace(-0.2, 0.2, 64), np.linspace(-0.2, 0.2, 64))
dirs = np.stack((l.ravel(), m.ravel(), np.sqrt(1-l.ravel()**2-m.ravel()**2)), axis=1)
center = np.array([0., 0., 1.])
gain = np.zeros((c, n, 2), np.complex64)
# Synthetic H5/V7 availability counts; positions/voltages are NOT observed data.
gain[:, [2, 4, 5, 6, 8], 0] = 1
gain[:, [0, 1, 2, 3, 4, 7, 9], 1] = 1
v = (rng.standard_normal((t, c, n, 2), dtype=np.float32) +
     1j*rng.standard_normal((t, c, n, 2), dtype=np.float32)).astype(np.complex64)
start = time.perf_counter()
imager = FXImager(pos, freq, dirs, center, gain, ntime=t, device=1)
setup_ms = (time.perf_counter()-start)*1e3
with imager:
    imager.image(v)  # one warmup
    times = []
    for _ in range(5):
        start = time.perf_counter()
        output = imager.image(v)
        times.append((time.perf_counter()-start)*1e3)
spec = importlib.util.spec_from_file_location('oracle', root/'tests/test_lband_fx.py')
oracle = importlib.util.module_from_spec(spec); spec.loader.exec_module(oracle)
channels = np.array([0, c-1]); pixels = np.arange(16)*257
expected = oracle.direct_oracle(v[:, channels], pos, freq[channels], dirs[pixels],
                                center, gain[channels])
actual = output[channels][:, pixels]
np.testing.assert_allclose(actual, expected, rtol=4e-5, atol=4e-5)
record = dict(kind='synthetic_dualpol_new_API_not_observation', device=1,
              T=t, C=c, physical_slots=n, active_antennas=10, active_H=5, active_V=7,
              P=p, precision='FP32_PEDANTIC_NO_TF32', remove_autocorr=False,
              setup_ms=setup_ms, warmup=1, repeats=5, image_wall_ms=times,
              image_wall_median_ms=float(np.median(times)),
              timing_includes='Python/native validation + packing/gains + H2D + covariance/imaging + D2H + output finite check',
              oracle_channels=channels.tolist(), oracle_pixels_per_channel=16,
              oracle_max_abs=float(np.max(abs(actual-expected))),
              oracle_max_scaled=float(np.max(abs(actual-expected))/np.max(abs(expected))))
(Path(__file__).resolve().parent/'API_BENCHMARK.json').write_text(json.dumps(record, indent=2)+'\n')
print(json.dumps(record, indent=2))

"""Exercise sky tracking through the real CLI/CUDA path, including changing windows."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import numpy as np
from source_functions import ROOT
import test_sky_geometry


class SkyRunnerTests(unittest.TestCase):
    def setup_case(self, path, config):
        pos = np.array([[0., 0., 0.], [10., 2., 0.], [2., 25., 7.], [40., -10., 5.]])
        freq = np.array([1.3e9, 1.4e9])
        np.savez(path/'geometry.npz', positions_m=pos, frequencies_hz=freq)
        (path/'sky.json').write_text(json.dumps(config))
        np.savez(path/'input.npz', data=np.ones((5, 2, 4, 2), np.complex64))
        command = [sys.executable, str(ROOT/'scripts/run_lband_fx.py'),
                   '--input', str(path/'input.npz'), '--geometry', str(path/'geometry.npz'),
                   '--sky-config', str(path/'sky.json'), '--output', str(path/'out'),
                   '--devices', '1', '--channel-tile', '1', '--window-sequences', '2',
                   '--voltage-convention', 'raw', '--coordinate-frame', 'local_ENU',
                   '--iq-convention', 'I+iQ', '--remove-autocorr']
        return pos, freq, command

    def test_missing_epoch_is_rejected_before_output_creation(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td)
            _, _, command = self.setup_case(p, test_sky_geometry.SkyGeometryTests().config() | {'input_start': None})
            result = subprocess.run(command, capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('first-sample timestamp', result.stderr)
            self.assertFalse((p/'out').exists())

    @unittest.skipUnless(os.environ.get('EPIC_FX_TEST_GPU') == '1', 'explicit GPU test required')
    def test_raw_nonzenith_source_recovers_through_cli_with_updated_windows(self):
        self.check_sky_run('1')

    @unittest.skipUnless(os.environ.get('EPIC_FX_TEST_TWO_GPU') == '1', 'explicit two-GPU test required')
    def test_two_gpu_sky_tracking(self):
        self.check_sky_run('0,1')

    def check_sky_run(self, devices):
        from sky_geometry import SkyModel
        with tempfile.TemporaryDirectory() as td:
            p = Path(td)
            config = test_sky_geometry.SkyGeometryTests().config() | {'sequence_seconds': .01}
            pos, freq, command = self.setup_case(p, config)
            command[command.index('--devices')+1] = devices
            model = SkyModel(config, pos, freq)
            voltage = np.empty((5, 2, 4, 2), np.complex64)
            for begin in (0, 2, 4):
                _, center, _ = model.window(begin, min(2, 5-begin))
                raw = np.exp(-2j*np.pi*freq[:, None]*(pos@center)[None]/299792458.)
                voltage[begin:begin+2] = raw[None, ..., None]
            np.savez(p/'input.npz', data=voltage)
            result = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            manifest = json.loads((p/'out/manifest.json').read_text())
            self.assertEqual(manifest['time_basis'], 'explicit_first_sample_UTC')
            self.assertEqual(len(manifest['chunks']), 6)
            self.assertEqual({c['device'] for c in manifest['chunks']}, set(map(int, devices.split(','))))
            centers = []
            for chunk in manifest['chunks']:
                with np.load(p/'out'/chunk['file']) as output:
                    # Four antennas, diagonal removed: 4**2 - 4 = 12.
                    peak = 16*33+16
                    np.testing.assert_array_equal(np.argmax(output['image'][..., 0], axis=1), [peak])
                    np.testing.assert_allclose(output['image'][0, peak], [12, 12, 12, 0], atol=2e-5)
                    expected_dirs, expected_center, expected_meta = model.window(chunk['sequence_start'], chunk['sequence_count'])
                    np.testing.assert_allclose(output['directions'], expected_dirs, atol=1e-12)
                    meta = json.loads(str(output['metadata_json']))
                    self.assertEqual(meta['midpoint_utc'], expected_meta['midpoint_utc'])
                    centers.append(output['phase_center'])
            self.assertGreater(np.linalg.norm(centers[0]-centers[-1]), 1e-7)

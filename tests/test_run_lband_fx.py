import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / 'scripts/run_lband_fx.py'


class CLIContractTests(unittest.TestCase):
    def test_runner_exists_and_documents_required_explicit_conventions(self):
        self.assertTrue(SCRIPT.is_file(), 'NPZ FX runner has not been implemented')
        result = subprocess.run([sys.executable, str(SCRIPT), '--help'],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0)
        for option in ('--input', '--geometry', '--output', '--devices',
                       '--channel-tile', '--window-sequences',
                       '--voltage-convention', '--coordinate-frame', '--iq-convention'):
            self.assertIn(option, result.stdout)


@unittest.skipUnless(os.environ.get('EPIC_FX_TEST_GPU') == '1', 'explicit GPU1 test required')
class CLIIntegrationTests(unittest.TestCase):
    def test_partial_window_and_channel_tiles_preserve_mean_and_do_not_overwrite(self):
        self.check_tiled_run('1')

    @unittest.skipUnless(os.environ.get('EPIC_FX_TEST_TWO_GPU') == '1', 'explicit two-GPU test required')
    def test_parallel_two_gpu_workers_preserve_channel_and_time_ranges(self):
        self.check_tiled_run('0,1')

    def check_tiled_run(self, devices):
        self.assertTrue(SCRIPT.is_file(), 'NPZ FX runner has not been implemented')
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)
            rng = np.random.default_rng(31)
            v = (rng.normal(size=(5, 3, 2, 2)) +
                 1j*rng.normal(size=(5, 3, 2, 2))).astype(np.complex64)
            pos = np.array([[0., 0., 0.], [0.73, 0.19, -0.12]])
            freq = np.array([1.2e9, 1.3e9, 1.4e9])
            dirs = np.array([[0., 0., 1.], [0.6, 0., 0.8]])
            gains = np.array([[[1+0.1j, 0.2-0.3j], [0.7+0.3j, 1+0j]]]*3)
            np.savez(path/'input.npz', data=v)
            np.savez(path/'geometry.npz', positions_m=pos, frequencies_hz=freq,
                     directions=dirs, phase_center=dirs[0], gains=gains)
            command = [sys.executable, str(SCRIPT), '--input', str(path/'input.npz'),
                       '--geometry', str(path/'geometry.npz'), '--output', str(path/'out'),
                       '--devices', devices, '--channel-tile', '2', '--window-sequences', '2',
                       '--voltage-convention', 'raw', '--coordinate-frame', 'local_ENU',
                       '--iq-convention', 'I+iQ', '--remove-autocorr']
            result = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            spec = importlib.util.spec_from_file_location('fx_oracle_test', ROOT/'tests/test_lband_fx.py')
            mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
            raw_steering_gain = np.exp(2j*np.pi*freq[:, None]*(pos@dirs[0])[None]/299792458.)
            effective_gain = gains * raw_steering_gain[..., None]
            manifest = json.loads((path/'out/manifest.json').read_text())
            self.assertEqual(len(manifest['chunks']), 6)
            self.assertEqual(set(c['device'] for c in manifest['chunks']),
                             set(map(int, devices.split(','))))
            for chunk in manifest['chunks']:
                begin, end = chunk['sequence_start'], chunk['sequence_end']
                lo, hi = chunk['channel_start'], chunk['channel_end']
                expected = mod.direct_oracle(v[begin:end, lo:hi], pos, freq[lo:hi],
                                            dirs, dirs[0], effective_gain[lo:hi], True)
                with np.load(path/'out'/chunk['file'], allow_pickle=False) as saved:
                    np.testing.assert_allclose(saved['image'], expected, rtol=4e-5, atol=4e-5)
                    np.testing.assert_array_equal(saved['frequencies_hz'], freq[lo:hi])
                    metadata = json.loads(str(saved['metadata_json']))
                    self.assertEqual(metadata['sequence_count'], end-begin)
                    self.assertEqual(metadata['normalization'], 'mean_over_valid_sequences')
                    self.assertEqual(metadata['time_basis'], 'sequence_indices_no_UTC')
            self.assertEqual(max(c['sequence_end'] for c in manifest['chunks']), 5)
            self.assertNotEqual(subprocess.run(command, capture_output=True).returncode, 0)


if __name__ == '__main__':
    unittest.main()

"""Independent direct-electric-field oracle for the native FX backend."""
import importlib.util
import os
from pathlib import Path
import unittest

import numpy as np

MODULE = Path(__file__).resolve().parents[1] / 'src/python/lband_fx.py'


def backend():
    assert MODULE.is_file(), 'native FX Python interface has not been implemented'
    spec = importlib.util.spec_from_file_location('lband_fx_under_test', MODULE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def direct_oracle(v, pos, freq, dirs, center, gains=None, remove=False):
    """Double precision field summation, independent of covariance algebra."""
    v = np.asarray(v, dtype=np.complex128)
    if gains is not None:
        v = v * np.asarray(gains, dtype=np.complex128)[None]
    delay = np.asarray(pos) @ (np.asarray(dirs) - center).T / 299792458.0
    phase = np.exp(2j * np.pi * np.asarray(freq)[:, None, None] * delay[None])
    field = np.einsum('cnp,tcnq->tcpq', phase, v)
    xx = np.mean(abs(field[..., 0])**2, axis=0)
    yy = np.mean(abs(field[..., 1])**2, axis=0)
    xy = np.mean(field[..., 0] * field[..., 1].conj(), axis=0)
    if remove:
        xx -= np.mean(np.sum(abs(v[..., 0])**2, axis=2), axis=0)[:, None]
        yy -= np.mean(np.sum(abs(v[..., 1])**2, axis=2), axis=0)[:, None]
        xy -= np.mean(np.sum(v[..., 0]*v[..., 1].conj(), axis=2), axis=0)[:, None]
    return np.stack((xx, yy, xy.real, xy.imag), axis=-1)


class CPUContractTests(unittest.TestCase):
    def test_interface_is_available(self):
        self.assertTrue(hasattr(backend(), 'FXImager'))

    def test_oracle_product_sign_and_physical_auto(self):
        v = np.array([[[[1+2j, 3+4j]]]], dtype=np.complex64)
        args = (v, [[0., 0., 0.]], [1.355e9], [[0., 0., 1.]], [0., 0., 1.])
        np.testing.assert_allclose(direct_oracle(*args), [[[5., 25., 11., 2.]]])
        np.testing.assert_allclose(direct_oracle(*args, remove=True), 0)

    def test_invalid_geometry_and_frequency_fail_before_cuda_load(self):
        fx = backend().FXImager
        args = dict(positions_m=np.zeros((2, 3)), frequencies_hz=[1e9],
                    directions=[[0., 0., 1.]], phase_center=[0., 0., 1.], ntime=4)
        for override in ({'positions_m': [[0., 0.]]},
                         {'positions_m': [[0., np.nan, 0.]]},
                         {'frequencies_hz': [0.]}, {'frequencies_hz': [np.inf]},
                         {'directions': [[0., 0., 2.]]},
                         {'phase_center': [0., 0., 0.]}, {'ntime': 0},
                         {'device': -1}, {'device': 1.5},
                         {'gains': np.ones((1, 2, 1))},
                         {'gains': np.full((1, 2, 2), np.nan+0j)}):
            with self.subTest(override=override), self.assertRaises(ValueError):
                fx(**(args | override))

    def test_library_override_error_is_explicit(self):
        old = os.environ.get('EPIC_FX_LIBRARY')
        os.environ['EPIC_FX_LIBRARY'] = '/definitely/missing/libepic_fx.so'
        try:
            with self.assertRaisesRegex(RuntimeError, 'EPIC_FX_LIBRARY'):
                backend().FXImager([[0., 0., 0.]], [1e9], [[0., 0., 1.]],
                                   [0., 0., 1.], ntime=1)
        finally:
            if old is None:
                os.environ.pop('EPIC_FX_LIBRARY', None)
            else:
                os.environ['EPIC_FX_LIBRARY'] = old

    def test_large_case_rejected_before_loading_cuda(self):
        try:
            backend().FXImager([[0., 0., 0.]], [1e9], [[0., 0., 1.]],
                               [0., 0., 1.], ntime=2147483647)
        except Exception as exc:
            self.assertIsInstance(exc, ValueError)
            self.assertIn('channel tile', str(exc))
        else:
            self.fail('oversized input allocation was accepted')


@unittest.skipUnless(os.environ.get('EPIC_FX_TEST_GPU') == '1',
                     'GPU tests require explicit EPIC_FX_TEST_GPU=1; GPU1 only')
class GPUOracleTests(unittest.TestCase):
    def check_case(self, v, pos, freq, dirs, center, gains=None, remove=False):
        fx = backend().FXImager
        expected = direct_oracle(v, pos, freq, dirs, center, gains, remove)
        with fx(pos, freq, dirs, center, gains=gains, ntime=len(v),
                device=1, remove_autocorr=remove) as imager:
            actual = imager.image(v)
            self.assertEqual(actual.dtype, np.float32)
            self.assertEqual(actual.shape, expected.shape)
            np.testing.assert_allclose(actual, expected, rtol=4e-5, atol=4e-5)
            # Persistent buffers/handle must support different successive data.
            again = imager.image(np.asfortranarray(v * np.complex64(0.5)))
            np.testing.assert_allclose(again, expected * 0.25, rtol=4e-5, atol=4e-5)
            scale = max(float(np.max(abs(expected))), 1.)
            print(f'FX_ORACLE remove={remove} shape={actual.shape} '
                  f'max_abs={np.max(abs(actual-expected)):.8g} '
                  f'max_scaled={np.max(abs(actual-expected))/scale:.8g}')
        with self.assertRaisesRegex(RuntimeError, 'closed'):
            imager.image(v)
        imager.close()  # idempotent

    def test_one_antenna_complex_products_and_all_physical_autos_zero(self):
        v = np.array([[[[1+2j, 3+4j]]], [[[2-1j, -3+2j]]]], dtype=np.complex64)
        for remove in (False, True):
            self.check_case(v, [[1.2, -0.4, 2.7]], [1.355e9],
                            [[0., 0., 1.], [0.6, 0., 0.8]], [0., 0., 1.],
                            gains=np.array([[[0.5+0.3j, -1.+0.8j]]]), remove=remove)

    def test_two_antenna_coherent_and_cross_pol_auto(self):
        v = np.ones((5, 1, 2, 2), dtype=np.complex64)
        for remove in (False, True):
            self.check_case(v, [[0., 0., 0.], [0., 0., 0.]], [1.4e9],
                            [[0., 0., 1.]], [0., 0., 1.], remove=remove)

    def test_noninteger_noncoplanar_two_frequencies_and_complex_gain(self):
        rng = np.random.default_rng(32)
        v = (rng.normal(size=(9, 2, 4, 2)) +
             1j*rng.normal(size=(9, 2, 4, 2))).astype(np.complex64)
        pos = np.array([[0., 0., 0.], [10.123, -3.22, 0.7],
                        [-8.31, 15.2, -2.9], [17.1, 1.3, 4.2]])
        dirs = np.array([[0., 0., 1.], [0.3, -0.4, np.sqrt(0.75)],
                         [-0.1, 0.2, np.sqrt(0.95)]])
        gains = (rng.normal(size=(2, 4, 2)) +
                 1j*rng.normal(size=(2, 4, 2))).astype(np.complex64)
        gains[0, 2, 0] = 0  # missing physical input
        for gain in (None, gains):
            for remove in (False, True):
                self.check_case(v, pos, [1.23e9, 1.48e9], dirs, dirs[1], gain, remove)

    def test_zero_input(self):
        self.check_case(np.zeros((3, 2, 2, 2), np.complex64),
                        [[0., 0., 0.], [0.21, 0.14, -0.01]], [1.2e9, 1.4e9],
                        [[0., 0., 1.]], [0., 0., 1.], remove=True)

    def test_shape_finite_and_invalid_device_errors(self):
        fx = backend().FXImager
        args = ([[0., 0., 0.]], [1e9], [[0., 0., 1.]], [0., 0., 1.])
        with fx(*args, ntime=2, device=1) as imager:
            with self.assertRaises(ValueError):
                imager.image(np.zeros((1, 1, 1, 2), np.complex64))
            with self.assertRaises(ValueError):
                imager.image(np.full((2, 1, 1, 2), np.nan+0j, np.complex64))
        with self.assertRaisesRegex(RuntimeError, 'device'):
            fx(*args, ntime=2, device=9999)

    @unittest.skipUnless(os.environ.get('EPIC_FX_TEST_TWO_GPU') == '1',
                         'two-card test needs an explicitly coordinated GPU0 window')
    def test_two_card_channel_shards_match_whole(self):
        rng = np.random.default_rng(17)
        v = (rng.normal(size=(7, 4, 3, 2)) +
             1j*rng.normal(size=(7, 4, 3, 2))).astype(np.complex64)
        pos = np.array([[0., 0., 0.], [1.3, 0.21, -0.3], [-0.73, 0.16, 0.9]])
        freq = np.array([1.2e9, 1.3e9, 1.4e9, 1.5e9])
        dirs = np.array([[0., 0., 1.], [0.6, 0., 0.8]])
        gains = (rng.normal(size=(4, 3, 2)) +
                 1j*rng.normal(size=(4, 3, 2))).astype(np.complex64)
        fx = backend().FXImager
        with fx(pos, freq, dirs, dirs[1], gains=gains, ntime=7,
                device=1, remove_autocorr=True) as context:
            whole = context.image(v)
        parts = []
        for device, indices in ((0, slice(0, 2)), (1, slice(2, 4))):
            with fx(pos, freq[indices], dirs, dirs[1], gains=gains[indices],
                    ntime=7, device=device, remove_autocorr=True) as context:
                parts.append(context.image(v[:, indices]))
        split = np.concatenate(parts)
        np.testing.assert_allclose(split, whole, rtol=4e-5, atol=4e-5)
        np.testing.assert_allclose(whole, direct_oracle(v, pos, freq, dirs,
                                 dirs[1], gains, True), rtol=4e-5, atol=4e-5)
        print(f'FX_TWO_GPU_SHARD max_abs={np.max(abs(split-whole)):.8g}')


if __name__ == '__main__':
    unittest.main()

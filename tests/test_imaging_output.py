import os
import tempfile
import unittest
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.image
from astropy.io import fits
from astropy.time import Time
from astropy.coordinates import SkyCoord, FK5
from astropy.utils import iers

from source_functions import epic_functions


class ImagingOutputTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        old = Path.cwd()
        self.addCleanup(os.chdir, old)
        os.chdir(self.tmp.name)
        self.scope = epic_functions('get_correction_grid', 'save_output')
        self.scope.update(matplotlib=matplotlib, fits=fits, Time=Time,
                          SkyCoord=SkyCoord, FK5=FK5,
                          _station_lon_lat_deg=lambda: (117.49,36.656))
        iers.conf.auto_download = False

    def test_delta_correction_preserves_fft_normalization(self):
        for n, oversample in [(64,1),(64,2),(128,1)]:
            with self.subTest(n=n, oversample=oversample):
                corr = self.scope['get_correction_grid'](np.ones((1,1,1)),n,1,1,oversample)
                np.testing.assert_allclose(corr, float((n*oversample)**4))

    def test_asymmetric_kernel_has_analytic_spatial_response(self):
        kernel = np.array([[[.75,.25],[0,0]]])
        corr = self.scope['get_correction_grid'](kernel,64,2,1).reshape(64,64)
        self.assertAlmostEqual(corr[0,0],64.**4)
        self.assertAlmostEqual(corr[32,0],4*64.**4)

    def test_odd_channel_count_does_not_reorder_kernels(self):
        kernels = np.array([[[1.,0],[0,0]],[[.75,.25],[0,0]],[[.9,.1],[0,0]]])
        corr = self.scope['get_correction_grid'](kernels,64,2,3).reshape(3,64,64)
        np.testing.assert_allclose(corr[:,32,0], [64.**4,4*64.**4,64.**4/.64])

    def test_invalid_kernel_is_rejected_instead_of_silent_delta(self):
        with self.assertRaises(ValueError):
            self.scope['get_correction_grid'](np.zeros((1,2,2)),64,2,1)

    def test_zero_response_is_masked_and_float_bridge_remains_finite(self):
        corr = self.scope['get_correction_grid'](np.ones((1,2,2)),64,2,1).reshape(64,64)
        self.assertEqual(np.count_nonzero(corr == 0),127)
        np.testing.assert_array_equal(corr[32,:],0)
        np.testing.assert_array_equal(corr[:,32],0)
        self.assertAlmostEqual(corr[0,0],64.**4)
        self.assertTrue(np.isfinite(corr.astype(np.float32)).all())

    def test_binned_fits_frequency_centers_and_product_labels(self):
        meta = dict(grid_size=64, grid_res=.02, chan0=54136, nchan=32,
                    chan_width=100000, time_tag=946728000*196000000,
                    img_len_ms=400)
        filename = self.scope['save_output'](np.zeros((32,64,64,4)),64,32,self.tmp.name,meta)
        with fits.open(filename) as hdus:
            header = hdus[1].header
            centers = header['CRVAL3']+(np.arange(32)+1-header['CRPIX3'])*header['CDELT3']
            self.assertEqual(header['CDELT3'],100000)
            self.assertEqual(centers[0],1353437500)
            self.assertEqual(centers[-1],1356537500)
            self.assertEqual(header['CTYPE4'],'CORRPROD')
            self.assertEqual(header['POLPROD'],'XX,YY,RE_XY,IM_XY')


if __name__ == '__main__':
    unittest.main()

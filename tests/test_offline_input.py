import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from source_functions import epic_functions


class OfflineInputTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / 'gulp.npz'
        self.scope = epic_functions('get_40ms_gulp')
        self.load = self.scope['get_40ms_gulp']
        self.env = patch.dict(os.environ, {'EPIC_GULP_NPZ': str(self.path)})
        self.env.start()
        self.addCleanup(self.env.stop)
        self.data = np.zeros((2, 3, 16, 2), dtype=np.complex64)
        for t in range(2):
            for c in range(3):
                self.data[t, c, :, :] = 3*t+c

    def save(self, data=None, meta=None):
        np.savez(self.path, data=self.data if data is None else data,
                 meta=np.array([1700000000., 54136.]) if meta is None else meta)

    def test_channels_remain_distinct_in_gpu_channel_major_layout(self):
        self.save()
        result = self.load()
        by_channel = result['data'].reshape(3, 2, 16, 2)
        np.testing.assert_array_equal(by_channel[:, :, 0, 0], [[0, 3], [1, 4], [2, 5]])
        np.testing.assert_array_equal(result['meta'][3:], [2, 3, 16, 2])

    def test_missing_explicit_input_does_not_read_default_observation(self):
        cwd = Path.cwd()
        self.addCleanup(os.chdir, cwd)
        os.chdir(self.tmp.name)
        fallback = Path('data/40ms_128chan_600offset_gulp_c64_virtransit.npz')
        fallback.parent.mkdir()
        np.savez(fallback, data=self.data, meta=[1., 54136.])
        with self.assertRaises(FileNotFoundError):
            self.load()

    def test_rejects_values_that_would_wrap_in_signed_nibbles(self):
        self.data[0, 0, 0, 0] = 8
        self.save()
        with self.assertRaises(ValueError):
            self.load()

    def test_rejects_nonfinite_voltage(self):
        self.data[0, 0, 0, 0] = np.nan
        self.save()
        with self.assertRaises(ValueError):
            self.load()

    def test_rejects_wrong_antenna_width(self):
        self.save(self.data[:, :, :4, :])
        with self.assertRaises(ValueError):
            self.load()

    def test_rejects_frequency_index_that_overflows_cpp_int(self):
        self.save(meta=[1700000000., 2.**40])
        with self.assertRaises(ValueError):
            self.load()

    def test_rejects_frequency_multiplication_overflow(self):
        for chan0 in (1e30, 90000., 85899.):
            with self.subTest(chan0=chan0):
                self.save(meta=[1700000000., chan0])
                with self.assertRaises(ValueError):
                    self.load()

    def test_accepts_last_representable_channel_frequency(self):
        self.save(meta=[1700000000., 85897.])
        self.assertEqual(self.load()['meta'][1], 85897.)

    def test_rejects_ambiguous_metadata_length(self):
        self.save(meta=[1., 54136., 99.])
        with self.assertRaises(ValueError):
            self.load()

    def test_rejects_mismatched_sequence_count(self):
        self.save()
        with self.assertRaises(ValueError):
            self.load(expected_nseqs=1000)

    def test_rejects_too_small_c_buffer(self):
        self.save()
        with self.assertRaises(ValueError):
            self.load(capacity_bytes=self.data.size - 1)

    def test_accepts_exact_capacity_and_signed_nibble_endpoints(self):
        self.data[0, 0, 0, 0] = -8+7j
        self.save()
        result = self.load(expected_nseqs=2, capacity_bytes=self.data.size)
        self.assertEqual(result['data'][0], -8+7j)

    def test_frequency_shards_are_disjoint_and_keep_absolute_channels(self):
        self.save()
        first = self.load(channel_start=0, channel_count=1)
        second = self.load(channel_start=1, channel_count=2)
        np.testing.assert_array_equal(first['data'].reshape(1,2,16,2)[:,:,0,0],[[0,3]])
        np.testing.assert_array_equal(second['data'].reshape(2,2,16,2)[:,:,0,0],[[1,4],[2,5]])
        self.assertEqual(first['meta'][1],54136)
        self.assertEqual(second['meta'][1],54137)
        self.assertEqual(first['data'].size+second['data'].size,self.data.size)

    def test_rejects_out_of_range_shard(self):
        self.save()
        with self.assertRaises(ValueError):
            self.load(channel_start=2,channel_count=2)

    def test_explicit_cli_input_takes_precedence_over_environment(self):
        self.save()
        os.environ['EPIC_GULP_NPZ'] = str(self.path.parent/'does-not-exist.npz')
        result = self.load(input_path=str(self.path))
        self.assertEqual(result['data'].size,self.data.size)


if __name__ == '__main__':
    unittest.main()

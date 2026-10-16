import importlib,json,unittest
from pathlib import Path
import numpy as np
from source_functions import ROOT

class SkyGeometryTests(unittest.TestCase):
 def module(self):
  self.assertTrue((ROOT/'src/python/sky_geometry.py').is_file(),'sky-directed observation model missing')
  return importlib.import_module('sky_geometry')
 def config(self):
  return dict(input_start='2018-01-11T22:00:00',timezone='Asia/Shanghai',time_reference='first_voltage_sample',sequence_seconds=40e-6,source_ra_deg=83.63322,source_dec_deg=22.01446,site_lon_deg=117.493,site_lat_deg=36.656,site_height_m=125.,coordinate_frame='local_ENU',image_size=33,pixel_deg=.005,max_phase_drift_rad=.05)
 def test_explicit_beijing_time_converts_to_utc_and_naive_needs_timezone(self):
  m=self.module();self.assertEqual(m.parse_utc('2018-01-11T22:00:00','Asia/Shanghai').isot,'2018-01-11T14:00:00.000000000')
  with self.assertRaises(ValueError):m.parse_utc('2018-01-11T22:00:00',None)
 def test_center_is_crab_at_observation_time_not_zenith(self):
  m=self.module();model=m.SkyModel(self.config(),np.array([[0,0,0],[100,20,3]]),np.array([1.355e9]))
  dirs,center,meta=model.window(0,1000)
  self.assertGreater(np.rad2deg(np.arccos(center[2])),10.)
  np.testing.assert_allclose(dirs[16*33+16],center,atol=2e-12)
  np.testing.assert_allclose(np.linalg.norm(dirs,axis=1),1,atol=1e-12)
  self.assertEqual(meta['source_icrs_deg'],[83.63322,22.01446])
  self.assertIn('14:00:00.020',meta['midpoint_utc'])
 def test_reject_unknown_epoch_wrong_frame_and_coarse_pixels(self):
  m=self.module()
  for override in [dict(input_start=None),dict(time_reference='recorder_start'),dict(coordinate_frame='unknown'),dict(pixel_deg=.02)]:
   cfg=self.config()|override
   with self.subTest(override=override),self.assertRaises(ValueError):
    model=m.SkyModel(cfg,np.array([[0,0,0],[480,0,0]]),[1.355e9]);model.window(0,1000)
 def test_long_static_geometry_integration_rejected_and_time_changes(self):
  m=self.module();model=m.SkyModel(self.config(),[[0,0,0],[480,0,0]],[1.355e9])
  with self.assertRaisesRegex(ValueError,'window|drift'):model.window(0,10000)
  _,a,_=model.window(0,1000);_,b,_=model.window(50000,1000)
  self.assertGreater(np.linalg.norm(a-b),1e-5)
 def test_raw_phasing_recovers_nonzenith_source_at_center(self):
  m=self.module();pos=np.array([[0,0,0],[10,2,0],[2,25,7],[40,-10,5.]])
  freq=np.array([1.3e9,1.4e9]);model=m.SkyModel(self.config(),pos,freq)
  dirs,center,_=model.window(0,1000)
  raw=np.exp(-2j*np.pi*freq[:,None]*(pos@center)[None]/299792458.)
  tracking=m.raw_tracking(pos,freq,center)
  steering=np.exp(2j*np.pi*freq[:,None,None]*(pos@(dirs-center).T)[None]/299792458.)
  power=abs(np.einsum('cn,cnp->cp',raw*tracking,steering))**2
  np.testing.assert_allclose(power[:,16*33+16],16.,atol=1e-10)
  np.testing.assert_array_equal(np.argmax(power,axis=1),[16*33+16]*2)

 def test_header_precision_can_be_supplied_explicitly_without_truncation(self):
  m=self.module()
  self.assertEqual(m.parse_utc('2018-01-11T22:00:00.123456780','Asia/Shanghai').isot,'2018-01-11T14:00:00.123456780')
  with self.assertRaises(ValueError):m.parse_utc('2018-01-11T22:00:00+00:00','Asia/Shanghai')

 def test_dst_ambiguous_and_nonexistent_naive_times_rejected(self):
  m=self.module()
  for stamp in ['2021-11-07T01:30:00','2021-03-14T02:30:00']:
   with self.subTest(stamp=stamp),self.assertRaises(ValueError):m.parse_utc(stamp,'America/New_York')
  a=m.parse_utc('2021-11-07T01:30:00-04:00','America/New_York')
  b=m.parse_utc('2021-11-07T01:30:00-05:00','America/New_York')
  self.assertAlmostEqual((b-a).sec,3600.,places=8)

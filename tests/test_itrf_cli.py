import json
import os
import csv
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
import numpy as np
from source_functions import ROOT
import test_sky_geometry


class ITRFCLITests(unittest.TestCase):
    def build(self, path):
        script=ROOT/'scripts/build_itrf_geometry.py'
        self.assertTrue(script.is_file(),'ITRF geometry export CLI missing')
        command=[sys.executable,str(script),'--mapping',str(ROOT/'antenna_mapping_9178_full16.yaml'),
                 '--coordinates',str(ROOT/'antenna_coordinates.csv'),'--frequency-start-hz','1353400000',
                 '--channel-width-hz','25000','--channels','2','--output',str(path/'geometry.npz')]
        result=subprocess.run(command,capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)
        return command

    def test_export_preserves_order_frequency_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td);command=self.build(p)
            with np.load(p/'geometry.npz') as saved:
                self.assertEqual(saved['positions_m'].shape,(16,3))
                np.testing.assert_array_equal(saved['frequencies_hz'],[1353400000,1353425000])
                self.assertEqual(saved['antenna_names'][0],'北4')
                np.testing.assert_array_equal(saved['positions_m'][10:],np.zeros((6,3)))
                metadata=json.loads(str(saved['geometry_metadata_json']))
                self.assertEqual(metadata['coordinate_authority'],'ITRF_X/Y/Z')
                self.assertTrue(117. < metadata['site_lon_deg'] < 118.)
            original=(p/'geometry.npz').read_bytes()
            self.assertNotEqual(subprocess.run(command,capture_output=True).returncode,0)
            self.assertEqual((p/'geometry.npz').read_bytes(),original)

    def test_sky_runner_rejects_site_inconsistent_with_enu_reference_before_cuda(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)
            np.savez(p/'input.npz',data=np.ones((1,1,2,2),np.complex64))
            metadata=dict(coordinate_authority='ITRF_X/Y/Z',coordinate_frame='local_ENU',site_lon_deg=117.,site_lat_deg=36.,site_height_m=100.)
            np.savez(p/'geometry.npz',positions_m=[[0.,0,0],[1.,2,3]],frequencies_hz=[1.355e9],geometry_metadata_json=json.dumps(metadata))
            (p/'sky.json').write_text(json.dumps(test_sky_geometry.SkyGeometryTests().config()))
            result=subprocess.run([sys.executable,str(ROOT/'scripts/run_lband_fx.py'),'--input',str(p/'input.npz'),'--geometry',str(p/'geometry.npz'),'--sky-config',str(p/'sky.json'),'--output',str(p/'out'),'--coordinate-frame','local_ENU','--iq-convention','I+iQ','--voltage-convention','raw'],capture_output=True,text=True)
            self.assertNotEqual(result.returncode,0)
            self.assertIn('ENU reference',result.stderr)
            self.assertFalse((p/'out').exists())

    @unittest.skipUnless(os.environ.get('EPIC_FX_TEST_TWO_GPU')=='1','explicit two-GPU test required')
    def test_itrf_synthetic_raw_source_through_two_gpu_sky_runner(self):
        from sky_geometry import SkyModel
        with tempfile.TemporaryDirectory() as td:
            p=Path(td);self.build(p)
            with np.load(p/'geometry.npz') as g:
                pos=g['positions_m'];freq=g['frequencies_hz'];names=list(g['antenna_names'])
                meta=json.loads(str(g['geometry_metadata_json']))
            config=test_sky_geometry.SkyGeometryTests().config()
            for key in ('site_lon_deg','site_lat_deg','site_height_m'):config.pop(key)
            model=SkyModel(config|{k:meta[k] for k in ('site_lon_deg','site_lat_deg','site_height_m')},pos,freq)
            _,center,_=model.window(0,2)
            with (ROOT/'antenna_coordinates.csv').open() as stream:
                rows={r['名称']:r for r in csv.DictReader(stream)}
            xyz=np.array([[float(rows[n]['ITRF_'+axis]) for axis in 'XYZ'] for n in names[:10]])
            center_itrf=np.asarray(meta['itrf_to_enu_rotation']).T@center
            raw=np.exp(-2j*np.pi*freq[:,None]*((xyz-np.asarray(meta['reference_itrf_m']))@center_itrf)[None]/299792458.)
            voltage=np.zeros((2,2,16,2),np.complex64);voltage[:,:,:10]=raw[None,...,None]
            np.savez(p/'input.npz',data=voltage)
            (p/'sky.json').write_text(json.dumps(config))
            result=subprocess.run([sys.executable,str(ROOT/'scripts/run_lband_fx.py'),'--input',str(p/'input.npz'),'--geometry',str(p/'geometry.npz'),'--sky-config',str(p/'sky.json'),'--output',str(p/'out'),'--devices','0,1','--channel-tile','1','--window-sequences','2','--coordinate-frame','local_ENU','--iq-convention','I+iQ','--voltage-convention','raw','--remove-autocorr'],capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
            manifest=json.loads((p/'out/manifest.json').read_text())
            self.assertEqual({c['device'] for c in manifest['chunks']},{0,1})
            for chunk in manifest['chunks']:
                with np.load(p/'out'/chunk['file']) as output:
                    self.assertEqual(int(np.argmax(output['image'][0,:,0])),16*33+16)
                    np.testing.assert_allclose(output['image'][0,16*33+16],[90,90,90,0],atol=1e-4)
                    saved=json.loads(str(output['metadata_json']))
                    self.assertEqual(saved['sky_config']['site_lat_deg'],meta['site_lat_deg'])

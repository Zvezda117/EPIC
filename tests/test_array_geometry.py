import csv
import importlib
import tempfile
import unittest
from pathlib import Path
import numpy as np
from source_functions import ROOT, epic_functions


class ITRFGeometryTests(unittest.TestCase):
    def api(self):
        self.assertTrue((ROOT/'src/python/array_geometry.py').is_file(), 'authoritative ITRF loader missing')
        return importlib.import_module('array_geometry')

    def test_equator_axes_and_preservation_of_baselines_and_delays(self):
        # At lon=lat=0, E=+Y, N=+Z, U=+X; this catches transpose/sign mistakes.
        m = self.api()
        xyz = np.array([[6378137.,0,0],[6378138.,2,3],[6378133.,-5,6]])
        g = m.itrf_to_enu(xyz, xyz[0])
        expected = [[0,0,0],[2,3,1],[-5,6,-4]]
        np.testing.assert_allclose(g['positions_m'], expected, atol=1e-9)
        s_itrf = np.array([.8,0,.6]); s_enu = np.array([0,.6,.8])
        np.testing.assert_allclose((xyz-xyz[0])@s_itrf, g['positions_m']@s_enu, atol=1e-9)
        np.testing.assert_allclose(np.linalg.norm(xyz-xyz[1],axis=1), np.linalg.norm(g['positions_m']-g['positions_m'][1],axis=1), atol=1e-9)

    def test_authority_ignores_legacy_columns_and_padding_does_not_move_site(self):
        m = self.api()
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'coords.csv'
            p.write_text('名称,ITRF_X,ITRF_Y,ITRF_Z,longitude,latitude,X/E\nA,6378137,0,0,999,bad,bad\nB,6378137,2,0,bad,bad,bad\n')
            a=m.load_itrf_array(p,['B','PAD01','A']); b=m.load_itrf_array(p,['A'])
            np.testing.assert_allclose(a['reference_itrf_m'],[6378137,1,0],atol=1e-9)
            np.testing.assert_array_equal(a['reference_itrf_m'],b['reference_itrf_m'])
            np.testing.assert_array_equal(a['positions_m'][1],[0,0,0])
            self.assertAlmostEqual(np.linalg.norm(a['positions_m'][0]-a['positions_m'][2]),2.)
            for names in (['missing'],['A','A']):
                with self.assertRaises(ValueError):m.load_itrf_array(p,names)

    def test_real_array_baselines_and_epic_entry_use_same_itrf_geometry(self):
        m=self.api()
        import os
        from unittest.mock import patch
        names=['北4','北3','东4','东3','东5','东6','东9','东10','东7','东8']+[f'PAD{i:02d}' for i in range(1,7)]
        g=m.load_itrf_array(ROOT/'antenna_coordinates.csv',names)
        with (ROOT/'antenna_coordinates.csv').open() as stream:
            rows=list(csv.DictReader(stream))
        byname={r['名称']:np.array([float(r['ITRF_'+k]) for k in 'XYZ']) for r in rows}
        for i in range(10):
            for j in range(i):
                self.assertAlmostEqual(np.linalg.norm(g['positions_m'][i]-g['positions_m'][j]),np.linalg.norm(byname[names[i]]-byname[names[j]]),places=8)
        f=epic_functions('_candidate_repo_paths','_find_repo_file','_read_simple_yaml','_load_coordinate_rows','_load_custom_array','_station_lon_lat_deg')
        with patch.dict(os.environ,{'EPIC_ARRAY_CONFIG':str(ROOT/'antenna_mapping_9178_full16.yaml'),'EPIC_ANTENNA_COORDS':str(ROOT/'antenna_coordinates.csv')}):
            actual=f['_load_custom_array'](16)
            np.testing.assert_allclose(actual['positions'],g['positions_m'],atol=1e-9)
            np.testing.assert_allclose(f['_station_lon_lat_deg'](),[g['site_lon_deg'],g['site_lat_deg']],atol=1e-12)

    def test_reject_missing_nonfinite_duplicate_or_unphysical_itrf(self):
        m=self.api()
        for xyz in ([[0.,0,0]],[[6378137,np.nan,0]]):
            with self.assertRaises(ValueError):m.itrf_to_enu(xyz)
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'c.csv'
            for content in ('名称,X/E,Y/N,Z/U\nA,1,2,3\n','名称,ITRF_X,ITRF_Y,ITRF_Z\nA,6378137,0,0\nA,6378137,1,0\n'):
                p.write_text(content)
                with self.assertRaises(ValueError):m.load_itrf_array(p,['A'])

    def test_legacy_position_readers_follow_itrf_authority(self):
        import ast
        from types import SimpleNamespace
        from unittest.mock import patch
        import os
        cases=[('process_crab_data.py','CrabDataProcessor','get_antenna_positions'),
               ('process_9197_single_channel.py','SimpleDataProcessor','get_antenna_positions')]
        names=['北3','北4','北5','北6']
        expected=self.api().load_itrf_array(ROOT/'antenna_coordinates.csv',names)['positions_m']
        for filename,classname,method in cases:
            tree=ast.parse((ROOT/'scripts'/filename).read_text())
            cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name==classname)
            func=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name==method)
            scope=dict(np=np,os=os,Path=Path)
            exec(compile(ast.Module(body=[func],type_ignores=[]),filename,'exec'),scope)
            obj=SimpleNamespace(nstands=4,antenna_index=dict(zip(names,range(4))),config={'antenna_to_epic_index':dict(zip(names,range(4)))})
            previous=os.getcwd()
            try:
                os.chdir(ROOT)
                actual=scope[method](obj)
            finally:os.chdir(previous)
            np.testing.assert_allclose(actual,expected,atol=1e-9,err_msg=filename)

    def test_legacy_voltage_and_geometry_order_agree_for_swapped_indices(self):
        import ast, os
        from types import SimpleNamespace, MethodType
        from contextlib import redirect_stdout
        from io import StringIO
        path=ROOT/'scripts/process_9197_single_channel.py'
        tree=ast.parse(path.read_text())
        cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='SimpleDataProcessor')
        methods=[n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name in ('process_all_antennas','get_antenna_positions')]
        scope=dict(np=np,os=os,Path=Path)
        exec(compile(ast.Module(body=methods,type_ignores=[]),str(path),'exec'),scope)
        names=['北4','北3','北5','北6']
        obj=SimpleNamespace(nstands=4,config={'antenna_to_epic_index':dict(zip(names,range(4)))})
        # Replace only expensive raw-file I/O with four distinguishable voltage fixtures.
        values={'HBaseband2Spec2':3,'VBaseband0Spec0':4,'VBaseband1Spec1':5,'VBaseband2Spec2':6}
        obj.process_antenna_group=lambda pattern,*args:np.full(2,values[pattern],np.complex64)
        obj.get_antenna_positions=MethodType(scope['get_antenna_positions'],obj)
        previous=os.getcwd()
        try:
            os.chdir(ROOT)
            with tempfile.TemporaryDirectory() as td,redirect_stdout(StringIO()):
                voltage,positions=scope['process_all_antennas'](obj,output_file=str(Path(td)/'out.npz'))
        finally:os.chdir(previous)
        np.testing.assert_array_equal(voltage[0,0,:,0],[4,3,5,6])
        np.testing.assert_allclose(positions,self.api().load_itrf_array(ROOT/'antenna_coordinates.csv',names)['positions_m'],atol=1e-9)

import importlib,sys,tempfile,unittest
from pathlib import Path
import numpy as np
from source_functions import ROOT
sys.path.insert(0,str(ROOT/'scripts'))
import convert_9178_to_epic as converter

class Packet9178Tests(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)
 def raw(self,name,counters,alter=None,tail=b''):
  a=np.zeros((len(counters),262400),np.uint8);h=a[:,:256].copy().view('<u4').reshape(-1,64)
  h[:,0]=0x18efdc01;h[:,1]=counters;h[:,2]=262400;h[:,3]=(32<<16)|0x11;h[:,4]=2;h[:,6]=13
  h[:,60]=0x260111;h[:,61]=0x142927;h[:,62]=np.arange(len(counters))*26214
  if alter:alter(h)
  a[:,:256]=h.view(np.uint8).reshape(-1,256)
  p=self.root/name;p.write_bytes(a.tobytes()+tail);return p
 def api(self):
  self.assertTrue((ROOT/'src/python/packet9178.py').is_file(),'packet integrity module missing')
  return importlib.import_module('packet9178')
 def test_counter_wrap_is_contiguous(self):
  a=self.api().audit_file(self.raw('a.data',[0xfffffffe,0xffffffff,0,1]))
  self.assertEqual(a['packets'],4);self.assertEqual(a['first_counter'],0xfffffffe)
 def test_reject_gap_duplicate_and_later_bad_header(self):
  api=self.api()
  for seq in ([1,3],[1,1],[2,1]):
   with self.subTest(seq=seq),self.assertRaises(ValueError):api.audit_file(self.raw('a.data',seq))
  p=self.raw('bad.data',[1,2],lambda h:h.__setitem__((1,0),0))
  with self.assertRaises(ValueError):api.audit_file(p)
 def test_reject_partial_packet(self):
  with self.assertRaises(ValueError):self.api().audit_file(self.raw('a.data',[1],tail=b'bad'))
 def test_cross_terminal_mismatch_rejected(self):
  api=self.api();a=self.raw('a.data',[3,4]);b=self.raw('b.data',[4,5])
  with self.assertRaises(ValueError):api.audit_group([a,b])
 def test_clock_is_evidence_without_assumed_timezone(self):
  a=self.api().audit_file(self.raw('a.data',[1,2]))
  self.assertEqual(a['clock_candidate']['first_local'],'2026-01-11T14:29:27.000000000')
  self.assertFalse(a['clock_candidate']['sample_epoch_confirmed'])
  self.assertIsNone(a['clock_candidate']['timezone'])
 def test_converter_rejects_gap_in_requested_voltage_window(self):
  p=self.raw('a.data',[1,3])
  with self.assertRaisesRegex(ValueError,'counter|contigu'):
   converter.read_terminal_channel(p,0,65537)
 def test_chunk_epoch_offsets_include_packet_and_sequence_offsets(self):
  api=self.api()
  self.assertAlmostEqual(api.chunk_epoch(1700000000.,3,10000),1700000000.+3*65536/250e6+.4,places=6)
  self.assertEqual(api.chunk_epoch(0.,3,10000),0.)
  with self.assertRaises(ValueError):api.chunk_epoch(-1,0,0)

 def test_converter_build_offsets_actual_output_epoch(self):
  from argparse import Namespace
  from contextlib import redirect_stdout
  from io import StringIO
  for name in {x.filename for x in converter.FULL10_INPUTS}:self.raw(name,[0,1,2])
  args=Namespace(data_dir=self.root,integration_ms=.08,start_seq=1,start_packet=1,
                 array_mode='full16',channels=4,iq_order='IQ',channel_offset=0,
                 dc_mask_chans=0,quantize_cnib=False,center_freq_mhz=1355.,tstart=1700000000.)
  with redirect_stdout(StringIO()):data,meta=converter.build_epic_gulp(args)
  self.assertEqual(data.shape,(2,4,16,2))
  self.assertAlmostEqual(meta[0],1700000000.+65536/250e6+40e-6,places=6)
  self.assertTrue(args.packet_audit['packet_counter_ranges_match'])

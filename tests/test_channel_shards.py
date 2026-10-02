import subprocess
import tempfile
import unittest
from pathlib import Path
from source_functions import ROOT


class ChannelShardTests(unittest.TestCase):
    def compile_and_run(self, program):
        with tempfile.TemporaryDirectory() as td:
            src, exe = Path(td)/'test.cpp', Path(td)/'test'
            src.write_text(program)
            build=subprocess.run(['g++','-std=c++17','-I'+str(ROOT/'src/ex'),str(src),'-o',str(exe)],capture_output=True,text=True)
            self.assertEqual(build.returncode,0,build.stderr)
            result=subprocess.run([str(exe)],capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)

    def test_two_devices_cover_each_channel_once_and_reject_bad_partitions(self):
        program = r'''
#include "offline_config.hpp"
#include <cassert>
#include <stdexcept>
int main() {
  auto a=OfflineChannelSlice(128,2,0,4,8);
  auto b=OfflineChannelSlice(128,2,1,4,8);
  assert(a.first==0 && a.count==64 && b.first==64 && b.count==64);
  assert(a.first+a.count==b.first && b.first+b.count==128);
  for(int bad=0;bad<5;++bad) {
    try {
      if(bad==0) OfflineChannelSlice(130,2,0,4,8);
      if(bad==1) OfflineChannelSlice(128,2,2,4,8);
      if(bad==2) OfflineChannelSlice(128,0,0,4,8);
      if(bad==3) OfflineChannelSlice(128,2,0,0,8);
      if(bad==4) OfflineChannelSlice(128,2,0,4,0);
      return bad+1;
    } catch(const std::invalid_argument&) {}
  }
  return 0;
}
'''
        self.compile_and_run(program)

    def test_rejects_nonintegral_device_or_stream_shards_and_bad_bins(self):
        self.compile_and_run(r'''
#include "offline_config.hpp"
#include <cassert>
#include <stdexcept>
int main() {
  for(int bad=0;bad<6;++bad) {
    try {
      if(bad==0) OfflineChannelSlice(129,2,0,1,1);
      if(bad==1) OfflineChannelSlice(128,2,0,4,3);
      if(bad==2) OfflineChannelSlice(0,2,0,4,8);
      if(bad==3) OfflineChannelSlice(128,2,-1,4,8);
      if(bad==4) OfflineChannelSlice(128,2,0,-1,8);
      if(bad==5) CheckedChannelBins(64,0);
      return bad+1;
    } catch(const std::invalid_argument&) {}
  }
  assert(CheckedChannelBins(64,4)==16);
  assert(OfflinePackedBytes(10000,64,16,2)==20480000);
  assert(OfflinePackedBytes(1,2147483647,1,1)==2147483647);
  try { OfflinePackedBytes(2,1073741824,1,1); return 9; }
  catch(const std::overflow_error&) {}
}
''')

    def test_utc_time_tag_requires_positive_known_time_and_bounds_rounding(self):
        self.compile_and_run(r'''
#include "offline_config.hpp"
#include <cassert>
#include <cmath>
#include <limits>
#include <stdexcept>
int main() {
  assert(OfflineTimeTag(1.5,196000000.)==294000000);
  assert(OfflineTimeTag(1.125,4.)==5);
  for(double bad : {0.,-1.,std::numeric_limits<double>::infinity(),
                    std::numeric_limits<double>::quiet_NaN(),1.e20}) {
    try { OfflineTimeTag(bad,196000000.); return 1; }
    catch(const std::invalid_argument&) {}
  }
}
''')

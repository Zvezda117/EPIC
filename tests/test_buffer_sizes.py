import subprocess
import tempfile
import unittest
from pathlib import Path

from source_functions import ROOT


class BufferSizeTests(unittest.TestCase):
    def test_descriptor_allocates_all_output_pixels_and_rejects_invalid_shape(self):
        source = r'''
#include <cuda_runtime.h>
#include "types.hpp"
#include <cassert>
#include <stdexcept>
int main() {
  MOFFCorrelatorDesc d;
  d.img_size=FULL; d.nchan_out=1024;
  assert(d.OutputBufferElements()==67108864);
  d.img_size=HALF; d.nchan_out=128;
  assert(d.OutputBufferElements()==2097152);
  d.nchan_out=0;
  try { d.OutputBufferElements(); return 1; } catch(const std::invalid_argument&) {}
  d.nchan_out=-1;
  try { d.OutputBufferElements(); return 2; } catch(const std::invalid_argument&) {}
  return 0;
}
'''
        with tempfile.TemporaryDirectory() as tmp:
            cpp, exe = Path(tmp)/'size.cpp', Path(tmp)/'size'
            cpp.write_text(source)
            build = subprocess.run(['g++','-std=c++17','-I/usr/local/cuda-12.1/include',
                                    '-I'+str(ROOT/'src/ex'),str(cpp),'-o',str(exe)],capture_output=True,text=True)
            self.assertEqual(build.returncode,0,build.stderr)
            run = subprocess.run([str(exe)],capture_output=True,text=True)
            self.assertEqual(run.returncode,0,run.stdout+run.stderr)

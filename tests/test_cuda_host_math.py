"""Exercise the production half arithmetic on CPU without requiring a GPU.

The four complete function definitions are extracted unchanged: including the
whole cu_helpers header brings in CUDA registration/device initialization.
"""
import re
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from source_functions import ROOT


class CudaHostMathTests(unittest.TestCase):
    def test_linear_products_and_common_phase_invariance(self):
        compiler = shutil.which('g++')
        include = Path('/usr/local/cuda-12.1/include')
        if not compiler or not (include / 'cuda_fp16.h').exists():
            self.skipTest('C++ compiler and CUDA half headers required')
        source = (ROOT / 'src/ex/cu_helpers.cuh').read_text()
        definitions = []
        for name in ['ComputeXX', 'ComputeYY', 'ComputeUU', 'ComputeVV']:
            match = re.search(r'\bT\s+' + name + r'\(typename FFT::value_type\s+pix\)\s*\{.*?\n\}', source, re.S)
            self.assertIsNotNone(match, name)
            definitions.append('template<class T, class FFT> ' + match.group())
        program = '#include <cuda_fp16.h>\n#include <cmath>\n#include <cstdio>\n'
        program += '\n'.join(definitions)
        program += r'''
struct F { struct value_type { __half2 x,y; }; };
int main() {
  struct Case { float xr,xi,yr,yi,xx,yy,re,im; };
  const Case cases[] = {{1,0,1,0,1,1,1,0}, {1,2,3,4,5,25,11,2},
                        {-2,1,-4,3,5,25,11,2}, {256,0,0,0,65536,0,0,0}};
  for (const auto &v: cases) {
    F::value_type p{__floats2half2_rn(v.xr,v.yr),__floats2half2_rn(v.xi,v.yi)};
    float got[]={ComputeXX<float,F>(p),ComputeYY<float,F>(p),ComputeUU<float,F>(p),ComputeVV<float,F>(p)};
    float want[]={v.xx,v.yy,v.re,v.im};
    for(int i=0;i<4;++i) if(got[i]!=want[i]) {
      printf("product %d: got=%g expected=%g\n",i,got[i],want[i]);return 1;
    }
  }
  return 0;
}
'''
        with tempfile.TemporaryDirectory() as tmp:
            cpp, exe = Path(tmp)/'math.cpp', Path(tmp)/'math'
            cpp.write_text(program)
            build = subprocess.run([compiler, '-std=c++17', '-I'+str(include), str(cpp), '-o', str(exe)], capture_output=True, text=True)
            self.assertEqual(build.returncode, 0, build.stderr)
            run = subprocess.run([str(exe)], capture_output=True, text=True)
            self.assertEqual(run.returncode, 0, run.stdout + run.stderr)

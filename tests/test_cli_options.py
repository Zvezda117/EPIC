"""Run the actual CLI parser/validator with only GPU discovery replaced.

No production CUDA kernel, metrics service or graph is started. Option parser
functions are compiled from their actual source with the real cxxopts header.
"""
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

from source_functions import ROOT


class CLIOptionsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.tmp.cleanup)
        source = (ROOT / 'src/ex/option_parser.hpp').read_text()
        start = source.index('cxxopts::Options GetEpicOptions()')
        stop = source.index('\n#endif', start)
        functions = source[start:stop]
        program = r'''
#include <cxxopts.hpp>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <initializer_list>
#include <iterator>
#include <optional>
#include <set>
#include <sstream>
#include <string>
#include <vector>
#include "constants.h"
#if __has_include("offline_config.hpp")
#include "offline_config.hpp"
#endif
constexpr int cudaSuccess=0;
int cudaGetDeviceCount(int* count) {
  const char* mode=std::getenv("GPU_CLI_TEST_MODE");
  if(mode && std::string(mode)=="error") return 1;
  *count=(mode && std::string(mode)=="none") ? 0 : 2;
  return cudaSuccess;
}
int GetNumGpus() { int count=0; cudaGetDeviceCount(&count); return count; }
const char* git_CommitSHA1() { return "test-build"; }
std::string metric_gpu_ids;
struct PrometheusExporter {
  static void AddInfoLabels(std::initializer_list<std::pair<std::string,std::string>> labels) {
    for(const auto& item : labels)
      if(item.first=="gpu_ids") metric_gpu_ids=item.second;
  }
};
''' + functions + r'''
int main(int argc,char** argv) {
  try {
    auto parser=GetEpicOptions();
    auto result=parser.parse(argc,argv);
    auto error=ValidateOptions(result);
    if(error) { std::cerr<<*error<<"\n"; return 2; }
    if(result.count("help")) std::cout<<parser.help();
    else if(!result["disable_metrics"].as<bool>()) {
      InitInfoMetric(result);
      std::cout<<"metric_gpu_ids="<<metric_gpu_ids<<"\n";
    }
    return 0;
  } catch(const std::exception& e) { std::cerr<<e.what()<<"\n"; return 3; }
}
'''
        cpp = Path(cls.tmp.name) / 'cli.cpp'
        cls.exe = Path(cls.tmp.name) / 'cli'
        cpp.write_text(program)
        build = subprocess.run([
            'g++', '-std=c++17', '-I'+str(ROOT/'src/ex'),
            '-I'+str(ROOT/'src/extern/cxxopts/include'), str(cpp), '-o', str(cls.exe)
        ], capture_output=True, text=True)
        if build.returncode:
            raise AssertionError(build.stderr)

    def run_cli(self, *args, mode='two'):
        env = dict(os.environ, GPU_CLI_TEST_MODE=mode)
        return subprocess.run([str(self.exe), *args], env=env,
                              capture_output=True, text=True)

    def base_args(self):
        return ['--offline', '--channels=128', '--ngpus=2', '--aeff=25,25',
                '--nts=1000', '--seq_accum=40', '--nstreams=8']

    def test_help_succeeds_without_cuda_devices_or_when_discovery_fails(self):
        for mode in ['none', 'error']:
            with self.subTest(mode=mode):
                result = self.run_cli('--help', mode=mode)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn('--offline', result.stdout)

    def test_valid_two_gpu_partition_and_short_offline_gulp(self):
        for extra in [[], ['--nts=25', '--seq_accum=1']]:
            result = self.run_cli(*(self.base_args()+extra))
            self.assertEqual(result.returncode, 0, result.stderr)

    def test_bad_parameters_return_errors_without_arithmetic_signals(self):
        for extra in [
            ['--chan_nbin=0'], ['--chan_nbin=-1'], ['--kernel_oversample=0'],
            ['--kernel_oversample=-2'], ['--gpu_ids=0'], ['--gpu_ids=0,0'],
            ['--gpu_ids=-1,0'], ['--gpu_ids=0,2'], ['--channels=129'],
            ['--channels=130'], ['--nstreams=3'], ['--nstreams=0'],
            ['--nstreams=9'], ['--nts=0'], ['--nimg_accum=2'],
            ['--nts=2147483647'], ['--npytbnfile='],
            ['--support=8', '--kernel_oversample=8'],
            ['--kernel_oversample=1073741824'],
        ]:
            with self.subTest(extra=extra):
                result = self.run_cli(*(self.base_args()+extra))
                self.assertGreater(result.returncode, 0, result.stderr)
                self.assertTrue(result.stderr)

    def test_cuda_failure_is_reported_as_no_available_devices(self):
        result = self.run_cli(*self.base_args(), mode='error')
        self.assertGreater(result.returncode, 0, result.stderr)
        self.assertIn('GPU', result.stderr.upper())

    def test_device_order_can_be_reversed_and_per_gpu_capacity_is_validated(self):
        result = self.run_cli(*(self.base_args()+['--gpu_ids=1,0', '--channels=2048']))
        self.assertEqual(result.returncode, 0, result.stderr)
        bad = self.run_cli(*(self.base_args()+['--channels=4096']))
        self.assertGreater(bad.returncode, 0, bad.stderr)

    def test_default_gpu_ids_do_not_crash_metrics_and_explicit_order_is_preserved(self):
        for extra, expected in [([], '0,1'), (['--gpu_ids=1,0'], '1,0')]:
            with self.subTest(extra=extra):
                result = self.run_cli(*(self.base_args()+extra))
                self.assertEqual(result.returncode, 0, result.stderr)
                value = result.stdout.strip().split('metric_gpu_ids=', 1)[1]
                self.assertEqual(value.rstrip(','), expected)

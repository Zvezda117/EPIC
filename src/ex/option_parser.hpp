/*
 Copyright (c) 2023 The EPIC++ authors

 Permission is hereby granted, free of charge, to any person obtaining a copy of
 this software and associated documentation files (the "Software"), to deal in
 the Software without restriction, including without limitation the rights to
 use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
 of the Software, and to permit persons to whom the Software is furnished to do
 so, subject to the following conditions:

 The above copyright notice and this permission notice shall be included in all
 copies or substantial portions of the Software.

 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 SOFTWARE.
 */

#ifndef SRC_EX_OPTION_PARSER_HPP_
#define SRC_EX_OPTION_PARSER_HPP_
#include <cxxopts.hpp>
#include <cmath>
#include <iostream>
#include <iterator>
#include <optional>
#include <set>
#include <string>
#include <sstream>
#include <vector>

#include "./constants.h"
#include "./metrics.hpp"
#include "./host_helpers.h"
#include "./offline_config.hpp"
#include "git.h"

cxxopts::Options GetEpicOptions() {
  cxxopts::Options options("epic++", "EPIC dual-pol imager");

  options.add_options("Online data processing")(
      "addr", "F-Engine UDP Stream Address",
      cxxopts::value<std::vector<std::string>>()->default_value(
          "239.168.40.11,239.168.40.12"))(
      "port", "F-Engine UDP Stream Port",
      cxxopts::value<std::vector<int>>()->default_value("4015,4015"))(
      "printendpoints",
      "Print the IP/port values and their channels for each endpoint in the "
      "LWA-SV station and exit",
      cxxopts::value<bool>()->default_value("false"))(
      "runtime", "EPIC run duration. Provide -1 to run continuously",
      cxxopts::value<int>()->default_value("5"))
      ("include_chans","Image only certain frequencies. Specify the value(s) of chan0.", cxxopts::value<std::vector<int>>());
  // ("utcstart", "F-Engine UDP Stream Start Time")

  options.add_options("Offline data processing")(
      "offline", "Load numpy-TBN data from disk",
      cxxopts::value<bool>()->default_value("false"))(
      "npytbnfile", "numpy-TBN data path", cxxopts::value<std::string>());

  options.add_options("Imaging options")(
      "remove_autocorr", "Subtract physical-antenna self correlations from all four products",
      cxxopts::value<bool>()->default_value("false"))(
      "imagesize", "1-D image size (can only be 64 or 128)",
      cxxopts::value<int>()->default_value("128"))(
      "imageres", "Pixel resolution in degrees",
      cxxopts::value<float>()->default_value("1.0"))(
      "nts", "Number of timestamps per span",
      cxxopts::value<int>()->default_value("1000"))(
      "seq_accum", "Online accumulation duration in milliseconds; offline integrates one complete NPZ gulp",
      cxxopts::value<int>()->default_value("40"))(
      "nimg_accum", "Number of images to accumulate before saving to the disk",
      cxxopts::value<int>()->default_value("1"))(
      "channels", "Input channels: offline total split across GPUs; online per instance; bandwidth = channels x 25 kHz",
      cxxopts::value<int>()->default_value("1024"))(
      "support", "Support size of the kernel (1 through 9)",
      cxxopts::value<int>()->default_value("2"))(
      "aeff", "Antenna effective area (experimental) in sq. m",
      cxxopts::value<std::vector<float>>()->default_value("25,16"))(
      "kernel_oversample",
      "Factor to over sample the kernel. Must be a power of 2.",
      cxxopts::value<int>()->default_value("2"))(
      "accum_16bit",
      "Use 16-bit precision for on-chip memory accumulation. Faster but less "
      "precise.",
      cxxopts::value<bool>()->default_value("false"))(
      "chan_nbin", "Binning factor for the number of channels",
      cxxopts::value<int>()->default_value("4"))
      ("out_dir","Output directory to store the long accumulation files. Defaults to the current directory.",cxxopts::value<std::string>()->default_value("./"));

  options.add_options("Execution options")(
      "nstreams", "Number of cuda streams to process images",
      cxxopts::value<int>()->default_value("8"))(
      "ngpus", "number of GPUs to simultaneously run EPIC",
      cxxopts::value<int>()->default_value("1"))(
      "gpu_ids",
      "GPU ID for each EPIC instance. Defaults to automatic assignment",
      cxxopts::value<std::vector<int>>());

  options.add_options("Metrics collection")("h,help", "Print usage")(
      "metrics_bind_addr",
      "Address to expose Prometheus metrics. Defaults to 127.0.0.1:8080",
      cxxopts::value<std::string>()->default_value("127.0.0.1:8080"))(
      "disable_metrics", "Disable writing metrics to the Prometheus endpoint",
      cxxopts::value<bool>()->default_value("false"));

  options.add_options("Database Operations")(
      "elev_limit_deg",
      "Extract pixels into the DB only if the source is above this elevation "
      "(degrees). "
      "Defaults to 10 deg",
      cxxopts::value<float>()->default_value("10.0"))(
      "watchdog_addr", "Address for the EPIC watchdog services.",
      cxxopts::value<std::string>()->default_value("localhost:2023"))
      ("epic_data_schema","Schema name for all epic data tables.",
      cxxopts::value<std::string>()->default_value("public"));

  options.add_options("Live Streaming")(
      "video_size", "Stream video dimensions. Defaults to 512",
      cxxopts::value<int>()->default_value("512"))(
      "stream_url", "RTMP endpoint to stream the video to",
      cxxopts::value<std::string>()->default_value(
          "rtmp://127.0.0.1:1857/live/epic"))(
            "stream_cmap", "Colormap for the output video stream. See https://ffmpeg.org/ffmpeg-filters.html#pseudocolor for available ones.",
            cxxopts::value<std::string>()->default_value("magma")
          );

  return options;
}

std::optional<std::string> ValidateOptions(const cxxopts::ParseResult& result) {
  using std::string_literals::operator""s;
  // RunEpic validates before handling --help. No CUDA discovery is needed to
  // print usage, including on hosts where CUDA initialization fails.
  if (result.count("help")) return {};
  const bool offline = result["offline"].as<bool>();
  if (result.count("npytbnfile") && result["npytbnfile"].as<std::string>().empty())
    return "npytbnfile must name a non-empty input path";
  auto ports = result["port"].as<std::vector<int>>();
  for (auto& port : ports) {
    if (port < 1 || port > 32768) {
      return "Invalid port number:  "s + std::to_string(port) +
             ". Port must be a number in 1-32768"s;
    }
  }

  int image_size = result["imagesize"].as<int>();
  if (!(image_size == 64 || image_size == 128)) {
    return "Invalid image size: "s + std::to_string(image_size) +
           ". Image size can only be 64 or 128"s;
  }

  int nts = result["nts"].as<int>();
  int accumulate = result["seq_accum"].as<int>();
  if (nts <= 0) return "nts must be greater than 0";
  if (nts > std::numeric_limits<int>::max() / SAMPLING_LEN_uS)
    return "nts exceeds the integer timing range; use a smaller time tile";
  if (accumulate <= 0) return "seq_accum must be greater than 0";
  const double gulp_ms = nts * (SAMPLING_LEN_uS * 1e-3);
  if (!offline && accumulate < gulp_ms) {
    return "Online sequence accumulation must cover a complete input gulp";
  }
  const int image_accum = result["nimg_accum"].as<int>();
  if (image_accum <= 0) return "nimg_accum must be greater than 0";
  if (offline && image_accum != 1)
    return "Offline processes one NPZ gulp; nimg_accum must be 1";
  const float resolution = result["imageres"].as<float>();
  if (!std::isfinite(resolution) || resolution <= 0)
    return "imageres must be positive and finite";

  int channels = result["channels"].as<int>();
  if (channels <= 0) {
    return "The number of output channels must be at least 1"s;
  }

  int ndevices = result["ngpus"].as<int>();

  if (ndevices <= 0) {
    return "ngpus must be greater than 0";
  }
  if (ndevices > 3) return "This executor supports at most 3 GPUs";

  auto aeffs = result["aeff"].as<std::vector<float>>();
  if (aeffs.size() < ndevices) {
    return "Effective area must be specified for each frequency window";
  }

  for (auto aeff : aeffs) {
    if (!std::isfinite(aeff) || aeff <= 0) {
      return "Antenna effective cannot be smaller than or equal to zero:  "s +
             std::to_string(aeff);
    }
  }

  int support = result["support"].as<int>();
  if (support <= 0 || support > MAX_ALLOWED_SUPPORT_SIZE) {
    return "Invalid support size: "s + std::to_string(support) +
           ". Support can only be between 1-"s +
           std::to_string(MAX_ALLOWED_SUPPORT_SIZE);
  }

  int kos = result["kernel_oversample"].as<int>();
  if (kos <= 0 || (kos & (kos - 1)) != 0) {
    return "Kernel oversampling factor must be a positive power of 2";
  }
  // ResetGcfElem assigns one thread per oversampled support pixel and
  // supports at most a 32x32 footprint. Divide first to avoid integer overflow.
  if (kos > 32 / support)
    return "support * kernel_oversample must not exceed 32";

  int nbin = result["chan_nbin"].as<int>();
  int nstreams = result["nstreams"].as<int>();
  if (nstreams <= 0 || nstreams > MAX_GULP_STREAMS) {
    return "nstreams must be between 1 and "s + std::to_string(MAX_GULP_STREAMS);
  }
  int per_device_channels = channels;
  try {
    if (offline) {
      per_device_channels =
          OfflineChannelSlice(channels, ndevices, 0, nbin, nstreams).count;
    } else {
      CheckedChannelBins(channels, nbin);
      if (channels % nstreams != 0)
        return "Each GPU channel count must be divisible by nstreams";
    }
    if (offline)
      OfflinePackedBytes(nts, per_device_channels, LWA_SV_NSTANDS, LWA_SV_NPOLS);
  } catch (const std::exception& error) {
    return error.what();
  }
  if (per_device_channels > MAX_CHANNELS_4090) {
    return "Per-GPU input channels exceed the supported limit of "s +
           std::to_string(MAX_CHANNELS_4090);
  }

  // cudaGetDeviceCount may fail without writing its output. Initialize once and
  // check the result instead of reading the legacy GetNumGpus uninitialized int.
  int available_gpus = 0;
  if (cudaGetDeviceCount(&available_gpus) != cudaSuccess) available_gpus = 0;
  if (ndevices > available_gpus) {
    return "Requested "s + std::to_string(ndevices) + " GPUs but only "s +
           std::to_string(available_gpus) + " available"s;
  }
  if (result.count("gpu_ids") > 0) {
    auto gpu_ids = result["gpu_ids"].as<std::vector<int>>();
    if (gpu_ids.size() != size_t(ndevices))
      return "A GPU ID must be specified for each EPIC instance";
    std::set<int> distinct;
    for (int id : gpu_ids) {
      if (id < 0 || id >= available_gpus)
        return "GPU IDs must reference available non-negative devices";
      if (!distinct.insert(id).second)
        return "GPU IDs must be distinct";
    }
  }

  const float elevation = result["elev_limit_deg"].as<float>();
  if (!std::isfinite(elevation) || elevation < 0 || elevation > 90) {
    return "Elevation limit must be between 0 and 90 degrees";
  }

  return {};
}

template<typename T>
std::string VJoin(std::vector<T> inp_vec, const char* delimiter=","){
  if(inp_vec.size()==0){
    return "";
  }

  if(inp_vec.size()==1){
    return std::to_string(inp_vec[0]);
  }
  std::stringstream result;
  std::copy(inp_vec.begin(), inp_vec.end(), std::ostream_iterator<T>(result, delimiter));
  return result.str();
}

template<>
std::string VJoin<>(std::vector<std::string> inp_vec, const char* delimiter){
  if(inp_vec.size()==0){
    return "";
  }

  if(inp_vec.size()==1){
    return inp_vec[0];
  }
  std::stringstream result;
  std::copy(inp_vec.begin(), inp_vec.end(), std::ostream_iterator<std::string>(result, delimiter));
  return result.str();
}

void InitInfoMetric(const cxxopts::ParseResult& result){
//git_CommitSHA1()
  using sVec = typename std::vector<std::string>;
  using iVec = typename std::vector<int>;
  iVec gpu_ids;
  if (result.count("gpu_ids")) {
    gpu_ids = result["gpu_ids"].as<iVec>();
  } else {
    for (int slot = 0; slot < result["ngpus"].as<int>(); ++slot)
      gpu_ids.push_back(slot);
  }
  PrometheusExporter::AddInfoLabels({
    {"F_engines", VJoin(result["addr"].as<sVec>())},
    {"ports",VJoin(result["port"].as<iVec>())},
    {"image_size",std::to_string(result["imagesize"].as<int>())},
    {"imageres", std::to_string(result["imageres"].as<float>())},
    {"nts",std::to_string(result["nts"].as<int>())},
    {"seq_accum",std::to_string(result["seq_accum"].as<int>())},
    {"nimg_accum",std::to_string(result["nimg_accum"].as<int>())},
    {"channels",std::to_string(result["channels"].as<int>())},
    {"support",std::to_string(result["support"].as<int>())},
    {"kernel_oversample",std::to_string(result["kernel_oversample"].as<int>())},
    {"accum_16bit",std::to_string(result["accum_16bit"].as<bool>())},
    {"chan_nbin",std::to_string(result["chan_nbin"].as<int>())},
    {"nstreams",std::to_string(result["nstreams"].as<int>())},
    {"ngpus",std::to_string(result["ngpus"].as<int>())},
    {"gpu_ids",VJoin(gpu_ids)},
    {"elev_limit_deg",std::to_string(result["elev_limit_deg"].as<float>())},
    {"video_size",std::to_string(result["video_size"].as<int>())},
    {"stream_cmap",result["stream_cmap"].as<std::string>()},
    {"imager_build_hash",git_CommitSHA1()}
  });
}
#endif  // SRC_EX_OPTION_PARSER_HPP_

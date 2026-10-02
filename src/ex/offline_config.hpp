#ifndef SRC_EX_OFFLINE_CONFIG_HPP_
#define SRC_EX_OFFLINE_CONFIG_HPP_

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <stdexcept>

struct OfflineChannelShard {
  int first;
  int count;
};

inline int CheckedChannelBins(int channels, int bin) {
  if (channels <= 0 || bin <= 0 || channels % bin != 0) {
    throw std::invalid_argument(
        "Channel count must be positive and divisible by a positive chan_nbin");
  }
  return channels / bin;
}

inline OfflineChannelShard OfflineChannelSlice(int channels, int devices,
                                              int slot, int bin, int streams) {
  if (channels <= 0 || devices <= 0 || slot < 0 || slot >= devices ||
      channels % devices != 0) {
    throw std::invalid_argument(
        "Offline channels must split evenly across GPUs with a valid device slot");
  }
  const int count = channels / devices;
  CheckedChannelBins(count, bin);
  if (streams <= 0 || count % streams != 0) {
    throw std::invalid_argument(
        "Each GPU channel count must be divisible by a positive nstreams");
  }
  return {slot * count, count};
}

inline size_t OfflinePackedBytes(int sequences, int channels, int stands,
                                 int polarizations) {
  size_t bytes = 1;
  for (int dimension : {sequences, channels, stands, polarizations}) {
    if (dimension <= 0) {
      throw std::invalid_argument("Offline input dimensions must be positive");
    }
    if (bytes > std::numeric_limits<size_t>::max() / size_t(dimension)) {
      throw std::overflow_error("Offline packed input size exceeds size_t");
    }
    bytes *= size_t(dimension);
  }
  // Device input byte counts still use int in MOFFCuHandler.
  if (bytes > size_t(std::numeric_limits<int>::max())) {
    throw std::overflow_error(
        "Offline packed input exceeds the GPU int byte-count range; use a smaller time/channel tile");
  }
  return bytes;  // A signed complex 4+4-bit CNib occupies one byte.
}

inline uint64_t OfflineTimeTag(double unix_start, double clock_hz) {
  const char* error =
      "NPZ must provide known positive Unix UTC within the time-tag range; "
      "synthetic/relative time use FX NPZ backend";
  if (!std::isfinite(unix_start) || unix_start <= 0 ||
      !std::isfinite(clock_hz) || clock_hz <= 0) {
    throw std::invalid_argument(error);
  }
  const long double ticks =
      std::round(static_cast<long double>(unix_start) * clock_hz);
  if (!std::isfinite(ticks) || ticks <= 0 ||
      ticks > static_cast<long double>(std::numeric_limits<uint64_t>::max())) {
    throw std::invalid_argument(error);
  }
  return static_cast<uint64_t>(ticks);
}

#endif  // SRC_EX_OFFLINE_CONFIG_HPP_

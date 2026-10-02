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

#ifndef SRC_RAFT_KERNELS_DUMMY_PACKET_GEN_HPP_
#define SRC_RAFT_KERNELS_DUMMY_PACKET_GEN_HPP_
#include <glog/logging.h>

#include <chrono>
#include <memory>
#include <raft>
#include <raftio>
#include <string>
#include <utility>

#include "../ex/buffer.hpp"
#include "../ex/constants.h"
#include "../ex/offline_config.hpp"
#include "../ex/py_funcs.hpp"
#include "../ex/types.hpp"

template <class Payload, class BufferMngr>
class DummyPktGen : public raft::kernel {
 private:
  size_t m_n_pkts{3};
  int m_expected_nseqs{1000};
  const int m_ngulps{2};
  const int m_ngulps_per_seq{1000};
  // 从 Python gulp 读取 nchan/nseqs/chan0，支持 1024 通道等；仅无 Python 时用默认
  const int m_nchan_default{128};
  const int64_t m_chan0_default{1128 + 600};
  int m_channel_start{0};
  int m_channel_count{128};
  std::string m_input_path;
  uint64_t m_time_tag0{0};
  std::unique_ptr<BufferMngr> m_buf_mngr{nullptr};

 public:
  DummyPktGen(size_t p_n_pkts = 1, std::string utcstart = "2023_06_19T00_00_00",
              int expected_nseqs = 1000, int channel_start = 0,
              int channel_count = 128, std::string input_path = "")
      : raft::kernel(), m_n_pkts(p_n_pkts), m_expected_nseqs(expected_nseqs),
        m_channel_start(channel_start), m_channel_count(channel_count),
        m_input_path(std::move(input_path)) {
    VLOG(3) << "Dummy pkt constructor";
    if (channel_start < 0) throw std::invalid_argument("Channel start must be non-negative");
    m_buf_mngr = std::make_unique<BufferMngr>(
        m_ngulps, OfflinePackedBytes(expected_nseqs, channel_count,
                                    LWA_SV_NSTANDS, LWA_SV_NPOLS));
    output.addPort<Payload>("gulp");
    // Preserve the legacy constructor signature. Only NPZ metadata provides
    // an observation timestamp; no station query or invented UTC is used.
    (void)utcstart;
  }

  raft::kstatus run() override {
    using std::string_literals::operator""s;
    for (size_t i = 0; i < m_n_pkts; ++i) {
      VLOG(3) << "Generating a gulp";
      auto pld = m_buf_mngr.get()->acquire_buf();
      LOG_IF(FATAL, !static_cast<bool>(pld)) << "Empty buffer in packet gen";

      auto start = std::chrono::high_resolution_clock::now();
      int nseqs = m_ngulps_per_seq;
      int nchan_from_file = m_nchan_default;
      int64_t chan0_from_file = m_chan0_default;
      double unix_start = 0;
      Get40msGulp(pld.get_mbuf()->GetDataPtr(), pld.get_mbuf()->buf_size(),
                  m_expected_nseqs, &nseqs, &nchan_from_file,
                  &chan0_from_file, m_channel_start, m_channel_count,
                  m_input_path, &unix_start);
      m_time_tag0 = OfflineTimeTag(unix_start, FS);
      VLOG(2) << "dummy_packet_gen: after Get40msGulp, nchan_from_file=" << nchan_from_file << " nseqs=" << nseqs;
      VLOG(3) << "Gulp gen duration: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(
                     std::chrono::high_resolution_clock::now() - start)
                     .count();
      VLOG(3) << "Gulp id: " << i << " nchan: " << nchan_from_file
              << " nseqs: " << nseqs;

      VLOG(3)
          << "Sending gulp at: "
          << std::chrono::duration_cast<std::chrono::milliseconds>(
                 std::chrono::high_resolution_clock::now().time_since_epoch())
                 .count();

      auto& mref = pld.get_mbuf()->GetMetadataRef();
      uint64_t seq_start = 0;
      mref["seq_start"] = seq_start;
      mref["time_tag"] = m_time_tag0;
      mref["seq_end"] = uint64_t(nseqs + seq_start);
      mref["nseqs"] = nseqs;
      mref["gulp_len_ms"] = nseqs * (SAMPLING_LEN_uS * 1e-3);
      mref["nchan"] = nchan_from_file;  // int，支持 1024 通道
      mref["chan0"] = chan0_from_file;
      mref["data_order"] = "c_maj"s;
      mref["nbytes"] =
          uint64_t(nchan_from_file) * LWA_SV_NPOLS * nseqs * LWA_SV_NSTANDS;  // one byte/CNib

      output["gulp"].push(pld);
    }
    LOG(INFO) << "Stopping gulp gen";
    return raft::stop;
  }
};

#endif  // SRC_RAFT_KERNELS_DUMMY_PACKET_GEN_HPP_

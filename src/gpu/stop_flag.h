#pragma once

// How the host stops running kernels. The host writes a flag in mapped
// pinned host memory; kernels poll a word in device memory, and at most one
// thread every kHostPollIntervalNs reads the host flag over PCIe and copies
// it into that word. No copy or stream is involved, so a stop reaches a
// kernel even while it occupies every SM, and PCIe sees at most ~50k reads a
// second however many threads poll.
//
// History: (1) a device flag set by a copy on a second stream broke when a
// results copy queued behind the kernel held the copy engine, and under
// compute-sanitizer, where the copy needed a free SM that a persistent
// kernel never leaves; (2) every poll reading host memory directly cost the
// GPU-resident YBWC 2.8x at depth 9 (millions of tasks, each polling at its
// first node, queue up on PCIe reads).

#include <atomic>

#include "gpu/cuda_check.h"

namespace cb::gpu {

// Device memory: the flag kernels poll, and when the host flag was last read.
struct StopWords {
    int stop;
    int unused;
    unsigned long long last_host_check_ns;  // %globaltimer
};

struct StopSignal {
    const volatile int* host_stop;  // mapped pinned host memory; may be null (never stop)
    StopWords* words;               // device memory
};

constexpr unsigned long long kHostPollIntervalNs = 20'000;

}  // namespace cb::gpu

namespace cb {

class HostStopFlag {
  public:
    HostStopFlag() {
        CUDA_CHECK(cudaHostAlloc(&host_, sizeof(int), cudaHostAllocMapped));
        CUDA_CHECK(cudaHostGetDevicePointer(&host_device_, host_, 0));
        CUDA_CHECK(cudaMalloc(&words_, sizeof(gpu::StopWords)));
        set(false);
    }
    ~HostStopFlag() {
        cudaFree(words_);
        cudaFreeHost(host_);
    }
    HostStopFlag(const HostStopFlag&) = delete;
    HostStopFlag& operator=(const HostStopFlag&) = delete;

    // Host side: any time, also while a kernel runs.
    void set(bool stop) {
        *static_cast<volatile int*>(host_) = stop ? 1 : 0;
        std::atomic_thread_fence(std::memory_order_seq_cst);
    }
    // Clears the device words; queue before each launch on the kernel's stream.
    void reset_device(cudaStream_t stream) {
        CUDA_CHECK(cudaMemsetAsync(words_, 0, sizeof(gpu::StopWords), stream));
    }
    gpu::StopSignal signal() const { return gpu::StopSignal{host_device_, words_}; }

  private:
    int* host_ = nullptr;
    int* host_device_ = nullptr;  // the same memory, as seen by kernels
    gpu::StopWords* words_ = nullptr;
};

}  // namespace cb

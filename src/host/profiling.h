#pragma once

// Profiling helpers: NVTX ranges (visible in Nsight Systems timelines) and a
// wall-clock timer.

#include <chrono>

#include <nvtx3/nvToolsExt.h>

namespace cb {

// Marks a named range on the current thread for the lifetime of the object.
class NvtxRange {
  public:
    explicit NvtxRange(const char* name) { nvtxRangePushA(name); }
    ~NvtxRange() { nvtxRangePop(); }
    NvtxRange(const NvtxRange&) = delete;
    NvtxRange& operator=(const NvtxRange&) = delete;
};

class Stopwatch {
  public:
    Stopwatch() : start_(std::chrono::steady_clock::now()) {}
    double elapsed_ms() const {
        return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start_).count();
    }

  private:
    std::chrono::steady_clock::time_point start_;
};

}  // namespace cb

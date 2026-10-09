#pragma once

#include <cstddef>
#include <vector>

#include "gpu/cuda_check.h"

namespace cb {

// Owning array in device memory.
template <class T>
class DeviceBuffer {
  public:
    DeviceBuffer() = default;
    explicit DeviceBuffer(size_t count) : count_(count) {
        if (count_) CUDA_CHECK(cudaMalloc(&data_, count_ * sizeof(T)));
    }
    ~DeviceBuffer() {
        if (data_) cudaFree(data_);
    }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    T* get() const { return data_; }
    size_t size() const { return count_; }

    void upload(const T* host, size_t count) {
        if (count) CUDA_CHECK(cudaMemcpy(data_, host, count * sizeof(T), cudaMemcpyHostToDevice));
    }
    void upload(const std::vector<T>& host) { upload(host.data(), host.size()); }
    void download(T* host, size_t count) const {
        if (count) CUDA_CHECK(cudaMemcpy(host, data_, count * sizeof(T), cudaMemcpyDeviceToHost));
    }
    void zero() {
        if (count_) CUDA_CHECK(cudaMemset(data_, 0, count_ * sizeof(T)));
    }

  private:
    T* data_ = nullptr;
    size_t count_ = 0;
};

}  // namespace cb

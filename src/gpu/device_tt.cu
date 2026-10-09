#include "gpu/device_tt.h"

#include "gpu/cuda_check.h"

namespace cb {

DeviceTT::~DeviceTT() { cudaFree(buckets_); }

void DeviceTT::resize(size_t mb) {
    const u64 count = tt_bucket_count(mb);
    if (buckets_) CUDA_CHECK(cudaFree(buckets_));
    buckets_ = nullptr;
    count_ = 0;
    CUDA_CHECK(cudaMalloc(&buckets_, size_t(count) * sizeof(TTBucket)));
    count_ = count;
    clear();
}

void DeviceTT::clear() {
    CUDA_CHECK(cudaMemset(buckets_, 0, size_t(count_) * sizeof(TTBucket)));
    CUDA_CHECK(cudaDeviceSynchronize());
}

}  // namespace cb

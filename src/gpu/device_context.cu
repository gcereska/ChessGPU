#include "gpu/device_context.h"

#include <cstdio>

#include "gpu/cuda_check.h"
#include "host/host_tables.h"

namespace cb {

DeviceContext::DeviceContext(int device) {
    CUDA_CHECK(cudaSetDevice(device));
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    info_.name = prop.name;
    info_.ordinal = device;
    info_.cc_major = prop.major;
    info_.cc_minor = prop.minor;
    info_.sm_count = prop.multiProcessorCount;
    info_.max_threads_per_sm = prop.maxThreadsPerMultiProcessor;
    info_.regs_per_sm = prop.regsPerMultiprocessor;
    info_.shared_per_sm = prop.sharedMemPerMultiprocessor;
    info_.shared_per_block_optin = prop.sharedMemPerBlockOptin;
    info_.l2_bytes = prop.l2CacheSize;
    info_.total_memory = prop.totalGlobalMem;

    CUDA_CHECK(cudaMalloc(&tables_, sizeof(AttackTables)));
    CUDA_CHECK(cudaMemcpy(tables_, &host_attack_tables(), sizeof(AttackTables), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&search_tables_, sizeof(SearchTables)));
    CUDA_CHECK(cudaMemcpy(search_tables_, &host_search_tables(), sizeof(SearchTables), cudaMemcpyHostToDevice));
}

DeviceContext::~DeviceContext() {
    if (tables_) cudaFree(tables_);
    if (search_tables_) cudaFree(search_tables_);
    if (magic_tables_) cudaFree(magic_tables_);
}

const MagicTables* DeviceContext::device_magic_tables() const {
    std::call_once(magic_once_, [this] {
        CUDA_CHECK(cudaMalloc(&magic_tables_, sizeof(MagicTables)));
        CUDA_CHECK(cudaMemcpy(magic_tables_, &host_magic_tables(), sizeof(MagicTables), cudaMemcpyHostToDevice));
    });
    return magic_tables_;
}

std::string describe(const DeviceInfo& info) {
    char text[512];
    std::snprintf(text, sizeof text,
                  "GPU %d: %s\n"
                  "  compute capability %d.%d, %d SMs, %zu MB\n"
                  "  per SM: %d threads, %d registers, %zu B shared memory\n"
                  "  per block: %zu B shared memory (opt-in); L2: %d KB\n",
                  info.ordinal, info.name.c_str(), info.cc_major, info.cc_minor, info.sm_count,
                  info.total_memory >> 20, info.max_threads_per_sm, info.regs_per_sm, info.shared_per_sm,
                  info.shared_per_block_optin, info.l2_bytes >> 10);
    return text;
}

}  // namespace cb

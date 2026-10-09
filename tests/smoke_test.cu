// Toolchain smoke test: the shared host/device helpers must give identical
// results on the CPU and the GPU, and match known answers.
#include <cstdio>

#include "core/bits.h"
#include "gpu/cuda_check.h"

// Named namespace, not anonymous: nvcc's generated stub for a kernel with
// internal linkage trips MSVC warning C4211 at /W4.
namespace smoke {

using cb::u64;

struct BitResults {
    int popcount;
    int lsb;
    u64 reversed;
};

CB_HD BitResults compute(u64 b) {
    return {cb::popcount(b), cb::lsb(b), cb::reverse_bits(b)};
}

__global__ void compute_kernel(const u64* in, BitResults* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = compute(in[i]);
}

}  // namespace smoke

using smoke::BitResults;
using smoke::compute;
using smoke::compute_kernel;
using cb::u64;

int main() {
    const u64 inputs[] = {
        0x0000000000000001ull, 0x8000000000000000ull, 0x00000000000000FFull,
        0x0123456789ABCDEFull, 0xFFFFFFFFFFFFFFFFull, 0x0000001000000000ull,
    };
    constexpr int n = sizeof(inputs) / sizeof(inputs[0]);

    u64* d_in = nullptr;
    BitResults* d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_in, sizeof(inputs)));
    CUDA_CHECK(cudaMalloc(&d_out, n * sizeof(BitResults)));
    CUDA_CHECK(cudaMemcpy(d_in, inputs, sizeof(inputs), cudaMemcpyHostToDevice));
    compute_kernel<<<1, 32>>>(d_in, d_out, n);
    CUDA_CHECK(cudaGetLastError());
    BitResults gpu[n];
    CUDA_CHECK(cudaMemcpy(gpu, d_out, sizeof(gpu), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));

    int failures = 0;
    for (int i = 0; i < n; ++i) {
        BitResults cpu = compute(inputs[i]);
        if (cpu.popcount != gpu[i].popcount || cpu.lsb != gpu[i].lsb ||
            cpu.reversed != gpu[i].reversed) {
            std::printf("MISMATCH for %016llx: cpu (%d, %d, %016llx), gpu (%d, %d, %016llx)\n",
                        static_cast<unsigned long long>(inputs[i]), cpu.popcount, cpu.lsb,
                        static_cast<unsigned long long>(cpu.reversed), gpu[i].popcount,
                        gpu[i].lsb, static_cast<unsigned long long>(gpu[i].reversed));
            ++failures;
        }
    }

    // Known answers, so a bug shared by both paths still gets caught.
    if (cb::popcount(0x0123456789ABCDEFull) != 32 || cb::lsb(0x0000001000000000ull) != 36 ||
        cb::reverse_bits(1ull) != 0x8000000000000000ull) {
        std::printf("known-answer check failed\n");
        ++failures;
    }

    std::printf(failures ? "smoke test FAILED\n" : "smoke test passed\n");
    return failures ? 1 : 0;
}

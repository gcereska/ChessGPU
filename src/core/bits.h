#pragma once

#include <bit>
#include <cstdint>

// Marks code that runs on both the CPU and the GPU. All core chess code
// (board, move generation, eval) uses it, so tests and debugging can run on
// the CPU against the exact code the GPU runs.
#if defined(__CUDACC__)
#define CB_HD __host__ __device__ __forceinline__
#else
#define CB_HD inline
#endif

// Fully unrolls the next loop in device code. Keeping array indices
// compile-time constants lets small arrays live in registers.
#if defined(__CUDA_ARCH__)
#define CB_UNROLL _Pragma("unroll")
#else
#define CB_UNROLL
#endif

namespace cb {

using u64 = std::uint64_t;

CB_HD int popcount(u64 b) {
#if defined(__CUDA_ARCH__)
    return __popcll(b);
#else
    return std::popcount(b);
#endif
}

// Index of the lowest set bit. b must be nonzero.
CB_HD int lsb(u64 b) {
#if defined(__CUDA_ARCH__)
    return __ffsll(static_cast<long long>(b)) - 1;
#else
    return std::countr_zero(b);
#endif
}

// Returns the index of the lowest set bit and clears it. b must be nonzero.
CB_HD int pop_lsb(u64& b) {
    const int s = lsb(b);
    b &= b - 1;
    return s;
}

// Bit 0 <-> bit 63, bit 1 <-> bit 62, and so on.
CB_HD u64 reverse_bits(u64 b) {
#if defined(__CUDA_ARCH__)
    return __brevll(b);
#else
    b = ((b >> 1) & 0x5555555555555555ull) | ((b & 0x5555555555555555ull) << 1);
    b = ((b >> 2) & 0x3333333333333333ull) | ((b & 0x3333333333333333ull) << 2);
    b = ((b >> 4) & 0x0F0F0F0F0F0F0F0Full) | ((b & 0x0F0F0F0F0F0F0F0Full) << 4);
    b = ((b >> 8) & 0x00FF00FF00FF00FFull) | ((b & 0x00FF00FF00FF00FFull) << 8);
    b = ((b >> 16) & 0x0000FFFF0000FFFFull) | ((b & 0x0000FFFF0000FFFFull) << 16);
    return (b >> 32) | (b << 32);
#endif
}

// All ones if b is true, else zero. Lets an update be applied without a branch:
// x ^= value & mask_if(condition).
CB_HD constexpr u64 mask_if(bool b) { return 0ull - static_cast<u64>(b); }

// Load of read-only data (lookup tables). In device code this goes through
// the read-only data cache path.
CB_HD u64 load_readonly(const u64* p) {
#if defined(__CUDA_ARCH__)
    return __ldg(reinterpret_cast<const unsigned long long*>(p));
#else
    return *p;
#endif
}

CB_HD int load_readonly(const int* p) {
#if defined(__CUDA_ARCH__)
    return __ldg(p);
#else
    return *p;
#endif
}

CB_HD short load_readonly(const short* p) {
#if defined(__CUDA_ARCH__)
    return __ldg(p);
#else
    return *p;
#endif
}

}  // namespace cb

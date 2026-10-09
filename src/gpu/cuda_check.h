#pragma once

#include <cstdio>
#include <cstdlib>

#include <cuda_runtime.h>

// Exits with the failing call, file and line on any CUDA API error.
#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err_ = (call);                                             \
        if (err_ != cudaSuccess) {                                             \
            std::fprintf(stderr, "CUDA error in %s at %s:%d: %s\n", #call,     \
                         __FILE__, __LINE__, cudaGetErrorString(err_));        \
            std::exit(1);                                                      \
        }                                                                      \
    } while (0)

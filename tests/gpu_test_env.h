#pragma once

// Shared state of the GPU test executable (defined in gpu_tests.cu).

#include "gpu/device_context.h"

namespace gpu_test_env {

bool quick();                        // --quick: smaller workloads (for compute-sanitizer)
const cb::DeviceContext& context();  // one device context for every test

}  // namespace gpu_test_env

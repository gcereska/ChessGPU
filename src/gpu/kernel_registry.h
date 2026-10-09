#pragma once

// How the host drivers (search_gpu.cu, ybwc_device.cu, abdada_device.cu)
// find the search kernels. The kernels for each feature set and table layout
// are compiled in a generated unit of their own (gpu/kernels.cu.in, one per
// combination enabled by CHESSGPU_FEATURES x CHESSGPU_TT_LAYOUTS), so the
// variant matrix builds in parallel; a unit defines search_kernel_for,
// ybwc_kernel_for and abdada_kernel_for for its combination and every
// compiled board and eval.

#include <stdexcept>
#include <string>

#include "gpu/attack_sources.cuh"  // before core/tt.h: <cuda/atomic> goes last (docs/ARCHITECTURE.md)
#include "core/features.h"
#include "core/tt.h"
#include "gpu/search_gpu.h"
#include "gpu/variants.h"
#include "gpu/abdada_kernels.cuh"  // before gpu/ybwc_kernels.cuh, which includes <cuda/atomic> last
#include "gpu/ybwc_kernels.cuh"

namespace cb::gpu {

template <class Board>
using SearchKernel = void (*)(const SearchTask<Board>*, unsigned, AttackSources, const SearchTables*,
                              const HistoryEntry*, const Move*, int, StopSignal, unsigned*, TaskResult*, int, TTDesc,
                              PruneParams, const LmrTable*);
using YbwcKernel = void (*)(YbwcDeviceArgs, int);
template <class Board>
using AbdadaKernel = void (*)(const SearchTask<Board>*, AbdadaDeviceArgs);

// A kernel for options' execution model, ordering and stack: what it needs.
template <class Board>
struct SearchKernelChoice {
    SearchKernel<Board> kernel = nullptr;  // null: this variant isn't compiled in
    size_t shared_per_block = 0;
    int workers_per_block = 0;  // at 32 working lanes per warp
};

struct YbwcKernelChoice {
    YbwcKernel kernel = nullptr;  // null: this variant isn't compiled in
    size_t shared_per_block = 0;
};

template <class Board, class Eval, class Features, class Layout>
SearchKernelChoice<Board> search_kernel_for(const GpuSearchOptions& options);
template <class Board, class Eval, class Features, class Layout>
YbwcKernelChoice ybwc_kernel_for(const GpuSearchOptions& options);
// Null: not compiled in (CHESSGPU_DEVICE_SEARCHES, CHESSGPU_SEARCH_ATTACKS).
template <class Board, class Eval, class Features, class Layout>
AbdadaKernel<Board> abdada_kernel_for(const GpuSearchOptions& options);

// A type as a value, so lambdas can take types as deduced arguments.
template <class T>
struct Tag {
    using type = T;
};

// Returns f(Tag<Features>{}, Tag<Layout>{}) for options' feature set
// and table layout (feature sets without a table use TTNone); throws if that
// combination isn't compiled in.
template <class R, class F>
R with_features(const GpuSearchOptions& o, F&& f) {
    // One block per listed set (core/feature_list.h); sets CMake didn't
    // compile have none, so their kernels are never referenced.
#define CB_WITH_FEATURES_CASE(Kind, Type, name, flags)                                                  \
    if constexpr (CHESSGPU_WITH_FEATURES_##Kind != 0) {                                                 \
        if (o.features == FeatureKind::Kind) {                                                          \
            if constexpr (!Type::kTT) {                                                                 \
                return f(Tag<Type>{}, Tag<TTNone>{});                                                   \
            } else {                                                                                    \
                if constexpr (CHESSGPU_WITH_PACKED64_LAYOUT != 0)                                       \
                    if (o.tt_layout == TTLayoutKind::Packed64) return f(Tag<Type>{}, Tag<TTPacked64>{});  \
                if constexpr (CHESSGPU_WITH_WIDE128_LAYOUT != 0)                                        \
                    if (o.tt_layout == TTLayoutKind::Wide128) return f(Tag<Type>{}, Tag<TTWide128>{});    \
            }                                                                                           \
        }                                                                                               \
    }
    CB_FEATURE_SETS(CB_WITH_FEATURES_CASE)
#undef CB_WITH_FEATURES_CASE
    throw std::invalid_argument(std::string("feature set '") + name_of(o.features) + "' with table layout '" +
                                name_of(o.tt_layout) +
                                "' is not compiled into this build (CMake options CHESSGPU_FEATURES, "
                                "CHESSGPU_TT_LAYOUTS)");
}

}  // namespace cb::gpu

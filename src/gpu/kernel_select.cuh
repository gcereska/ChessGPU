#pragma once

// Definitions of search_kernel_for, ybwc_kernel_for and abdada_kernel_for
// (gpu/kernel_registry.h), included by the generated kernel units only:
// taking a kernel's address here instantiates it in that unit.

#include "core/eval_terms.h"
#include "gpu/kernel_registry.h"
#include "gpu/search_kernels.cuh"
#include "gpu/abdada_kernels.cuh"
#include "gpu/ybwc_kernels.cuh"

namespace cb::gpu {

// Sizes of the per-worker move arenas (ListOrdering, warp model) and of the
// shared-memory stack window. Tunables, like the block size.
constexpr int kThreadArenaMoves = 2048;
constexpr int kWarpArenaMoves = 2048;
constexpr int kStackWindow = 4;

template <class Board, class Eval, class Features, class Layout, class Attacks>
SearchKernelChoice<Board> search_kernel_with(const GpuSearchOptions& o) {
    using StagedLevel = SearchLevel<StagedPicker::State>;
    using ListLevel = SearchLevel<ListPicker::State>;
    SearchKernelChoice<Board> c;
    if (o.model == ExecModel::Thread) {
        c.workers_per_block = o.block_size;
        if (o.ordering == OrderingKind::Staged && o.stack == StackKind::Local) {
#if CHESSGPU_WITH_THREAD_MODEL && CHESSGPU_WITH_STAGED_ORDERING && CHESSGPU_WITH_LOCAL_STACK
            c.kernel = &search_thread_local_kernel<Board, Eval, StagedOrdering, Features, Layout, Attacks>;
#endif
        } else if (o.ordering == OrderingKind::Staged && o.stack == StackKind::Shared) {
#if CHESSGPU_WITH_THREAD_MODEL && CHESSGPU_WITH_STAGED_ORDERING && CHESSGPU_WITH_SHARED_STACK
            c.kernel = &search_thread_window_kernel<Board, Eval, kStackWindow, Features, Layout, Attacks>;
            c.shared_per_block = WindowStack<StagedLevel, kMaxPly + 1, kStackWindow>::shared_bytes(o.block_size);
#endif
        } else if (o.ordering == OrderingKind::List) {
#if CHESSGPU_WITH_THREAD_MODEL && CHESSGPU_WITH_LIST_ORDERING
            c.kernel = &search_thread_local_kernel<Board, Eval, ListOrdering<kThreadArenaMoves>, Features, Layout, Attacks>;
#endif
        }
    } else {
#if CHESSGPU_WITH_WARP_MODEL
        c.workers_per_block = o.block_size / 32;
        c.kernel = &search_warp_kernel<Board, Eval, kWarpArenaMoves, Features, Layout, Attacks>;
        c.shared_per_block = size_t(o.block_size / 32) * sizeof(WarpShared<ListLevel, kMaxPly + 1, kWarpArenaMoves>);
#endif
    }
    (void)sizeof(StagedLevel);
    (void)sizeof(ListLevel);
    return c;
}

template <class Board, class Eval, class Features, class Layout, class Attacks>
YbwcKernelChoice ybwc_kernel_with(const GpuSearchOptions& o) {
    using StagedLevel = SearchLevel<StagedPicker::State>;
    using ListLevel = SearchLevel<ListPicker::State>;
    YbwcKernelChoice c;
    if (o.model == ExecModel::Thread) {
        if (o.ordering == OrderingKind::Staged && o.stack == StackKind::Local) {
#if CHESSGPU_WITH_THREAD_MODEL && CHESSGPU_WITH_STAGED_ORDERING && CHESSGPU_WITH_LOCAL_STACK
            c.kernel = &ybwc_thread_local_kernel<Board, Eval, StagedOrdering, Features, Layout, Attacks>;
#endif
        } else if (o.ordering == OrderingKind::Staged && o.stack == StackKind::Shared) {
#if CHESSGPU_WITH_THREAD_MODEL && CHESSGPU_WITH_STAGED_ORDERING && CHESSGPU_WITH_SHARED_STACK
            c.kernel = &ybwc_thread_window_kernel<Board, Eval, kStackWindow, Features, Layout, Attacks>;
            c.shared_per_block = WindowStack<StagedLevel, kMaxPly + 1, kStackWindow>::shared_bytes(o.block_size);
#endif
        } else if (o.ordering == OrderingKind::List) {
#if CHESSGPU_WITH_THREAD_MODEL && CHESSGPU_WITH_LIST_ORDERING
            c.kernel = &ybwc_thread_local_kernel<Board, Eval, ListOrdering<kThreadArenaMoves>, Features, Layout, Attacks>;
#endif
        }
    } else {
#if CHESSGPU_WITH_WARP_MODEL
        c.kernel = &ybwc_warp_kernel<Board, Eval, kWarpArenaMoves, Features, Layout, Attacks>;
        c.shared_per_block = size_t(o.block_size / 32) * sizeof(WarpShared<ListLevel, kMaxPly + 1, kWarpArenaMoves>);
#endif
    }
    (void)sizeof(StagedLevel);
    (void)sizeof(ListLevel);
    return c;
}

// The kernels of options' sliding-attack variant (CMake CHESSGPU_SEARCH_ATTACKS).
template <class Board, class Eval, class Features, class Layout>
SearchKernelChoice<Board> search_kernel_for(const GpuSearchOptions& o) {
#if CHESSGPU_WITH_HQ_SEARCH_ATTACKS
    if (o.attacks == AttacksKind::Hq) return search_kernel_with<Board, Eval, Features, Layout, HqAttacks>(o);
#endif
#if CHESSGPU_WITH_MAGIC_SEARCH_ATTACKS
    if (o.attacks == AttacksKind::Magic) return search_kernel_with<Board, Eval, Features, Layout, MagicAttacks>(o);
#endif
    return {};
}

template <class Board, class Eval, class Features, class Layout>
YbwcKernelChoice ybwc_kernel_for(const GpuSearchOptions& o) {
#if CHESSGPU_WITH_YBWC_DEVICE_SEARCH && CHESSGPU_WITH_HQ_SEARCH_ATTACKS
    if (o.attacks == AttacksKind::Hq) return ybwc_kernel_with<Board, Eval, Features, Layout, HqAttacks>(o);
#endif
#if CHESSGPU_WITH_YBWC_DEVICE_SEARCH && CHESSGPU_WITH_MAGIC_SEARCH_ATTACKS
    if (o.attacks == AttacksKind::Magic) return ybwc_kernel_with<Board, Eval, Features, Layout, MagicAttacks>(o);
#endif
    (void)o;
    return {};
}

template <class Board, class Eval, class Features, class Layout>
AbdadaKernel<Board> abdada_kernel_for(const GpuSearchOptions& o) {
#if CHESSGPU_WITH_ABDADA_DEVICE_SEARCH && CHESSGPU_WITH_HQ_SEARCH_ATTACKS
    if (o.attacks == AttacksKind::Hq) return &abdada_thread_kernel<Board, Eval, Features, Layout, HqAttacks>;
#endif
#if CHESSGPU_WITH_ABDADA_DEVICE_SEARCH && CHESSGPU_WITH_MAGIC_SEARCH_ATTACKS
    if (o.attacks == AttacksKind::Magic) return &abdada_thread_kernel<Board, Eval, Features, Layout, MagicAttacks>;
#endif
    (void)o;
    return nullptr;
}

}  // namespace cb::gpu

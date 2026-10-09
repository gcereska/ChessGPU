#pragma once

// The GPU ABDADA search's kernel (core/abdada.h): every worker,
// one thread (thread model), runs search_task on the same root task with list
// ordering, sharing the transposition table and a table of positions being
// searched in device memory. The first worker to finish posts its result;
// the others stop at their next poll. The host driver is gpu/abdada_device.cu;
// the kernel is instantiated per feature set and table layout in the
// generated kernel units (gpu/kernels.cu.in).
//
// Plain CUDA atomics and volatile accesses rather than libcu++: Visual
// Studio's dependency scanner stops at <cuda/atomic> (docs/ARCHITECTURE.md),
// so this header can come before gpu/ybwc_kernels.cuh. Every access is
// relaxed: the busy table only steers work, and the result is read by the
// host after the kernel ends.

#include "core/abdada.h"
#include "gpu/search_kernels.cuh"

namespace cb::gpu {

// Move arena of each worker (list ordering): as the thread model's.
constexpr int kAbdadaArenaMoves = 2048;

struct AbdadaDeviceArgs {
    AttackSources sources;
    const SearchTables* search_tables;
    const HistoryEntry* history;  // game entries (the root's parent last), may be null
    const Move* pv;               // may be null
    int pv_length;
    TTDesc tt;
    PruneParams prune;
    const LmrTable* lmr;
    u64* busy;  // positions being searched (core/abdada.h)
    u64 busy_mask;
    AbdadaParams params;
    unsigned workers;
    int lanes;                     // workers per warp, from lane 0
    int* done;                     // 0 while searching, 1 once a worker has finished the root
    TaskResult* result;            // the finishing worker's
    unsigned long long* nodes;     // all workers
    unsigned long long* deferred;  // moves put back
    unsigned* started;             // workers that ran
    unsigned* finished;            // workers that searched the root to the end (the first, and any that
                                   // finished before their next stop poll)
    unsigned long long* id_sum;    // sum of worker + 1 over the workers that ran: tells duplicate numbers
    StopSignal signal;
};

// A worker: stops when the host stops the search or another worker has
// finished; claims, releases and checks positions in the shared table.
// Copies of the arguments it needs (not a pointer to the kernel's parameter,
// which would put the parameter block in local memory).
struct DeviceAbdadaControl {
    static constexpr bool kAbdada = true;
    AbdadaParams params;
    u64* busy_table;
    u64 mask;
    const int* done;
    StopSignal signal;
    u32 worker;
    unsigned long long deferred = 0;

    __device__ bool should_stop(u64) {
        return *static_cast<volatile const int*>(done) != 0 || stop_requested(signal);
    }
    __device__ bool busy(u64 key) const {
        const u64 slot = *static_cast<volatile const u64*>(&busy_table[key & mask]);
        return abdada::busy_for(slot, key, worker);
    }
    __device__ void claim(u64 key) const {
        *static_cast<volatile u64*>(&busy_table[key & mask]) = abdada::slot_value(key, worker);
    }
    __device__ void release(u64 key) const {
        atomicCAS(reinterpret_cast<unsigned long long*>(&busy_table[key & mask]),
                  (unsigned long long)abdada::slot_value(key, worker), 0ull);
    }
    __device__ void count_deferred() { ++deferred; }
};

template <class Board, class Eval, class Features, class Layout, class Attacks>
__global__ void __maxnreg__(kThreadKernelMaxRegisters)
    abdada_thread_kernel(const SearchTask<Board>* __restrict__ task, AbdadaDeviceArgs a) {
    const unsigned lane = threadIdx.x & 31;
    if (int(lane) >= a.lanes) return;
    const unsigned warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const unsigned worker = warp * unsigned(a.lanes) + lane;
    if (worker >= a.workers) return;
    atomicAdd(a.started, 1u);
    atomicAdd(a.id_sum, (unsigned long long)worker + 1);
    using Ordering = ListOrdering<kAbdadaArenaMoves>;
    using Level = SearchLevel<typename Ordering::State>;
    const SearchEnv<Attacks> env{make_attacks<Attacks>(a.sources), a.search_tables, a.history, a.pv, a.pv_length,
                                 a.tt, a.prune, a.lmr};
    LocalStack<Level, kMaxPly + 1> stack;
    typename Ordering::Workspace ws;
    DeviceAbdadaControl control{a.params, a.busy, a.busy_mask, a.done, a.signal, worker};
    const SearchResult r = search_task<Board, Eval, Ordering, decltype(stack), DeviceAbdadaControl, false, Features,
                                       Layout>(*task, env, stack, ws, control, nullptr);
    atomicAdd(a.nodes, (unsigned long long)r.nodes);
    atomicAdd(a.deferred, control.deferred);
    if (!r.aborted) {
        atomicAdd(a.finished, 1u);
        if (atomicCAS(a.done, 0, 1) == 0) store_result(a.result, r);
    }
}

}  // namespace cb::gpu

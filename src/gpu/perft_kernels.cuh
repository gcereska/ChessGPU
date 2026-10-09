#pragma once

// Perft kernels for the two execution models (see docs/ARCHITECTURE.md, Results):
//   thread model: each thread searches one task's subtree on its own
//   warp model:   the 32 lanes of a warp search one task together
// Both are persistent: they keep claiming tasks from an atomic counter until
// none are left (see docs/ARCHITECTURE.md, Parallel search on the GPU).

#include "core/perft.h"
#include "gpu/attack_sources.cuh"
#include "gpu/perft_gpu.h"

namespace cb::gpu {

constexpr int kMaxLevels = kGpuMaxTaskDepth - 1;  // move-list levels per worker
constexpr unsigned kFullMask = 0xFFFFFFFFu;
constexpr int kMaxMovesPerPosition = MoveList::capacity;

// Warp-wide sum and maximum over all 32 lanes, the same value in every lane.
// Compute capability 8.0 and newer have an instruction for each; 7.5 (Turing)
// combines lanes in a shuffle tree.
__device__ inline unsigned warp_sum(unsigned v) {
#if __CUDA_ARCH__ >= 800
    return __reduce_add_sync(kFullMask, v);
#else
    for (int offset = 16; offset > 0; offset /= 2) v += __shfl_xor_sync(kFullMask, v, offset);
    return v;
#endif
}

__device__ inline unsigned warp_max(unsigned v) {
#if __CUDA_ARCH__ >= 800
    return __reduce_max_sync(kFullMask, v);
#else
    for (int offset = 16; offset > 0; offset /= 2) {
        const unsigned other = __shfl_xor_sync(kFullMask, v, offset);
        v = other > v ? other : v;
    }
    return v;
#endif
}

// ---------------------------------------------------------------------------
// Thread model
// ---------------------------------------------------------------------------

template <class Board, class Attacks, class Moves>
__global__ void perft_thread_kernel(const Board* __restrict__ tasks, unsigned num_tasks, int depth,
                                    AttackSources sources, unsigned* __restrict__ next_task,
                                    unsigned long long* __restrict__ counts) {
    const Attacks attacks = make_attacks<Attacks>(sources);
    PerftStack<kMaxLevels, typename Moves::template Buffer<Board>> stack;  // per-thread, in local memory
    for (unsigned t = atomicAdd(next_task, 1u); t < num_tasks; t = atomicAdd(next_task, 1u))
        counts[t] = perft_iterative(tasks[t], attacks, depth, stack);
}

// ---------------------------------------------------------------------------
// Warp model. Every lane holds the same board and applies the same moves, so
// control flow stays uniform; only move generation is split across lanes
// (lane i handles the i-th piece of the side to move).
// ---------------------------------------------------------------------------

// Move lists of one warp, in shared memory (Move or Move32).
template <class MoveType>
struct WarpLists {
    MoveType moves[kMaxLevels][kMaxMovesPerPosition];
};

// Square of the n-th (0-based) set bit of b, or NO_SQUARE.
CB_HD Square nth_square(u64 b, int n) {
    for (int i = 0; i < n && b; ++i) b &= b - 1;
    return b ? Square(lsb(b)) : NO_SQUARE;
}

// The piece this lane generates moves for, or NO_SQUARE.
__device__ inline Square lane_piece(const LegalContext& c, int lane) {
    return nth_square(only_king_moves(c) ? bit(c.king) : c.our, lane);
}

// Number of legal moves; the same value in every lane.
template <class Board, class Attacks>
__device__ unsigned warp_count_legal(const Board& b, const Attacks& a, int lane) {
    const LegalContext c = make_legal_context(b, a);
    const Square from = lane_piece(c, lane);
    MoveCounter counter;
    if (from != NO_SQUARE) generate_moves_from(b, a, c, from, counter);
    return warp_sum(counter.count);
}

// Writes all legal moves to out[] (shared memory) and returns their number;
// the same value in every lane.
template <class Board, class Attacks, class Moves>
__device__ int warp_generate_legal(const Board& b, const Attacks& a, typename Moves::MoveType* out, int lane) {
    const LegalContext c = make_legal_context(b, a);
    const Square from = lane_piece(c, lane);
    typename Moves::template Buffer<Board, 32> mine;  // no piece has more than 27 legal moves
    mine.bind(b);
    if (from != NO_SQUARE) generate_moves_from(b, a, c, from, mine);

    // An inclusive prefix sum of the per-lane counts gives each lane its
    // output offset; the last lane's sum is the total.
    int inclusive = mine.size;
    for (int d = 1; d < 32; d <<= 1) {
        const int below = __shfl_up_sync(kFullMask, inclusive, d);
        if (lane >= d) inclusive += below;
    }
    const int total = __shfl_sync(kFullMask, inclusive, 31);
    const int offset = inclusive - mine.size;

    __syncwarp();  // every lane is done reading the list previously stored here
    for (int i = 0; i < mine.size; ++i) out[offset + i] = mine.moves[i];
    __syncwarp();  // the new list is visible to every lane
    return total;
}

// Same DFS as perft_iterative, with generation and counting split across the warp.
template <class Board, class Attacks, class Moves>
__device__ u64 warp_perft(Board b, const Attacks& a, int depth, WarpLists<typename Moves::MoveType>& lists,
                          int lane) {
    if (depth <= 0) return 1;
    if (depth == 1) return warp_count_legal(b, a, lane);

    // Identical in every lane, so each lane keeps its own copy.
    int size[kMaxLevels];
    int next[kMaxLevels];
    Undo undo[kMaxLevels];
    size[0] = warp_generate_legal<Board, Attacks, Moves>(b, a, lists.moves[0], lane);
    next[0] = 0;
    int ply = 0;
    u64 nodes = 0;
    for (;;) {
        if (next[ply] == size[ply]) {
            if (ply == 0) break;
            --ply;
            unmake_move(b, lists.moves[ply][next[ply] - 1], undo[ply]);
            continue;
        }
        const auto m = lists.moves[ply][next[ply]++];
        undo[ply] = make_move(b, m);
        if (depth - ply - 1 == 1) {
            nodes += warp_count_legal(b, a, lane);
            unmake_move(b, m, undo[ply]);
        } else {
            ++ply;
            size[ply] = warp_generate_legal<Board, Attacks, Moves>(b, a, lists.moves[ply], lane);
            next[ply] = 0;
        }
    }
    return nodes;
}

// Needs blockDim.x a multiple of 32 and blockDim.x / 32 * sizeof(WarpLists<...>)
// bytes of dynamic shared memory.
template <class Board, class Attacks, class Moves>
__global__ void perft_warp_kernel(const Board* __restrict__ tasks, unsigned num_tasks, int depth,
                                  AttackSources sources, unsigned* __restrict__ next_task,
                                  unsigned long long* __restrict__ counts) {
    using Lists = WarpLists<typename Moves::MoveType>;
    extern __shared__ __align__(16) unsigned char shared_bytes[];
    Lists& lists = reinterpret_cast<Lists*>(shared_bytes)[threadIdx.x / 32];
    const int lane = int(threadIdx.x & 31);
    const Attacks attacks = make_attacks<Attacks>(sources);
    for (;;) {
        unsigned t = 0;
        if (lane == 0) t = atomicAdd(next_task, 1u);
        t = __shfl_sync(kFullMask, t, 0);
        if (t >= num_tasks) break;
        const u64 nodes = warp_perft<Board, Attacks, Moves>(tasks[t], attacks, depth, lists, lane);
        if (lane == 0) counts[t] = nodes;
    }
}

}  // namespace cb::gpu

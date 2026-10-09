#pragma once

// Search kernels. Each worker runs core/search.h's search_task on one task
// at a time, claiming tasks from an atomic counter (persistent kernels, like
// perft):
//   thread model: one thread per task, with StagedOrdering (stack in local
//                 memory or a shared-memory window) or ListOrdering (move
//                 arena in local memory)
//   warp model:   one warp per task. All 32 lanes run search_task in step
//                 on identical data; move generation and picking are split
//                 across lanes (WarpListOrdering), and the stack and move
//                 arena live in shared memory, one per warp.
// Every variant visits exactly the same tree as the CPU search of the same
// task (same move order), so results and node counts must match the CPU.
// Kernels also take the feature set (core/features.h) and transposition
// table layout (core/tt.h) as template parameters, and the table itself.

#include "gpu/attack_sources.cuh"
#include "core/search.h"
#include "gpu/perft_kernels.cuh"
#include "gpu/stop_flag.h"

namespace cb::gpu {

// Result of one task as stored by the kernels (16 bytes).
struct TaskResult {
    int score;
    u16 best;     // Move::raw()
    u8 aborted;
    u8 over_budget;
    unsigned long long nodes;
};

__device__ inline unsigned long long global_timer_ns() {
    unsigned long long t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
}

// Has the host asked running kernels to stop (UCI "stop", a time limit)?
// Reads the device word; at most one thread every kHostPollIntervalNs (the
// one that wins the timestamp exchange) reads the host flag over PCIe and,
// if it is set, sets the device word for everyone (see gpu/stop_flag.h).
__device__ inline bool stop_requested(const StopSignal& s) {
    if (s.host_stop == nullptr) return false;
    if (*reinterpret_cast<volatile int*>(&s.words->stop) != 0) return true;
    const unsigned long long now = global_timer_ns();
    const unsigned long long last = *reinterpret_cast<volatile unsigned long long*>(&s.words->last_host_check_ns);
    if (now - last < kHostPollIntervalNs) return false;
    if (atomicCAS(&s.words->last_host_check_ns, last, now) != last) return false;  // another thread checks
    if (*s.host_stop == 0) return false;
    atomicExch(&s.words->stop, 1);
    return true;
}

// Thread model: each thread decides for itself.
struct ThreadControl {
    StopSignal signal;
    __device__ bool should_stop(u64) { return stop_requested(signal); }
};

// Warp model: lane 0 probes the table and broadcasts what it found, so every
// lane takes the same path even while other workers write the entry; only
// lane 0 stores (core/search.h, tt_probe).
struct WarpTTAccess {
    int lane;
    template <class View>
    __device__ TTData warp_tt_probe(const View& tt, u64 key) const {
        u64 raw = 0;
        if (lane == 0) raw = tt.probe(key).raw;
        return TTData{__shfl_sync(kFullMask, raw, 0)};
    }
    template <class View>
    __device__ void warp_tt_store(const View& tt, u64 key, TTData d) const {
        if (lane == 0) tt.store(key, d);
    }
};

// Warp model: lane 0 decides and broadcasts, so all lanes stay on one path.
struct WarpControl : WarpTTAccess {
    StopSignal signal;
    __device__ WarpControl(StopSignal s, int lane_) : WarpTTAccess{lane_}, signal(s) {}
    __device__ bool should_stop(u64) {
        int stop = 0;
        if (lane == 0) stop = stop_requested(signal) ? 1 : 0;
        return __shfl_sync(kFullMask, stop, 0) != 0;
    }
};

// ---------------------------------------------------------------------------
// Thread model: stack in a shared-memory window
// ---------------------------------------------------------------------------

// The kWindow most recent stack levels stay in shared memory; older ones
// spill to local memory as the search goes deeper and come back when it
// returns. Levels are stored as 32-bit words in a structure-of-arrays layout
// (word w of slot s for thread t at [(s * kWords + w) * stride + t]), so a
// warp's accesses hit 32 different banks.
template <class Level, int kLevels, int kWindow>
struct WindowStack {
    static_assert(sizeof(Level) % 4 == 0, "levels are copied as 32-bit words");
    static constexpr int kWords = int(sizeof(Level) / 4);
    static constexpr int capacity = kLevels;
    // Shared memory a block of `threads` threads needs.
    static constexpr size_t shared_bytes(int threads) { return size_t(kWindow) * kWords * 4 * size_t(threads); }

    u32* shared;      // this thread's first word
    int stride;       // threads per block
    int window_base;  // lowest level currently held in shared memory
    Level spill[kLevels];

    __device__ WindowStack(u32* block_shared, int thread, int threads)
        : shared(block_shared + thread), stride(threads), window_base(0) {}

    __device__ void write_slot(int level, const Level& l) {
        u32 words[kWords];
        memcpy(words, &l, sizeof(Level));
        u32* base = shared + size_t(level % kWindow) * kWords * stride;
        CB_UNROLL
        for (int w = 0; w < kWords; ++w) base[w * stride] = words[w];
    }
    __device__ Level read_slot(int level) const {
        u32 words[kWords];
        const u32* base = shared + size_t(level % kWindow) * kWords * stride;
        CB_UNROLL
        for (int w = 0; w < kWords; ++w) words[w] = base[w * stride];
        Level l;
        memcpy(&l, words, sizeof(Level));
        return l;
    }

    // Levels above i are dead when i is stored (the search is about to go
    // down from level i).
    __device__ void store(int i, const Level& l) {
        while (i - window_base >= kWindow) {
            spill[window_base] = read_slot(window_base);
            ++window_base;
        }
        write_slot(i, l);
    }
    // Levels above i are dead when i is loaded (the search came back up to i).
    __device__ Level load(int i) {
        if (i < window_base) {
            write_slot(i, spill[i]);
            window_base = i;
        }
        return read_slot(i);
    }
    __device__ u64 key(int i) const {
        if (i < window_base) return spill[i].key;
        return read_slot(i).key;
    }
};

// ---------------------------------------------------------------------------
// Warp model
// ---------------------------------------------------------------------------

// The warp's stack in shared memory: lane 0 writes, every lane reads.
template <class Level, int kLevels>
struct WarpStack {
    static constexpr int capacity = kLevels;
    Level* levels;  // kLevels entries in shared memory
    int lane;

    __device__ void store(int i, const Level& l) {
        __syncwarp();  // every lane has finished reading what is overwritten
        if (lane == 0) levels[i] = l;
        __syncwarp();
    }
    __device__ Level load(int i) const { return levels[i]; }
    __device__ u64 key(int i) const { return levels[i].key; }
};

// Moves and order keys of the warp's nodes on the current path.
template <int kCapacity>
struct WarpArena {
    Move moves[kCapacity];
    u32 keys[kCapacity];  // 0 once picked
};

// ListPicker's order, computed by the whole warp: lane i generates the
// moves of the i-th piece and their keys, a prefix sum places them in the
// warp's arena, and each pick is a warp-wide maximum of the keys left.
template <int kCapacity>
struct WarpListOrdering {
    static constexpr const char* kName = "warp-list";
    using State = ListPicker::State;
    struct Workspace {
        WarpArena<kCapacity>* arena;
        int lane;
    };

    template <class Board, class Attacks>
    __device__ static void init(State& st, Workspace& ws, int top, const Board& b, const Attacks& a, const CheckInfo&,
                                PickMode mode, Move pv, Killers killers) {
        const LegalContext c = make_legal_context(b, a);
        const Square from = lane_piece(c, ws.lane);
        MoveBuffer<32> mine;  // no piece has more than 27 legal moves
        if (from != NO_SQUARE) generate_moves_from(b, a, c, from, mine);
        u32 keys[32];
        int count = 0;
        for (int i = 0; i < mine.size; ++i) {
            const u32 key = order_key(b, mine.moves[i], mode == PickMode::All ? pv : Move(),
                                      mode == PickMode::All ? killers : Killers{});
            if (mode == PickMode::Quiescence && !is_quiescence_group(key)) continue;
            if (mode == PickMode::QuiescenceChecks && !is_quiescence_group(key) && !is_quiet_check(b, a, mine.moves[i]))
                continue;
            mine.moves[count] = mine.moves[i];
            keys[count] = key;
            ++count;
        }
        int inclusive = count;
        for (int d = 1; d < 32; d <<= 1) {
            const int below = __shfl_up_sync(kFullMask, inclusive, d);
            if (ws.lane >= d) inclusive += below;
        }
        const int total = __shfl_sync(kFullMask, inclusive, 31);
        const int offset = top + inclusive - count;
        st.begin = u16(top);
        st.picked = 0;
        st.mode = u8(mode);
        st.overflow = top + total > kCapacity ? 1 : 0;
        st.end = u16(st.overflow ? top : top + total);
        if (st.overflow) return;
        __syncwarp();  // every lane is done with what was stored here before
        for (int i = 0; i < count; ++i) {
            ws.arena->moves[offset + i] = mine.moves[i];
            ws.arena->keys[offset + i] = keys[i];
        }
        __syncwarp();
    }
    __device__ static int top(const State& st) { return st.end; }
    __device__ static bool overflowed(const State& st) { return st.overflow != 0; }

    template <class Board, class Attacks>
    __device__ static Move next(State& st, Workspace& ws, const Board&, const Attacks&, const CheckInfo&, Move) {
        u32 best_key = 0;
        int best_index = -1;
        for (int i = st.begin + ws.lane; i < st.end; i += 32) {
            const u32 k = ws.arena->keys[i];
            if (k > best_key) {
                best_key = k;
                best_index = i;
            }
        }
        const u32 warp_best = warp_max(best_key);
        if (warp_best == 0) return Move();  // every key is 0: all moves tried
        // Keys are unique, so exactly one lane holds the maximum.
        const unsigned owner = __ballot_sync(kFullMask, best_key == warp_best);
        const int index = __shfl_sync(kFullMask, best_index, __ffs(owner) - 1);
        const Move m = ws.arena->moves[index];
        __syncwarp();
        if (ws.lane == 0) ws.arena->keys[index] = 0;
        __syncwarp();
        ++st.picked;
        return m;
    }
};

// Shared memory per warp in the warp model.
template <class Level, int kLevels, int kArena>
struct WarpShared {
    Level levels[kLevels];
    WarpArena<kArena> arena;
};

// ---------------------------------------------------------------------------
// Kernels
// ---------------------------------------------------------------------------

__device__ inline void store_result(TaskResult* out, const SearchResult& r) {
    TaskResult t;
    t.score = r.score;
    t.best = r.best.raw();
    t.aborted = r.aborted ? 1 : 0;
    t.over_budget = r.over_budget ? 1 : 0;
    t.nodes = r.nodes;
    *out = t;
}

// Register cap of the thread-model kernels. Uncapped they use ~246 registers
// (8 warps per SM); at 168 a few values spill, 12 warps fit, and throughput
// on many tasks rose ~25% while one worker alone slowed ~3% (measured with
// bench-tasks). A tunable, like the block size.
constexpr int kThreadKernelMaxRegisters = 168;

// Thread model: `lanes` threads of each warp (1-32) work; the others exit at
// once. Threads of a warp running different searches mostly execute one at a
// time (their control flow diverges), so a small batch runs faster spread
// one task per warp than packed 32 per warp.
template <class Board, class Eval, class Ordering, class Features, class Layout, class Attacks>
__global__ void __maxnreg__(kThreadKernelMaxRegisters) search_thread_local_kernel(const SearchTask<Board>* __restrict__ tasks, unsigned num_tasks,
                                           AttackSources sources,
                                           const SearchTables* __restrict__ search_tables,
                                           const HistoryEntry* __restrict__ history, const Move* __restrict__ pv,
                                           int pv_length, StopSignal signal, unsigned* __restrict__ next_task,
                                           TaskResult* __restrict__ results, int lanes, TTDesc tt,
                                           PruneParams prune, const LmrTable* __restrict__ lmr) {
    if (int(threadIdx.x & 31) >= lanes) return;
    using Level = SearchLevel<typename Ordering::State>;
    const SearchEnv<Attacks> env{make_attacks<Attacks>(sources), search_tables, history, pv, pv_length, tt, prune, lmr};
    LocalStack<Level, kMaxPly + 1> stack;
    typename Ordering::Workspace ws;
    ThreadControl control{signal};
    for (unsigned t = atomicAdd(next_task, 1u); t < num_tasks; t = atomicAdd(next_task, 1u)) {
        const SearchResult r = search_task<Board, Eval, Ordering, decltype(stack), ThreadControl, false, Features, Layout>(
            tasks[t], env, stack, ws, control, nullptr);
        store_result(&results[t], r);
    }
}

template <class Board, class Eval, int kWindow, class Features, class Layout, class Attacks>
__global__ void __maxnreg__(kThreadKernelMaxRegisters) search_thread_window_kernel(const SearchTask<Board>* __restrict__ tasks, unsigned num_tasks,
                                            AttackSources sources,
                                            const SearchTables* __restrict__ search_tables,
                                            const HistoryEntry* __restrict__ history, const Move* __restrict__ pv,
                                            int pv_length, StopSignal signal, unsigned* __restrict__ next_task,
                                            TaskResult* __restrict__ results, int lanes, TTDesc tt,
                                            PruneParams prune, const LmrTable* __restrict__ lmr) {
    if (int(threadIdx.x & 31) >= lanes) return;
    using Level = SearchLevel<StagedOrdering::State>;
    extern __shared__ __align__(16) unsigned char shared_bytes[];
    const SearchEnv<Attacks> env{make_attacks<Attacks>(sources), search_tables, history, pv, pv_length, tt, prune, lmr};
    WindowStack<Level, kMaxPly + 1, kWindow> stack(reinterpret_cast<u32*>(shared_bytes), int(threadIdx.x),
                                                  int(blockDim.x));
    StagedOrdering::Workspace ws;
    ThreadControl control{signal};
    for (unsigned t = atomicAdd(next_task, 1u); t < num_tasks; t = atomicAdd(next_task, 1u)) {
        stack.window_base = 0;
        const SearchResult r =
            search_task<Board, Eval, StagedOrdering, decltype(stack), ThreadControl, false, Features, Layout>(
                tasks[t], env, stack, ws, control, nullptr);
        store_result(&results[t], r);
    }
}

template <class Board, class Eval, int kArena, class Features, class Layout, class Attacks>
__global__ void search_warp_kernel(const SearchTask<Board>* __restrict__ tasks, unsigned num_tasks,
                                   AttackSources sources,
                                   const SearchTables* __restrict__ search_tables,
                                   const HistoryEntry* __restrict__ history, const Move* __restrict__ pv, int pv_length,
                                   StopSignal signal, unsigned* __restrict__ next_task,
                                   TaskResult* __restrict__ results, int /*lanes: always 32*/, TTDesc tt,
                                   PruneParams prune, const LmrTable* __restrict__ lmr) {
    using Ordering = WarpListOrdering<kArena>;
    using Level = SearchLevel<typename Ordering::State>;
    using Shared = WarpShared<Level, kMaxPly + 1, kArena>;
    extern __shared__ __align__(16) unsigned char shared_bytes[];
    Shared& mine = reinterpret_cast<Shared*>(shared_bytes)[threadIdx.x / 32];
    const int lane = int(threadIdx.x & 31);
    const SearchEnv<Attacks> env{make_attacks<Attacks>(sources), search_tables, history, pv, pv_length, tt, prune, lmr};
    WarpStack<Level, kMaxPly + 1> stack{mine.levels, lane};
    typename Ordering::Workspace ws{&mine.arena, lane};
    WarpControl control{signal, lane};
    for (;;) {
        unsigned t = 0;
        if (lane == 0) t = atomicAdd(next_task, 1u);
        t = __shfl_sync(kFullMask, t, 0);
        if (t >= num_tasks) break;
        const SearchResult r = search_task<Board, Eval, Ordering, decltype(stack), WarpControl, false, Features, Layout>(
            tasks[t], env, stack, ws, control, nullptr);
        if (lane == 0) store_result(&results[t], r);
    }
}

}  // namespace cb::gpu

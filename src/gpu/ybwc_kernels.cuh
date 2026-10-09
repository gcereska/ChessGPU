#pragma once

// The GPU-resident YBWC's device code (host/ybwc.cpp's node logic run by the
// workers themselves): explicit nodes, the work queue, workers and kernels.
// The host driver is gpu/ybwc_device.cu; the kernels are instantiated per
// feature set and table layout in the generated kernel units (gpu/kernels.cu.in).

#include "core/ybwc_rules.h"
#include "gpu/search_kernels.cuh"

// Last: Visual Studio's CUDA dependency scanner stops at this header (see core/tt.h).
#include <cuda/atomic>

namespace cb {

namespace gpu {

// Node states and predicted types (YbwcNodeType values), as in host/ybwc.cpp.
// kYNull: an explicit node waiting for its null move's search (feature sets
// with null-move pruning), before its moves.
enum : int { kYIdle = 0, kYEldest = 1, kYBrothers = 2, kYQueued = 3, kYDone = 4, kYNull = 5 };
// YNode::flags.
enum : u8 {
    kYAfterNull = 1,  // the move into the node was a null move
    kYReduced = 2,    // late move reduction: searched shallower with a zero window (see report)
    kYInCheck = 4,    // explicit nodes: the side to move is in check (set when the moves start)
};
enum : int { kYPv = int(YbwcNodeType::Pv), kYCut = int(YbwcNodeType::Cut), kYAll = int(YbwcNodeType::All) };

// A node of the explicit tree, or a task, in device memory. Fields set when
// the node is created don't change afterwards; an explicit node's
// bookkeeping changes only under its `lock`; `state` is also read without
// the lock, by tasks checking whether an ancestor has finished.
struct YNode {
    QuadBoard board;
    int parent;  // -1: the root
    int slot;    // index of `move` among the parent's moves
    int ply;
    int depth;
    int alpha;  // window, own point of view
    int beta;
    int parent_alpha;  // the parent's alpha this node was searched with: its result is exact above it
    int best;
    int best_child;
    int best_exact;
    int brothers_alpha;
    int remaining;
    int next_slot;
    int in_flight;
    int wave;
    int value;
    int moves_offset;  // explicit nodes: moves in the move pool, in search order
    int num_moves;
    int state;
    int lock;
    u16 move;  // from the parent (none: the parent's null move, slot -1)
    u16 task_best;
    u8 on_pv;
    u8 task;
    u8 type;
    u8 flags;  // kYAfterNull, kYReduced, kYInCheck
};

static_assert(sizeof(YNode) == 128, "YNode layout changed");
static_assert(offsetof(YNode, parent) == 40 && offsetof(YNode, wave) == 96 && offsetof(YNode, state) == 112 &&
                  offsetof(YNode, on_pv) == 124,
              "YNode layout changed");

struct YbwcDeviceArgs {
    const AttackTables* attack_tables;
    const MagicTables* magic_tables;  // the search's Attacks variant is MagicAttacks (else may be null)
    const SearchTables* search_tables;
    const Move* pv;  // may be null
    int pv_length;
    HistoryEntry* history;  // game entries, then node n's entry at game_count + n
    int game_count;
    int root_repeated;
    int leaf_depth;
    int max_explicit_ply;
    u32 node_budget;
    int cut_waves;
    YNode* nodes;
    int node_capacity;
    int* node_count;
    Move* moves;
    int move_capacity;
    int* move_count;
    int* queue;
    unsigned* ready;
    unsigned queue_capacity;
    unsigned* head;
    unsigned* tail;
    int* done;  // 0 running, 1 root finished, 2 aborted (stop, deadline or a full pool)
    StopSignal signal;
    unsigned long long* nodes_searched;
    unsigned* explicit_nodes;
    unsigned* tasks;
    unsigned* skipped;
    unsigned* over_budget;
    int* overflow;
    unsigned* null_tries;  // explicit nodes that searched a null move
    unsigned* null_cuts;   // ... and ended with its result
    unsigned* lmr_reductions;  // moves of explicit nodes searched reduced first
    unsigned* lmr_researches;  // ... and again at full depth
    unsigned* fut_pruned;      // moves of explicit nodes skipped by futility pruning
    unsigned* killer_stores;   // explicit-node cutoffs that changed their ply's killers
    unsigned* killers_used;    // explicit nodes with a killer among their moves
    unsigned* qs_pruned;       // moves of explicit quiescence nodes skipped by delta or SEE pruning
    u32* killers;          // explicit nodes' killers by ply, packed (feature sets with killers)
    TTDesc tt;             // the transposition table (feature sets with one)
    PruneParams prune;     // settings of the pruning techniques
    const LmrTable* lmr;   // late move reductions (feature sets with them)
};

template <class T>
__device__ inline cuda::atomic_ref<T, cuda::thread_scope_device> aref(T& x) {
    return cuda::atomic_ref<T, cuda::thread_scope_device>(x);
}

// Never consulted: explicit nodes have no stack below them.
struct YNoStack {
    CB_HD u64 key(int) const { return 0; }
};

// The node logic of host/ybwc.cpp, for concurrent workers. complete,
// activate and expand are called, not inlined: each runs once per explicit
// node, and complete has many callers (inlined, every caller got its own
// copy of move generation: 4x the compile time, 27 MB cubins). With a table,
// explicit nodes use it like search_task's nodes: a stored result can end
// the node (never at the root or on the previous PV), a stored move is tried
// first, and every explicit node that finishes is stored. With null-move
// pruning, an explicit node allowed one (null_move_allowed, as in
// search_task) first queues its null move as a child of its own (slot -1)
// and waits in state kYNull: a result of at least beta ends it (as in
// search_task), otherwise its moves start. With late move reductions, a
// late quiet move (as in search_task) starts as a reduced child with the
// zero window (alpha, alpha + 1); if it beats alpha, a full child replaces
// it in the same slot. With futility pruning, the futile moves (as in
// search_task, with the node's alpha when its moves are generated) never
// become children. With killer moves, explicit nodes share one killer table
// by ply (tasks keep their own, as search_task does): a quiet move whose
// result ends its parent with a cutoff becomes the newest killer of the
// parent's ply, and nodes generating their moves later order with it.
// With quiescence pruning, an explicit quiescence node (made by splitting a
// task over its node budget) skips the captures delta and SEE pruning skip
// in search_task, deciding with its alpha when its moves are generated.
template <class Eval, class Features, class Layout, class Attacks>
struct YCoordinator {
    const YbwcDeviceArgs& a;
    Attacks attacks;
    TTView<Layout> tt;

    __device__ bool is_explicit(int depth, int ply) const { return depth > a.leaf_depth && ply < a.max_explicit_ply; }

    // Entry of the position before node n (where its history chain starts).
    __device__ int parent_entry(int n) const {
        const int p = a.nodes[n].parent;
        return p < 0 ? a.game_count - 1 : a.game_count + p;
    }

    __device__ bool search_over() const { return aref(*a.done).load(cuda::memory_order_relaxed) != 0; }

    __device__ void abort_search() const {
        int expected = 0;
        aref(*a.done).compare_exchange_strong(expected, 2, cuda::memory_order_relaxed, cuda::memory_order_relaxed);
    }

    __device__ bool ancestor_done(int n) const {
        for (int p = a.nodes[n].parent; p >= 0; p = a.nodes[p].parent)
            if (aref(a.nodes[p].state).load(cuda::memory_order_relaxed) == kYDone) return true;
        return false;
    }

    __device__ void lock(int n) const {
        int sleep = 32;
        for (;;) {
            int expected = 0;
            if (aref(a.nodes[n].lock)
                    .compare_exchange_weak(expected, 1, cuda::memory_order_acquire, cuda::memory_order_relaxed))
                return;
            __nanosleep(sleep);
            sleep = min(sleep * 2, 1024);
        }
    }
    __device__ void unlock(int n) const { aref(a.nodes[n].lock).store(0, cuda::memory_order_release); }

    __device__ void push(int n) const {
        const unsigned i = atomicAdd(a.tail, 1u);
        if (i >= a.queue_capacity) {
            atomicExch(a.overflow, 1);
            abort_search();
            return;
        }
        a.queue[i] = n;
        aref(a.ready[i]).store(1u, cuda::memory_order_release);
    }

    // The next node to work on, or -1 once the search is over. Claims a
    // queue slot, then waits for it to be filled.
    __device__ int pop() const {
        const unsigned i = atomicAdd(a.head, 1u);
        if (i >= a.queue_capacity) return -1;
        int sleep = 64;
        while (aref(a.ready[i]).load(cuda::memory_order_acquire) == 0) {
            if (search_over()) return -1;
            __nanosleep(sleep);
            sleep = min(sleep * 2, 8192);
        }
        return a.queue[i];
    }

    __device__ int alloc_node() const {
        const int n = atomicAdd(a.node_count, 1);
        if (n >= a.node_capacity) {
            atomicExch(a.overflow, 1);
            abort_search();
            return -1;
        }
        return n;
    }

    __device__ int static_eval(const QuadBoard& b) const {
        return evaluate_position<Eval>(b, compute_accumulator<Eval>(b, a.search_tables->eval), state::side(b.state()),
                                       attacks);
    }

    // A new node below explicit node p, not started.
    __device__ int create_node(int p, const QuadBoard& board, int slot, int depth, Move m, bool on_pv, int type,
                               u8 flags) const {
        const int c = alloc_node();
        if (c < 0) return -1;
        const YNode& par = a.nodes[p];
        YNode& ch = a.nodes[c];
        ch.board = board;
        ch.parent = p;
        ch.slot = slot;
        ch.ply = par.ply + 1;
        ch.depth = depth;
        ch.best = -kScoreInfinite;
        ch.best_child = -1;
        ch.best_exact = 0;
        ch.brothers_alpha = 0;
        ch.remaining = 0;
        ch.next_slot = 0;
        ch.in_flight = 0;
        ch.wave = 0;
        ch.value = 0;
        ch.moves_offset = 0;
        ch.num_moves = 0;
        ch.lock = 0;
        ch.move = m.raw();
        ch.task_best = 0;
        ch.on_pv = on_pv ? 1 : 0;
        ch.task = is_explicit(ch.depth, ch.ply) ? 0 : 1;
        ch.type = u8(type);
        ch.flags = flags;  // every byte written: the node is read and copied as whole words
        return c;
    }

    // The killers of explicit nodes at `ply` (none without killer moves).
    __device__ Killers load_killers(int ply) const {
        if constexpr (Features::kKillers) {
            return unpack_killers(aref(a.killers[ply]).load(cuda::memory_order_relaxed));
        } else {
            return Killers{};
        }
    }

    // A cutoff at an explicit node at `ply` by move m: a quiet m becomes the
    // ply's newest killer. Updates of the same ply from other workers may
    // come in between; each update is one whole word.
    __device__ void store_killer(int ply, Move m) const {
        if constexpr (Features::kKillers) {
            if (m.is_none() || !can_be_killer(m)) return;
            auto word = aref(a.killers[ply]);
            u32 old = word.load(cuda::memory_order_relaxed);
            for (;;) {
                const u32 next = pack_killers(add_killer(unpack_killers(old), m));
                if (next == old) return;
                if (word.compare_exchange_weak(old, next, cuda::memory_order_relaxed, cuda::memory_order_relaxed)) {
                    atomicAdd(a.killer_stores, 1u);
                    return;
                }
            }
        }
    }

    // Child `slot` of explicit node p, not started. With late move
    // reductions, a late quiet move that doesn't give check, at a node not in
    // check, gets its reduced depth (and kYReduced), unless `full`: its
    // re-search.
    __device__ int create_child(int p, int slot, bool full = false) const {
        const YNode& par = a.nodes[p];
        const Move m = a.moves[par.moves_offset + slot];
        QuadBoard b = par.board;
        make_move(b, m);
        const Move pv_move = (a.pv != nullptr && par.ply < a.pv_length) ? a.pv[par.ply] : Move();
        int depth = par.depth - 1;
        u8 flags = 0;
        if constexpr (Features::kCheckExt) {
            if (par.depth >= 1 && side_to_move_in_check(b, attacks)) ++depth;  // as in search_task
        }
        if constexpr (Features::kLmr) {
            if (!full && !(par.flags & kYInCheck) && !m.is_capture() && !m.is_promotion()) {
                const int r = a.lmr->reduction(par.depth, slot + 1);
                if (r > 0 && !side_to_move_in_check(b, attacks)) {
                    depth -= r;
                    flags = kYReduced;
                    atomicAdd(a.lmr_reductions, 1u);
                }
            }
        }
        return create_node(p, b, slot, depth, m, par.on_pv && m == pv_move,
                           int(child_type(YbwcNodeType(par.type), slot)), flags);
    }

    // Creates and queues child `slot` of p with the window (alpha, p's beta)
    // from p's view, or the zero window (alpha, alpha + 1) if it's reduced.
    __device__ bool start_move(int p, int slot, int alpha, bool full = false) const {
        const int c = create_child(p, slot, full);
        if (c < 0) return false;
        start_child(c, alpha, (a.nodes[c].flags & kYReduced) ? alpha + 1 : a.nodes[p].beta);
        return true;
    }

    // The null move of explicit node p, not started: reduced depth, never on
    // the PV, expected to fail low (an all node) so that p cuts off.
    __device__ int create_null_child(int p) const {
        const YNode& par = a.nodes[p];
        QuadBoard b = par.board;
        make_null_move(b);
        return create_node(p, b, -1, par.depth - 1 - null_move_reduction(a.prune, par.depth), Move(), false, kYAll,
                           kYAfterNull);
    }

    // Queues child c with window (alpha_p, beta_p) from its parent's view.
    __device__ void start_child(int c, int alpha_p, int beta_p) const {
        YNode& ch = a.nodes[c];
        ch.parent_alpha = alpha_p;
        ch.alpha = -beta_p;
        ch.beta = -alpha_p;
        ch.state = ch.task ? kYQueued : kYIdle;
        push(c);
    }

    // Starts the next young brothers of p (under p's lock).
    __device__ void start_wave(int p) const {
        YNode& par = a.nodes[p];
        int size = par.num_moves - par.next_slot;
        if (a.cut_waves && par.type == kYCut) {
            par.wave = par.wave == 0 ? 1 : par.wave * 2;
            size = min(size, par.wave);
        }
        const int first = par.next_slot;
        par.next_slot += size;
        par.in_flight += size;
        const int alpha = max(par.brothers_alpha, par.best);
        for (int slot = first; slot < first + size; ++slot)
            if (!start_move(p, slot, alpha)) return;
    }

    // Under p's lock: child c finished with value v_child (its own view).
    // Returns true when p is finished too, with value `out`.
    __device__ bool report(int p, int c, int v_child, int& out) const {
        YNode& par = a.nodes[p];
        const YNode& ch = a.nodes[c];
        const int v = -v_child;
        const bool exact = v > ch.parent_alpha;
        if constexpr (Features::kLmr) {
            if ((ch.flags & kYReduced) && v > ch.parent_alpha) {
                // The reduced move beat its alpha: search it again at full
                // depth, in the same slot (which stays in flight).
                atomicAdd(a.lmr_researches, 1u);
                start_move(p, ch.slot, par.state == kYEldest ? par.alpha : max(par.brothers_alpha, par.best), true);
                return false;
            }
        }
        if (par.state == kYNull) {
            // The null move's result: a cutoff (as in search_task), or the moves.
            if (v >= par.beta) {
                atomicAdd(a.null_cuts, 1u);
                out = null_move_score(v, par.beta);
                return true;
            }
            return expand(p, tt_move_of(p), false, false, par.alpha, -kScoreInfinite, out);
        }
        if (par.state == kYEldest) {
            if (v > par.best) {  // always, except below a stand pat
                par.best = v;
                par.best_child = c;
                par.best_exact = exact;
            }
            if (par.best >= par.beta) {
                store_killer(par.ply, Move::from_raw(ch.move));
                out = par.best;
                return true;
            }
            par.brothers_alpha = max(par.alpha, par.best);
            par.remaining = par.num_moves - 1;
            par.next_slot = 1;
            par.in_flight = 0;
            par.wave = 0;
            aref(par.state).store(kYBrothers, cuda::memory_order_relaxed);
            if (par.remaining == 0) {
                out = par.best;
                return true;
            }
            start_wave(p);
            return false;
        }
        --par.remaining;
        --par.in_flight;
        const int best_slot = par.best_child >= 0 ? a.nodes[par.best_child].slot : (1 << 30);
        if (better_child(v, exact, ch.slot, par.best, par.best_exact != 0, best_slot)) {
            par.best = v;
            par.best_child = c;
            par.best_exact = exact;
        }
        if (v >= par.beta || par.remaining == 0) {
            if (v >= par.beta) store_killer(par.ply, Move::from_raw(ch.move));
            out = par.best;
            return true;
        }
        if (par.in_flight == 0) start_wave(p);
        return false;
    }

    // Node n (owned by this worker) is finished with value v: record it and
    // carry finished results up the tree.
    __device__ __noinline__ void complete(int n, int v) const {
        a.nodes[n].value = v;
        aref(a.nodes[n].state).store(kYDone, cuda::memory_order_release);
        for (;;) {
            const int p = a.nodes[n].parent;
            if (p < 0) {
                int expected = 0;
                aref(*a.done).compare_exchange_strong(expected, 1, cuda::memory_order_release,
                                                      cuda::memory_order_relaxed);
                return;
            }
            lock(p);
            if (a.nodes[p].state == kYDone) {
                unlock(p);
                return;
            }
            int pv = 0;
            const bool finished = report(p, n, v, pv);
            if (finished) {
                a.nodes[p].value = pv;
                aref(a.nodes[p].state).store(kYDone, cuda::memory_order_release);
            }
            unlock(p);
            if (!finished) return;
            store_tt(p, pv);
            n = p;
            v = pv;
        }
    }

    // The table's move for explicit node p (activated: its history entry holds its key).
    __device__ Move tt_move_of(int p) const {
        if constexpr (Features::kTT) {
            return tt_use<false>(tt.probe(a.history[a.game_count + p].key), a.nodes[p].ply, a.nodes[p].depth, false,
                                 false, 0, 0)
                .move;
        } else {
            return Move();
        }
    }

    // Stores finished explicit node p (activated: its history entry holds its key).
    __device__ void store_tt(int p, int v) const {
        if constexpr (Features::kTT) {
            const YNode& x = a.nodes[p];
            const bool quiescence = x.depth <= 0;
            if (quiescence && !Features::kTTQuiescence) return;
            const Move best = x.best_child >= 0 ? Move::from_raw(a.nodes[x.best_child].move) : Move();
            // x.alpha may have been raised by a stand pat: the bound is then
            // weaker than it could be, never wrong.
            tt.store(a.history[a.game_count + p].key,
                     tt_entry(best, v, x.ply, x.depth, x.alpha, x.beta, tt.generation));
        }
    }

    // Starts explicit node n (window in its alpha/beta), as host/ybwc.cpp's activate.
    __device__ __noinline__ void activate(int n) const {
        atomicAdd(a.explicit_nodes, 1u);
        YNode& x = a.nodes[n];
        const QuadBoard b = x.board;
        const int ply = x.ply;
        const u64 key = compute_key(b, attacks, a.search_tables->zobrist);
        const int halfmove = state::halfmove(b.state());
        if (ply > 0) {
            SearchTask<QuadBoard> probe;
            probe.ply = u8(ply);
            probe.flags = 0;
            probe.history = parent_entry(n);
            if (detail::is_repetition(YNoStack{}, 0, key, halfmove, probe, a.history) || insufficient_material(b)) {
                complete(n, kScoreDraw);
                return;
            }
        }
        const CheckInfo ci = make_check_info(b, attacks);
        if (ply > 0 && halfmove >= 100) {
            complete(n, (ci.in_check() && count_legal(b, attacks) == 0) ? mated_in(ply) : kScoreDraw);
            return;
        }
        if (ply >= kMaxPly) {
            complete(n, static_eval(b));
            return;
        }
        const bool quiescence = x.depth <= 0;
        Move tt_move;
        if constexpr (Features::kTT) {
            if (!quiescence || Features::kTTQuiescence) {
                const TTUse u = tt_use<Features::kTTCutoffs>(tt.probe(key), ply, x.depth, quiescence, x.on_pv != 0,
                                                             x.alpha, x.beta);
                tt_move = u.move;  // ordered first only if it's among the legal moves
                if (u.cut) {
                    complete(n, u.score);
                    return;
                }
            }
        }
        const bool stand_pat = quiescence && !ci.in_check();
        int alpha = x.alpha;
        int best = -kScoreInfinite;
        if (stand_pat) {
            best = static_eval(b);
            if (best >= x.beta) {
                complete(n, best);
                return;
            }
            alpha = max(alpha, best);
        }
        // The node's history entry: its descendants' repetition checks and
        // the table store when it finishes read it.
        HistoryEntry e;
        e.key = key;
        e.prev = parent_entry(n);
        e.ply = short(ply);
        e.repeated = u8(ply == 0 && a.root_repeated ? 1 : 0);
        e.unused = 0;
        a.history[a.game_count + n] = e;
        if constexpr (Features::kNullMove) {
            const u8 flags =
                u8((x.on_pv ? level_flags::kOnPv : 0) | ((x.flags & kYAfterNull) ? level_flags::kAfterNull : 0));
            if (!quiescence && null_move_allowed<Eval>(a.prune, ply, x.depth, flags, ci.in_check(), x.beta, b, ci.us,
                                                       compute_accumulator<Eval>(b, a.search_tables->eval),
                                                       attacks)) {
                atomicAdd(a.null_tries, 1u);
                // No other worker knows this node before its null move is queued.
                aref(x.state).store(kYNull, cuda::memory_order_relaxed);
                const int c = create_null_child(n);
                if (c < 0) return;
                start_child(c, x.beta - 1, x.beta);
                return;
            }
        }
        int out = 0;
        if (expand(n, tt_move, ci.in_check(), stand_pat, alpha, best, out)) complete(n, out);
    }

    // Generates explicit node n's moves and starts its eldest child (alpha
    // and best: raised by a stand pat). Returns true, with the node's value
    // in `out`, if it has no moves to search.
    __device__ __noinline__ bool expand(int n, Move tt_move, bool in_check, bool stand_pat, int alpha, int best,
                                        int& out) const {
        YNode& x = a.nodes[n];
        const QuadBoard b = x.board;
        const int ply = x.ply;
        const bool quiescence = x.depth <= 0;
        MoveList list;
        generate_legal(b, attacks, list);
        const Move pv_move = (!quiescence && x.on_pv && a.pv != nullptr && ply < a.pv_length) ? a.pv[ply] : Move();
        const Move first = pv_move.is_none() ? tt_move : pv_move;
        const Killers killers = load_killers(ply);
        // Futility pruning: quiet moves that don't give check go if this
        // bound can't reach alpha.
        int futility = kScoreInfinite;
        if constexpr (Features::kFutility) {
            const u8 flags = u8((in_check ? level_flags::kInCheck : 0) | (x.on_pv ? level_flags::kOnPv : 0));
            futility = futility_bound(a.prune, static_eval(b), x.depth, flags, alpha);
        }
        int pruned = 0;
        int qs_pruned = 0;
        int qs_bound = -kScoreInfinite;  // delta pruning: the skipped moves' best bound
        bool killer_found = false;
        u32 keys[MoveList::capacity];
        Move ordered[MoveList::capacity];
        int count = 0;
        for (int i = 0; i < list.size; ++i) {
            const u32 k = order_key(b, list.moves[i], first, killers);
            if (stand_pat && !is_quiescence_group(k)) {
                // Quiet checks at quiescence's first ply (kQChecks), as in search_task.
                bool quiet_check = false;
                if constexpr (qchecks_v<Features>) quiet_check = x.depth == 0 && is_quiet_check(b, attacks, list.moves[i]);
                if (!quiet_check) continue;
            }
            if constexpr (Features::kKillers) {
                if (order::group(k) == order::kKiller) killer_found = true;
            }
            if constexpr (Features::kQPrune) {
                if (stand_pat && (list.moves[i].is_capture() || list.moves[i].is_promotion())) {
                    const int bound = quiescence_prune_bound(b, attacks, a.prune, list.moves[i], best, alpha);
                    if (bound != kScoreInfinite) {
                        if (bound > qs_bound) qs_bound = bound;
                        ++qs_pruned;
                        continue;
                    }
                }
            }
            if constexpr (Features::kFutility) {
                const Move m = list.moves[i];
                if (futility <= alpha && !m.is_capture() && !m.is_promotion() && !gives_check(b, m, attacks)) {
                    ++pruned;
                    continue;
                }
            }
            // Insertion sort, highest key first.
            int j = count++;
            while (j > 0 && keys[j - 1] < k) {
                keys[j] = keys[j - 1];
                ordered[j] = ordered[j - 1];
                --j;
            }
            keys[j] = k;
            ordered[j] = list.moves[i];
        }
        if (killer_found) atomicAdd(a.killers_used, 1u);
        if (qs_pruned > 0) {
            atomicAdd(a.qs_pruned, unsigned(qs_pruned));
            if (qs_bound > best) best = qs_bound;  // at least the bound (fail-soft), as in search_task
        }
        if (pruned > 0) {
            atomicAdd(a.fut_pruned, unsigned(pruned));
            if (futility > best) best = futility;  // at least the bound (fail-soft), as in search_task
        }
        if (count == 0) {
            // No move to search: mate, stalemate, the stand pat, or every move futile.
            out = in_check ? mated_in(ply) : ((quiescence || pruned > 0) ? best : kScoreDraw);
            return true;
        }
        const int offset = atomicAdd(a.move_count, count);
        if (offset + count > a.move_capacity) {
            atomicExch(a.overflow, 1);
            abort_search();
            return false;
        }
        for (int i = 0; i < count; ++i) a.moves[offset + i] = ordered[i];
        x.moves_offset = offset;
        x.num_moves = count;
        x.alpha = alpha;  // raised by the stand pat
        x.best = best;    // the stand pat, or -infinity
        x.best_child = -1;
        x.best_exact = 0;
        x.task = 0;
        if (in_check) x.flags |= kYInCheck;
        aref(x.state).store(kYEldest, cuda::memory_order_release);
        start_move(n, 0, alpha);
        return false;
    }
};

// Stops a task when the search is stopped or over, or an ancestor finished.
struct YTaskCheck {
    const YbwcDeviceArgs* a;
    int node;

    __device__ bool stop() const {
        if (stop_requested(a->signal)) return true;
        if (aref(*a->done).load(cuda::memory_order_relaxed) != 0) return true;
        for (int p = a->nodes[node].parent; p >= 0; p = a->nodes[p].parent)
            if (aref(a->nodes[p].state).load(cuda::memory_order_relaxed) == kYDone) return true;
        return false;
    }
};

struct YThreadControl {
    YTaskCheck check;
    __device__ bool should_stop(u64) { return check.stop(); }
};

struct YWarpControl : WarpTTAccess {
    YTaskCheck check;
    __device__ YWarpControl(YTaskCheck c, int lane_) : WarpTTAccess{lane_}, check(c) {}
    __device__ bool should_stop(u64) {
        int stop = 0;
        if (lane == 0) stop = check.stop() ? 1 : 0;
        return __shfl_sync(kFullMask, stop, 0) != 0;
    }
};

// Workers: how a task's subtree gets searched.
template <class Board, class Eval, class Ordering, class Features, class Layout>
struct YThreadLocalWorker {
    static constexpr bool kWarp = false;
    using Level = SearchLevel<typename Ordering::State>;
    LocalStack<Level, kMaxPly + 1> stack;
    typename Ordering::Workspace ws;

    __device__ YThreadLocalWorker(unsigned char*) {}
    __device__ bool leader() const { return true; }
    __device__ int share(int v) const { return v; }
    __device__ void sync() const {}
    template <class Env>
    __device__ SearchResult search(const SearchTask<Board>& t, const Env& env, YTaskCheck check) {
        YThreadControl control{check};
        return search_task<Board, Eval, Ordering, decltype(stack), YThreadControl, false, Features, Layout>(
            t, env, stack, ws, control, nullptr);
    }
};

template <class Board, class Eval, int kWindow, class Features, class Layout>
struct YThreadWindowWorker {
    static constexpr bool kWarp = false;
    using Level = SearchLevel<StagedOrdering::State>;
    WindowStack<Level, kMaxPly + 1, kWindow> stack;
    StagedOrdering::Workspace ws;

    __device__ explicit YThreadWindowWorker(unsigned char* shared)
        : stack(reinterpret_cast<u32*>(shared), int(threadIdx.x), int(blockDim.x)) {}
    __device__ bool leader() const { return true; }
    __device__ int share(int v) const { return v; }
    __device__ void sync() const {}
    template <class Env>
    __device__ SearchResult search(const SearchTask<Board>& t, const Env& env, YTaskCheck check) {
        YThreadControl control{check};
        stack.window_base = 0;
        return search_task<Board, Eval, StagedOrdering, decltype(stack), YThreadControl, false, Features, Layout>(
            t, env, stack, ws, control, nullptr);
    }
};

template <class Board, class Eval, int kArena, class Features, class Layout>
struct YWarpWorker {
    static constexpr bool kWarp = true;
    using Ordering = WarpListOrdering<kArena>;
    using Level = SearchLevel<typename Ordering::State>;
    using Shared = WarpShared<Level, kMaxPly + 1, kArena>;
    WarpStack<Level, kMaxPly + 1> stack;
    typename Ordering::Workspace ws;
    int lane;

    __device__ explicit YWarpWorker(unsigned char* shared)
        : stack{reinterpret_cast<Shared*>(shared)[threadIdx.x / 32].levels, int(threadIdx.x & 31)},
          ws{&reinterpret_cast<Shared*>(shared)[threadIdx.x / 32].arena, int(threadIdx.x & 31)},
          lane(int(threadIdx.x & 31)) {}
    __device__ bool leader() const { return lane == 0; }
    __device__ int share(int v) const { return __shfl_sync(kFullMask, v, 0); }
    __device__ void sync() const { __syncwarp(); }
    template <class Env>
    __device__ SearchResult search(const SearchTask<Board>& t, const Env& env, YTaskCheck check) {
        YWarpControl control{check, lane};
        return search_task<Board, Eval, Ordering, decltype(stack), YWarpControl, false, Features, Layout>(
            t, env, stack, ws, control, nullptr);
    }
};

// What the leader decided to do with a popped node.
enum : int { kYActSkip = 0, kYActTask = 1, kYActDone = 2 };

template <class Board, class Eval, class Features, class Layout, class Attacks, class Worker>
__device__ void ybwc_worker_loop(const YbwcDeviceArgs& a, int lanes, unsigned char* shared) {
    if (!Worker::kWarp && int(threadIdx.x & 31) >= lanes) return;
    Worker w(shared);
    const AttackSources sources{a.attack_tables, a.magic_tables};
    const YCoordinator<Eval, Features, Layout, Attacks> coord{a, make_attacks<Attacks>(sources), TTView<Layout>(a.tt)};
    const SearchEnv<Attacks> env{make_attacks<Attacks>(sources), a.search_tables, a.history, a.pv, a.pv_length, a.tt,
                                 a.prune, a.lmr};
    for (;;) {
        int n = -1;
        if (w.leader()) n = coord.pop();
        n = w.share(n);
        if (n < 0) return;
        int action = kYActDone;
        if (w.leader()) {
            YNode& x = a.nodes[n];
            if (coord.search_over() || coord.ancestor_done(n)) {
                atomicAdd(a.skipped, 1u);
                action = kYActSkip;
            } else if (!x.task) {
                coord.activate(n);
                action = kYActDone;
            } else {
                // Young brothers start from the parent's best score so far.
                if (x.parent >= 0 && x.slot > 0) {
                    YNode& par = a.nodes[x.parent];
                    const int al = max(aref(par.brothers_alpha).load(cuda::memory_order_relaxed),
                                       aref(par.best).load(cuda::memory_order_relaxed));
                    x.parent_alpha = al;
                    // A reduced move's zero window (al, al + 1), unless the parent is cutting off.
                    x.alpha = ((x.flags & kYReduced) && al < par.beta) ? -(al + 1) : -par.beta;
                    x.beta = -al;
                }
                if (x.alpha >= x.beta) {
                    atomicAdd(a.skipped, 1u);  // the parent is cutting off right now
                    action = kYActSkip;
                } else {
                    action = kYActTask;
                }
            }
        }
        action = w.share(action);
        w.sync();  // the leader's writes to the node are visible to every lane
        if (action != kYActTask) continue;
        const YNode& x = a.nodes[n];
        SearchTask<Board> t;
        t.board = convert_board<Board>(x.board);
        t.alpha = short(x.alpha);
        t.beta = short(x.beta);
        t.depth = static_cast<signed char>(x.depth);
        t.ply = u8(x.ply);
        t.flags = u8((x.on_pv ? kTaskOnPv : 0) | (x.ply == 0 && a.root_repeated ? kTaskRootRepeated : 0) |
                     ((x.flags & kYAfterNull) ? kTaskAfterNull : 0));
        t.unused = 0;
        t.history = coord.parent_entry(n);
        t.node_budget = a.node_budget;
        const SearchResult r = w.search(t, env, YTaskCheck{&a, n});
        if (w.leader()) {
            atomicAdd(a.nodes_searched, (unsigned long long)r.nodes);
            atomicAdd(a.tasks, 1u);
            if (r.over_budget) {
                atomicAdd(a.over_budget, 1u);
                a.nodes[n].task = 0;
                coord.activate(n);
            } else if (r.aborted) {
                if (stop_requested(a.signal)) coord.abort_search();
                // otherwise an ancestor finished: the result isn't needed
            } else {
                a.nodes[n].task_best = r.best.raw();
                coord.complete(n, r.score);
            }
        }
        w.sync();
    }
}

template <class Board, class Eval, class Ordering, class Features, class Layout, class Attacks>
__global__ void __maxnreg__(kThreadKernelMaxRegisters) ybwc_thread_local_kernel(YbwcDeviceArgs a, int lanes) {
    ybwc_worker_loop<Board, Eval, Features, Layout, Attacks,
                     YThreadLocalWorker<Board, Eval, Ordering, Features, Layout>>(a, lanes, nullptr);
}

template <class Board, class Eval, int kWindow, class Features, class Layout, class Attacks>
__global__ void __maxnreg__(kThreadKernelMaxRegisters) ybwc_thread_window_kernel(YbwcDeviceArgs a, int lanes) {
    extern __shared__ __align__(16) unsigned char shared_bytes[];
    ybwc_worker_loop<Board, Eval, Features, Layout, Attacks,
                     YThreadWindowWorker<Board, Eval, kWindow, Features, Layout>>(a, lanes, shared_bytes);
}

template <class Board, class Eval, int kArena, class Features, class Layout, class Attacks>
__global__ void ybwc_warp_kernel(YbwcDeviceArgs a, int lanes) {
    extern __shared__ __align__(16) unsigned char shared_bytes[];
    ybwc_worker_loop<Board, Eval, Features, Layout, Attacks, YWarpWorker<Board, Eval, kArena, Features, Layout>>(
        a, lanes, shared_bytes);
}

}  // namespace gpu

}  // namespace cb

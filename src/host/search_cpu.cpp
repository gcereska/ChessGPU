#include "host/search_cpu.h"

#include <memory>
#include <mutex>
#include <stdexcept>
#include <thread>

#include "host/host_tables.h"
#include "host/task_runner.h"

namespace cb {

namespace {

struct CpuControl {
    const std::atomic<bool>* stop = nullptr;
    std::chrono::steady_clock::time_point deadline = std::chrono::steady_clock::time_point::max();

    bool should_stop(u64) {
        if (stop && stop->load(std::memory_order_relaxed)) return true;
        return deadline != std::chrono::steady_clock::time_point::max() &&
               std::chrono::steady_clock::now() >= deadline;
    }
};

// A worker of an ABDADA search (core/abdada.h): stops when stopped, at the
// deadline, or when another worker has finished the root.
struct CpuAbdadaControl {
    static constexpr bool kAbdada = true;
    AbdadaParams params;
    const std::atomic<bool>* stop = nullptr;
    const std::atomic<bool>* done = nullptr;
    std::chrono::steady_clock::time_point deadline = std::chrono::steady_clock::time_point::max();
    u64* table = nullptr;
    u64 mask = 0;
    u32 worker = 0;
    u64 deferred = 0;

    bool should_stop(u64) {
        if (done->load(std::memory_order_relaxed)) return true;
        if (stop && stop->load(std::memory_order_relaxed)) return true;
        return deadline != std::chrono::steady_clock::time_point::max() &&
               std::chrono::steady_clock::now() >= deadline;
    }
    bool busy(u64 key) const {
        return abdada::busy_for(std::atomic_ref<u64>(table[key & mask]).load(std::memory_order_relaxed), key, worker);
    }
    void claim(u64 key) {
        std::atomic_ref<u64>(table[key & mask]).store(abdada::slot_value(key, worker), std::memory_order_relaxed);
    }
    void release(u64 key) {
        u64 mine = abdada::slot_value(key, worker);
        std::atomic_ref<u64>(table[key & mask])
            .compare_exchange_strong(mine, 0, std::memory_order_relaxed, std::memory_order_relaxed);
    }
    void count_deferred() { ++deferred; }
};

// What a CPU search of one variant needs besides the position.
template <class Board, class Eval, class Ordering, class Features, class Layout>
struct CpuSearcher {
    using Level = SearchLevel<typename Ordering::State>;
    LocalStack<Level, kMaxPly + 1> stack;
    typename Ordering::Workspace workspace;
    PvTable pv_table;
    LmrTable lmr{};  // feature sets with late move reductions

    template <class Control>
    FixedDepthResult run(const Position& root, const GameHistory& history, int depth, int alpha, int beta,
                         const std::vector<Move>& pv_hint, const TTDesc& tt, const PruneParams& prune,
                         Control& control) {
        if constexpr (Features::kLmr) lmr = make_lmr_table(prune);
        SearchTask<Board> task;
        task.board = to_board<Board>(root);
        task.alpha = short(alpha);
        task.beta = short(beta);
        task.depth = static_cast<signed char>(depth);
        task.ply = 0;
        task.flags = u8(kTaskOnPv | (history.root_repeated ? kTaskRootRepeated : 0));
        task.history = history.root_parent();
        // Magic bitboards: twice hyperbola quintessence's speed on the CPU
        // (docs/ARCHITECTURE.md, Results); the GPU kernels keep HQ.
        SearchEnv<MagicAttacks> env{MagicAttacks(host_attack_tables(), host_magic_tables()),
                                    &host_search_tables(),
                                    history.entries.empty() ? nullptr : history.entries.data(),
                                    pv_hint.empty() ? nullptr : pv_hint.data(),
                                    int(pv_hint.size()),
                                    tt,
                                    prune,
                                    &lmr};
        FixedDepthResult r;
        r.search = search_task<Board, Eval, Ordering, decltype(stack), Control, true, Features, Layout>(
            task, env, stack, workspace, control, &pv_table);
        if (!r.search.aborted) r.pv.assign(pv_table.moves[0], pv_table.moves[0] + pv_table.length[0]);
        return r;
    }
};

// Calls f.template operator()<Board, Eval, Ordering, Features, Layout>() for
// the chosen variants (feature sets without a TT use the TTNone layout).
template <class F>
auto dispatch(const CpuSearchOptions& o, F&& f) {
    if (uses_tt(o.features) && o.tt.buckets == nullptr)
        throw std::invalid_argument(std::string("feature set '") + name_of(o.features) + "' needs a table");
    validate_prune(o.prune);
    const auto with_board = [&]<class Eval, class Ordering, class Features, class Layout>() {
        if (o.board == BoardKind::Std) return f.template operator()<StdBoard, Eval, Ordering, Features, Layout>();
        return f.template operator()<QuadBoard, Eval, Ordering, Features, Layout>();
    };
    const auto with_eval = [&]<class Ordering, class Features, class Layout>() {
        if (o.eval == EvalKind::Material)
            return with_board.template operator()<MaterialEval, Ordering, Features, Layout>();
        if (o.eval == EvalKind::Terms)
            return with_board.template operator()<PestoTermsEval, Ordering, Features, Layout>();
        if (o.eval == EvalKind::Nnue) return with_board.template operator()<NnueEval, Ordering, Features, Layout>();
        return with_board.template operator()<PestoEval, Ordering, Features, Layout>();
    };
    const auto with_ordering = [&]<class Features, class Layout>() {
        if (o.ordering == OrderingKind::Staged) return with_eval.template operator()<StagedOrdering, Features, Layout>();
        return with_eval.template operator()<ListOrdering<kCpuArenaMoves>, Features, Layout>();
    };
    const auto with_layout = [&]<class Features>() {
        if constexpr (!Features::kTT) {
            return with_ordering.template operator()<Features, TTNone>();
        } else {
            if (o.tt_layout == TTLayoutKind::Wide128) return with_ordering.template operator()<Features, TTWide128>();
            return with_ordering.template operator()<Features, TTPacked64>();
        }
    };
    return visit_features(o.features, [&](auto t) {
        return with_layout.template operator()<typename decltype(t)::type>();
    });
}

void check_arguments(int depth, int alpha, int beta) {
    if (depth < 0 || depth >= kMaxPly) throw std::invalid_argument("search depth out of range");
    if (alpha < -kScoreInfinite || beta > kScoreInfinite || alpha >= beta)
        throw std::invalid_argument("search window out of range");
}

}  // namespace

FixedDepthResult cpu_search(const Position& root, const GameHistory& history, int depth, int alpha, int beta,
                            const CpuSearchOptions& options, const std::vector<Move>& pv_hint,
                            const std::atomic<bool>* stop, std::chrono::steady_clock::time_point deadline) {
    check_arguments(depth, alpha, beta);
    return dispatch(options, [&]<class Board, class Eval, class Ordering, class Features, class Layout>() {
        auto searcher = std::make_unique<CpuSearcher<Board, Eval, Ordering, Features, Layout>>();
        CpuControl control{stop, deadline};
        return searcher->run(root, history, depth, alpha, beta, pv_hint, options.tt, options.prune, control);
    });
}

FixedDepthResult cpu_search_abdada(const Position& root, const GameHistory& history, int depth, int alpha, int beta,
                                   const CpuSearchOptions& options, const std::vector<Move>& pv_hint, int workers,
                                   AbdadaTable& table, const std::atomic<bool>* stop,
                                   std::chrono::steady_clock::time_point deadline, AbdadaStats* stats,
                                   const AbdadaParams& params) {
    check_arguments(depth, alpha, beta);
    if (workers < 1 || workers > 256) throw std::invalid_argument("ABDADA workers must be 1 to 256");
    validate_abdada(params);
    CpuSearchOptions o = options;
    o.ordering = OrderingKind::List;  // deferring a move needs a move list
    // Workers that stop early leave their claims behind.
    table.clear();
    std::atomic<bool> done{false};
    std::mutex mutex;
    FixedDepthResult best;
    int winner = -1;
    std::vector<u64> nodes(size_t(workers), 0), deferred(size_t(workers), 0);
    dispatch(o, [&]<class Board, class Eval, class Ordering, class Features, class Layout>() {
        if constexpr (!can_defer_v<Ordering>) {
            return 0;  // not reached: o.ordering is List
        } else {
        const auto work = [&](int w) {
            auto searcher = std::make_unique<CpuSearcher<Board, Eval, Ordering, Features, Layout>>();
            CpuAbdadaControl control;
            control.params = params;
            control.stop = stop;
            control.done = &done;
            control.deadline = deadline;
            control.table = table.slots();
            control.mask = table.mask();
            control.worker = u32(w);
            const FixedDepthResult r =
                searcher->run(root, history, depth, alpha, beta, pv_hint, o.tt, o.prune, control);
            nodes[size_t(w)] = r.search.nodes;
            deferred[size_t(w)] = control.deferred;
            if (!r.search.aborted) {
                const std::lock_guard<std::mutex> lock(mutex);
                if (winner < 0) {
                    best = r;
                    winner = w;
                    done.store(true, std::memory_order_relaxed);
                }
            }
        };
        std::vector<std::thread> pool;
        for (int w = 1; w < workers; ++w) pool.emplace_back(work, w);
        work(0);
        for (std::thread& t : pool) t.join();
        return 0;
        }
    });
    if (winner < 0) {
        best = FixedDepthResult{};
        best.search.aborted = true;
    }
    best.search.nodes = 0;
    u64 all_deferred = 0;
    for (int w = 0; w < workers; ++w) {
        best.search.nodes += nodes[size_t(w)];
        all_deferred += deferred[size_t(w)];
    }
    if (stats) *stats = AbdadaStats{workers, winner, best.search.nodes, all_deferred};
    return best;
}

std::vector<SearchResult> CpuTaskRunner::run(const std::vector<TaskSpec>& tasks,
                                             const std::vector<HistoryEntry>& history, size_t,
                                             const std::vector<Move>& pv) {
    return dispatch(options_, [&]<class Board, class Eval, class Ordering, class Features, class Layout>() {
        auto searcher = std::make_unique<CpuSearcher<Board, Eval, Ordering, Features, Layout>>();
        CpuControl control{&stop_, deadline_};
        if constexpr (Features::kLmr) searcher->lmr = make_lmr_table(options_.prune);
        const SearchEnv<MagicAttacks> env{MagicAttacks(host_attack_tables(), host_magic_tables()),
                                          &host_search_tables(),
                                          history.empty() ? nullptr : history.data(),
                                          pv.empty() ? nullptr : pv.data(),
                                          int(pv.size()),
                                          options_.tt,
                                          options_.prune,
                                          &searcher->lmr};
        std::vector<SearchResult> out;
        out.reserve(tasks.size());
        for (const TaskSpec& s : tasks) {
            if (s.history >= int(history.size())) throw std::invalid_argument("task history index out of range");
            SearchTask<Board> t;
            t.board = convert_board<Board>(s.board);
            t.alpha = short(s.alpha);
            t.beta = short(s.beta);
            t.depth = static_cast<signed char>(s.depth);
            t.ply = u8(s.ply);
            t.flags = s.flags;
            t.history = s.history;
            t.node_budget = s.node_budget;
            out.push_back(search_task<Board, Eval, Ordering, decltype(searcher->stack), CpuControl, false, Features,
                                      Layout>(t, env, searcher->stack, searcher->workspace, control, nullptr));
        }
        return out;
    });
}

std::string describe(const CpuSearchOptions& o) {
    std::string s = std::string(name_of(o.board)) + "/" + name_of(o.eval) + "/" + name_of(o.ordering);
    if (uses_tt(o.features)) s += std::string(" ") + name_of(o.features) + " " + name_of(o.tt_layout);
    return s + describe_prune(o.features, o.prune);
}

std::string CpuTaskRunner::describe() const { return "cpu " + cb::describe(options_); }

void validate_abdada(const AbdadaParams& p) {
    if (p.min_depth < 1 || p.min_depth > 20) throw std::invalid_argument("ABDADA minimum depth must be 1 to 20");
    if (p.defer_first > 1) throw std::invalid_argument("ABDADA defer-first must be 0 or 1");
}

std::string describe(const AbdadaParams& p) {
    std::string s;
    if (p.min_depth != kAbdadaMinDepth) s += " from depth " + std::to_string(p.min_depth);
    if (p.defer_first) s += " defer first";
    return s;
}

FixedDepthResult cpu_search_fixed(const Position& root, const GameHistory& history, int depth, int alpha, int beta,
                                  const CpuSearchOptions& options, const std::vector<Move>& pv_hint) {
    check_arguments(depth, alpha, beta);
    return dispatch(options, [&]<class Board, class Eval, class Ordering, class Features, class Layout>() {
        auto searcher = std::make_unique<CpuSearcher<Board, Eval, Ordering, Features, Layout>>();
        NoStop control;
        return searcher->run(root, history, depth, alpha, beta, pv_hint, options.tt, options.prune, control);
    });
}

}  // namespace cb

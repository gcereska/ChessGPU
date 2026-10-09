#pragma once

// What every search (CPU, GPU baselines, YBWC variants) offers the engine:
// one fixed-depth search of a root at a time. Iterative deepening, time
// management and reporting are built on top, once for all of them.

#include <atomic>
#include <chrono>
#include <functional>
#include <string>
#include <vector>

#include <memory>

#include "host/game_history.h"
#include "host/search_cpu.h"
#include "host/tt_storage.h"

namespace cb {

using Clock = std::chrono::steady_clock;

class SearchBackend {
  public:
    virtual ~SearchBackend() = default;

    // Searches `root` to `depth` with a full window, ordering by `pv_hint`
    // (the previous iteration's PV). Returns early, with result.search.aborted
    // set, once stop is requested or the deadline passes.
    virtual FixedDepthResult search_depth(const Position& root, const GameHistory& history, int depth,
                                          const std::vector<Move>& pv_hint) = 0;

    // The same within the window (alpha, beta), fail-soft: a score at or
    // below alpha is an upper bound, one at or above beta a lower bound.
    // Backends whose search takes no root window (honors_window() false)
    // search the full window.
    virtual FixedDepthResult search_window(const Position& root, const GameHistory& history, int depth,
                                           const std::vector<Move>& pv_hint, int alpha, int beta) {
        (void)alpha;
        (void)beta;
        return search_depth(root, history, depth, pv_hint);
    }
    virtual bool honors_window() const { return false; }

    // Safe to call from any thread while a search runs.
    virtual void request_stop() = 0;
    // Called before each new search (never while one runs).
    virtual void clear_stop() = 0;
    virtual void set_deadline(Clock::time_point deadline) = 0;

    // One line describing the variant, e.g. "gpu-independent quad/pesto/thread/staged/local split 2".
    virtual std::string describe() const = 0;

    // A new search (iterative deepening calls it once per "go"): entries of
    // earlier searches in the transposition table age. No-op without a table.
    virtual void new_search() {}
    // Empties the transposition table (a new game, UCI "Clear Hash").
    virtual void clear_hash() {}

    // Statistics of the last search_depth call, for profiling ("" if none).
    virtual std::string last_stats() const { return ""; }
};

// A root move's score and line (MultiPV).
struct RootLine {
    Move move;
    int score = 0;
    std::vector<Move> pv;  // starts with `move`
};

// One finished iteration of iterative deepening.
struct IterationReport {
    int depth = 0;
    int score = 0;
    u64 nodes = 0;        // this iteration
    u64 total_nodes = 0;  // all iterations so far
    double ms = 0;        // since the search started
    std::vector<Move> pv;
    std::vector<RootLine> lines;  // MultiPV: the best root moves, best first (empty for a single line)
};

// A search the engine runs, with its limits already turned into times.
struct SearchRequest {
    Position root;
    GameHistory history;
    int max_depth = kMaxPly - 1;
    u64 max_nodes = 0;  // checked between iterations (0: no limit)
    // No new iteration starts after `soft`; a running one is stopped at
    // `hard`. Clock::time_point::max() means no limit.
    Clock::time_point soft = Clock::time_point::max();
    Clock::time_point hard = Clock::time_point::max();
    // Aspiration windows (UCI Aspiration): from depth kAspirationMinDepth,
    // each iteration first searches the previous score +- this many
    // centipawns, widening the failed side (the margin doubling each time)
    // until the score falls inside. 0: full windows. Backends that take no
    // root window always search the full one.
    int aspiration_cp = 0;
    // A fail low (the score fell below an aspiration window: the best move so
    // far may be bad) lets the rest of that iteration run until `fail_low`
    // instead of `hard` (UCI FailLowTime). Clock::time_point::min(), the
    // default: no extension.
    Clock::time_point fail_low = Clock::time_point::min();
    // MultiPV (UCI MultiPV): report this many best root moves per iteration
    // (host/multipv.h). 1: the normal search.
    int multipv = 1;
};

constexpr int kAspirationMinDepth = 4;
constexpr int kMaxMultiPv = 256;  // UCI MultiPV's maximum

struct SearchOutcome {
    Move best;  // empty only if the root has no legal moves
    int score = 0;
    int depth = 0;  // deepest finished iteration
    int researches = 0;  // aspiration windows that failed and were searched again
    u64 nodes = 0;
    double ms = 0;
    std::vector<Move> pv;
};

// Iterative deepening from depth 1. Depth 1 always finishes (it ignores the
// limits), so there is a move whenever the root has one. `report` (may be
// empty) is called after every finished iteration. With request.multipv > 1,
// multipv_deepening (host/multipv.h) runs instead.
SearchOutcome iterative_deepening(SearchBackend& backend, const SearchRequest& request,
                                  const std::function<void(const IterationReport&)>& report);

// The CPU search as a backend, with its own table (host memory) for
// feature sets that use one.
class CpuBackend final : public SearchBackend {
  public:
    explicit CpuBackend(const CpuSearchOptions& options, size_t hash_mb = kDefaultHashMb);
    FixedDepthResult search_depth(const Position& root, const GameHistory& history, int depth,
                                  const std::vector<Move>& pv_hint) override {
        return search_window(root, history, depth, pv_hint, -kScoreInfinite, kScoreInfinite);
    }
    FixedDepthResult search_window(const Position& root, const GameHistory& history, int depth,
                                   const std::vector<Move>& pv_hint, int alpha, int beta) override;
    bool honors_window() const override { return true; }
    void request_stop() override { stop_.store(true); }
    void clear_stop() override { stop_.store(false); }
    void set_deadline(Clock::time_point deadline) override { deadline_ = deadline; }
    std::string describe() const override;
    void new_search() override {
        if (tt_) tt_->new_search();
    }
    void clear_hash() override {
        if (tt_) tt_->clear();
    }

  private:
    CpuSearchOptions options_;
    std::unique_ptr<HostTT> tt_;
    std::atomic<bool> stop_{false};
    Clock::time_point deadline_ = Clock::time_point::max();
};

// ABDADA on the CPU (core/abdada.h, cpu_search_abdada): `workers` threads
// search each iteration together, sharing the table.
class CpuAbdadaBackend final : public SearchBackend {
  public:
    CpuAbdadaBackend(const CpuSearchOptions& options, int workers, size_t hash_mb = kDefaultHashMb,
                     const AbdadaParams& params = {});
    FixedDepthResult search_depth(const Position& root, const GameHistory& history, int depth,
                                  const std::vector<Move>& pv_hint) override {
        return search_window(root, history, depth, pv_hint, -kScoreInfinite, kScoreInfinite);
    }
    FixedDepthResult search_window(const Position& root, const GameHistory& history, int depth,
                                   const std::vector<Move>& pv_hint, int alpha, int beta) override;
    bool honors_window() const override { return true; }
    void request_stop() override { stop_.store(true); }
    void clear_stop() override { stop_.store(false); }
    void set_deadline(Clock::time_point deadline) override { deadline_ = deadline; }
    std::string describe() const override;
    std::string last_stats() const override;
    void new_search() override {
        if (tt_) tt_->new_search();
    }
    void clear_hash() override {
        if (tt_) tt_->clear();
    }

  private:
    CpuSearchOptions options_;
    int workers_;
    AbdadaParams params_;
    std::unique_ptr<HostTT> tt_;
    AbdadaTable table_;
    AbdadaStats stats_;
    std::atomic<bool> stop_{false};
    Clock::time_point deadline_ = Clock::time_point::max();
};

// Time for one move, from the UCI "go" parameters of the side to move.
struct TimeControl {
    double time_ms = -1;  // remaining clock time (-1: none given)
    double inc_ms = 0;
    int moves_to_go = 0;  // 0: unknown (sudden death)
    double move_time_ms = -1;
    double overhead_ms = 50;  // reserved for communication and process latency
};

struct TimeBudget {
    double soft_ms = -1;  // don't start a new iteration after this (-1: no limit)
    double hard_ms = -1;  // stop the running iteration at this (-1: no limit)
    // After a fail low, stop at this instead (SearchRequest::fail_low); equal
    // to hard_ms when there is no extension.
    double fail_low_ms = -1;
};

// fail_low_percent (UCI FailLowTime): how much longer than hard_ms an
// iteration that failed low may run, at most 40% of the clock left; 0: no
// extension, and none for a fixed move time.
TimeBudget plan_time(const TimeControl& tc, int fail_low_percent = 0);

}  // namespace cb

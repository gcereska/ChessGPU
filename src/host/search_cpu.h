#pragma once

// Single-threaded CPU search: the reference the GPU searches are checked
// against, and a one-core baseline for comparing speed.

#include <algorithm>
#include <atomic>
#include <chrono>
#include <string>
#include <vector>

#include "core/search.h"
#include "host/game_history.h"
#include "host/variants.h"

namespace cb {

struct CpuSearchOptions {
    BoardKind board = BoardKind::Quad;
    EvalKind eval = EvalKind::Pesto;
    OrderingKind ordering = OrderingKind::Staged;
    FeatureKind features = FeatureKind::None;
    TTLayoutKind tt_layout = TTLayoutKind::Wide128;   // feature sets with a TT
    TTDesc tt{};  // the table (host memory, e.g. a HostTT): required by feature sets with a TT
    PruneParams prune{};  // settings of the feature set's pruning techniques
};

// "quad/pesto/staged", plus " tt wide128" with a table and the pruning
// settings the feature set uses (describe_prune).
std::string describe(const CpuSearchOptions& options);

struct FixedDepthResult {
    SearchResult search;
    std::vector<Move> pv;  // empty if the search was aborted
};

// One search of the root at `depth` with window (alpha, beta). `pv_hint` is
// the PV used for ordering (as from a previous iteration). Stops early
// (search.aborted) once *stop is true (stop may be null) or `deadline`
// passes.
FixedDepthResult cpu_search(const Position& root, const GameHistory& history, int depth, int alpha, int beta,
                            const CpuSearchOptions& options, const std::vector<Move>& pv_hint,
                            const std::atomic<bool>* stop,
                            std::chrono::steady_clock::time_point deadline = std::chrono::steady_clock::time_point::max());

// The same without stopping: for tests and benchmarks.
FixedDepthResult cpu_search_fixed(const Position& root, const GameHistory& history, int depth, int alpha, int beta,
                                  const CpuSearchOptions& options, const std::vector<Move>& pv_hint = {});

// The table of positions being searched for ABDADA (core/abdada.h): 2^bits
// slots of 64 bits.
class AbdadaTable {
  public:
    explicit AbdadaTable(int bits = 18) : slots_(size_t(1) << bits, 0) {}
    u64* slots() { return slots_.data(); }
    u64 mask() const { return u64(slots_.size() - 1); }
    void clear() { std::fill(slots_.begin(), slots_.end(), u64(0)); }
    // Positions claimed now (tests: a search that ends without stopping a
    // worker releases every claim).
    size_t claims() const { return size_t(std::count_if(slots_.begin(), slots_.end(), [](u64 s) { return s != 0; })); }

  private:
    std::vector<u64> slots_;
};

struct AbdadaStats {
    int workers = 0;
    int winner = -1;      // the worker whose result was taken
    u64 nodes = 0;        // all workers
    u64 deferred = 0;     // moves put back because another worker was searching them
};

// ABDADA on the CPU (core/abdada.h): `workers` threads search the root
// together with list ordering (deferring needs a move list), sharing
// options.tt and `table`; the first to finish answers. With one worker it is
// exactly cpu_search with list ordering; with more, nondeterministic (the
// score is still an alpha-beta score of the root; without a table, exactly
// the sequential one). Stops early, aborted, like cpu_search.
FixedDepthResult cpu_search_abdada(const Position& root, const GameHistory& history, int depth, int alpha, int beta,
                                   const CpuSearchOptions& options, const std::vector<Move>& pv_hint, int workers,
                                   AbdadaTable& table, const std::atomic<bool>* stop = nullptr,
                                   std::chrono::steady_clock::time_point deadline =
                                       std::chrono::steady_clock::time_point::max(),
                                   AbdadaStats* stats = nullptr, const AbdadaParams& params = {});

// Throws std::invalid_argument for settings out of range.
void validate_abdada(const AbdadaParams& params);
std::string describe(const AbdadaParams& params);  // "" for the defaults

// Move arena size for ListOrdering on the CPU: big enough that no real
// search can fill it.
constexpr int kCpuArenaMoves = 8192;

}  // namespace cb

// Aspiration windows (host/search_backend.h, SearchRequest::aspiration_cp):
// - without a table, where every window that holds the score gives the
//   exact score, iterative deepening with aspiration finds the same score and
//   best move at every depth as with full windows, and narrow margins do
//   fail and search again;
// - with a table, every iteration still ends with a legal move;
// - backends without root windows ignore the margin;
// - after a fail low, the rest of that iteration runs until
//   SearchRequest::fail_low (UCI FailLowTime), and the next iteration is back
//   at the hard limit; with the default, nothing changes; plan_time's
//   extension is the given share of the hard limit, at most 40% of the
//   clock, and none for a fixed move time.

#include <chrono>
#include <cmath>
#include <string>
#include <utility>
#include <vector>

#include "host/fen.h"
#include "host/host_tables.h"
#include "host/search_backend.h"
#include "host/uci_move.h"
#include "test_harness.h"
#include "test_util.h"

using namespace cb;

namespace {

#ifdef NDEBUG
constexpr int kDepth = 6;
#else
constexpr int kDepth = 4;
#endif

struct Iteration {
    int depth;
    int score;
    Move best;
};

std::vector<Iteration> deepen(SearchBackend& backend, const Position& p, int aspiration, int* researches = nullptr) {
    SearchRequest request;
    request.root = p;
    request.max_depth = kDepth;
    request.aspiration_cp = aspiration;
    backend.clear_hash();
    std::vector<Iteration> out;
    const SearchOutcome o = iterative_deepening(backend, request, [&](const IterationReport& r) {
        out.push_back({r.depth, r.score, r.pv.empty() ? Move() : r.pv.front()});
    });
    if (researches) *researches += o.researches;
    return out;
}

bool is_legal(const Position& p, Move m) {
    MoveList list;
    generate_legal(to_board<QuadBoard>(p), HqAttacks(host_attack_tables()), list);
    for (int i = 0; i < list.size; ++i)
        if (list.moves[i] == m) return true;
    return false;
}

// Returns a scripted score per depth (fail-soft: the score itself, wherever
// the window is) and records each search's depth, window and the deadline
// in force when it started.
class ScriptedBackend final : public SearchBackend {
  public:
    struct Call {
        int depth, alpha, beta;
        Clock::time_point deadline;
    };
    explicit ScriptedBackend(std::vector<int> scores) : scores_(std::move(scores)) {}
    FixedDepthResult search_depth(const Position& root, const GameHistory& history, int depth,
                                  const std::vector<Move>& pv_hint) override {
        return search_window(root, history, depth, pv_hint, -kScoreInfinite, kScoreInfinite);
    }
    FixedDepthResult search_window(const Position&, const GameHistory&, int depth, const std::vector<Move>&,
                                   int alpha, int beta) override {
        calls.push_back({depth, alpha, beta, deadline_});
        FixedDepthResult r;
        r.search.score = scores_[size_t(depth)];
        r.search.best = Move(E2, E4, DOUBLE_PUSH);
        r.search.nodes = 1;
        r.pv = {r.search.best};
        return r;
    }
    bool honors_window() const override { return true; }
    void request_stop() override {}
    void clear_stop() override {}
    void set_deadline(Clock::time_point deadline) override { deadline_ = deadline; }
    std::string describe() const override { return "scripted"; }

    std::vector<Call> calls;

  private:
    std::vector<int> scores_;  // by depth
    Clock::time_point deadline_ = Clock::time_point::max();
};

}  // namespace

TEST_CASE(aspiration_matches_full_windows_without_a_table) {
    CpuBackend backend(CpuSearchOptions{BoardKind::Quad, EvalKind::Pesto, OrderingKind::Staged});
    int researches = 0;
    for (const std::string& fen : testutil::test_fens()) {
        const Position p = testutil::parse_or_die(fen);
        const std::vector<Iteration> full = deepen(backend, p, 0);
        for (const int margin : {1, 10, 40}) {
            const std::vector<Iteration> asp = deepen(backend, p, margin, &researches);
            REQUIRE_MSG(asp.size() == full.size(), fen);
            for (size_t i = 0; i < full.size(); ++i) {
                const std::string where = fen + " margin " + std::to_string(margin) + " depth " +
                                          std::to_string(full[i].depth);
                CHECK_MSG(asp[i].score == full[i].score && asp[i].best == full[i].best,
                          where + ": " + std::to_string(asp[i].score) + " " + move_to_uci(asp[i].best) + " vs " +
                              std::to_string(full[i].score) + " " + move_to_uci(full[i].best));
            }
        }
    }
    CHECK_MSG(researches > 10, std::to_string(researches) + " re-searches");
}

TEST_CASE(aspiration_with_a_table) {
    CpuSearchOptions o{BoardKind::Quad, EvalKind::Terms, OrderingKind::Staged};
    o.features = FeatureKind::TTKillersSee;
    CpuBackend backend(o, 16);
    int researches = 0;
    for (const std::string& fen : testutil::test_fens()) {
        const Position p = testutil::parse_or_die(fen);
        const std::vector<Iteration> asp = deepen(backend, p, 15, &researches);
        REQUIRE_MSG(!asp.empty(), fen);
        MoveList list;
        generate_legal(to_board<QuadBoard>(p), HqAttacks(host_attack_tables()), list);
        for (const Iteration& it : asp)
            CHECK_MSG(list.size == 0 ? it.best.is_none() : is_legal(p, it.best),
                      fen + " depth " + std::to_string(it.depth) + ": " + move_to_uci(it.best));
    }
    CHECK(researches > 0);
}

TEST_CASE(aspiration_fail_low_extends_the_iteration) {
    // Scores by depth: steady, then a drop at depth 5 (a fail low below the
    // 30 cp window), steady again, then a rise at depth 7 (a fail high,
    // which gets no extension).
    const std::vector<int> scores = {0, 50, 50, 50, 50, -200, -190, -100};
    const Clock::time_point now = Clock::now();
    for (const bool extend : {true, false}) {
        ScriptedBackend backend(scores);
        SearchRequest request;
        request.root = testutil::parse_or_die(testutil::test_fens().front());
        request.max_depth = 7;
        request.aspiration_cp = 30;
        request.hard = now + std::chrono::hours(1);
        if (extend) request.fail_low = now + std::chrono::hours(2);
        const SearchOutcome o = iterative_deepening(backend, request, {});
        CHECK(o.depth == 7 && o.score == -100 && o.researches == 2);
        // Depths 1-4 once each, depth 5 twice (the fail low, then the
        // wider window), depth 6 once, depth 7 twice (the fail high).
        REQUIRE_MSG(backend.calls.size() == 9, std::to_string(backend.calls.size()));
        for (size_t i = 0; i < backend.calls.size(); ++i) {
            const ScriptedBackend::Call& c = backend.calls[i];
            const bool researched = i == 5;  // depth 5's second search
            const Clock::time_point expected =
                c.depth == 1 ? Clock::time_point::max() : (researched && extend ? request.fail_low : request.hard);
            CHECK_MSG(c.deadline == expected, (extend ? "extend: " : "default: ") + std::string("call ") +
                                                  std::to_string(i) + " depth " + std::to_string(c.depth));
        }
        CHECK(backend.calls[4].depth == 5 && backend.calls[4].alpha == 20 && backend.calls[4].beta == 80);
        CHECK(backend.calls[5].depth == 5 && backend.calls[5].alpha == -260 && backend.calls[5].beta == 80);
        CHECK(backend.calls[6].depth == 6 && backend.calls[6].alpha == -230 && backend.calls[6].beta == -170);
        CHECK(backend.calls[7].depth == 7 && backend.calls[7].alpha == -220 && backend.calls[7].beta == -160);
        CHECK(backend.calls[8].depth == 7 && backend.calls[8].alpha == -220 && backend.calls[8].beta == -40);
    }
}

TEST_CASE(plan_time_fail_low_extension) {
    TimeControl tc;
    tc.time_ms = 60000;
    tc.overhead_ms = 50;
    const TimeBudget none = plan_time(tc);
    CHECK(none.fail_low_ms == none.hard_ms);
    // 60 s: the hard limit doubles (well under 40% of the clock).
    const TimeBudget doubled = plan_time(tc, 100);
    CHECK(doubled.hard_ms == none.hard_ms && doubled.soft_ms == none.soft_ms);
    CHECK_MSG(std::abs(doubled.fail_low_ms - 2 * none.hard_ms) < 1e-6,
              std::to_string(doubled.fail_low_ms) + " vs " + std::to_string(none.hard_ms));
    // 1 s left: capped at 40% of the usable clock.
    tc.time_ms = 1000;
    const TimeBudget low = plan_time(tc, 400);
    CHECK_MSG(std::abs(low.fail_low_ms - 0.4 * 950) < 1e-6 && low.fail_low_ms > low.hard_ms,
              std::to_string(low.fail_low_ms) + " hard " + std::to_string(low.hard_ms));
    // A fixed move time is never extended; no clock means no limits.
    TimeControl fixed;
    fixed.move_time_ms = 1000;
    const TimeBudget f = plan_time(fixed, 400);
    CHECK(f.fail_low_ms == f.hard_ms);
    const TimeBudget infinite = plan_time(TimeControl{}, 400);
    CHECK(infinite.hard_ms < 0 && infinite.fail_low_ms < 0);
}

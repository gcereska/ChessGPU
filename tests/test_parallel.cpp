// Parallel search coordinators (independent subtrees, YBWC) with the CPU
// task runner: their scores and best moves must equal the sequential
// search's exactly.

#include <string>
#include <vector>

#include "host/game_history.h"
#include "host/search_cpu.h"
#include "host/split_search.h"
#include "host/task_runner.h"
#include "host/uci_move.h"
#include "core/ybwc_rules.h"
#include "host/ybwc.h"
#include "test_harness.h"
#include "test_util.h"

using namespace cb;

namespace {

#ifdef NDEBUG
constexpr int kMaxDepth = 5;
#else
constexpr int kMaxDepth = 3;
#endif

const CpuSearchOptions kOptions{BoardKind::Quad, EvalKind::Pesto, OrderingKind::Staged};

}  // namespace

TEST_CASE(parallel_independent_matches_sequential) {
    CpuTaskRunner runner(kOptions);
    for (const std::string& fen : testutil::test_fens()) {
        const Position p = testutil::parse_or_die(fen);
        std::vector<Move> pv;
        for (int depth = 1; depth <= kMaxDepth - 1; ++depth) {
            const FixedDepthResult seq = cpu_search_fixed(p, GameHistory{}, depth, -kScoreInfinite, kScoreInfinite,
                                                          kOptions, pv);
            for (int split = 0; split <= std::min(depth, 2); ++split) {
                const FixedDepthResult par = search_independent(runner, p, GameHistory{}, depth, split, pv);
                const std::string where = fen + " depth " + std::to_string(depth) + " split " + std::to_string(split);
                CHECK_MSG(par.search.score == seq.search.score, where);
                CHECK_MSG(par.search.best == seq.search.best, where);
                if (split == 0) CHECK_MSG(par.search.nodes == seq.search.nodes, where);
            }
            pv = seq.pv;
        }
    }
}

TEST_CASE(parallel_ybwc_matches_sequential) {
    CpuTaskRunner runner(kOptions);
    const YbwcOptions variants[] = {{0, 1, false, 0},  {0, 3, false, 0},    {1, 2, false, 0}, {1, 8, false, 0},
                                    {2, 8, false, 0},  {3, 8, false, 0},    {0, 8, false, 0}, {0, 8, true, 0},
                                    {1, 8, true, 0},   {2, 3, true, 0},     {3, 8, false, 2000},
                                    {4, 2, true, 3000}, {6, 1, false, 1500}, {8, 0, true, 1100},
                                    {8, 0, false, 1025}};
    int compared = 0;
    for (const std::string& fen : testutil::test_fens()) {
        const Position p = testutil::parse_or_die(fen);
        std::vector<Move> pv;
        for (int depth = 1; depth <= kMaxDepth; ++depth) {
            const FixedDepthResult seq = cpu_search_fixed(p, GameHistory{}, depth, -kScoreInfinite, kScoreInfinite,
                                                          kOptions, pv);
            for (const YbwcOptions& o : variants) {
                YbwcStats stats;
                const FixedDepthResult par = search_ybwc(runner, p, GameHistory{}, depth, o, pv, &stats);
                const std::string where = fen + " depth " + std::to_string(depth) + " leaf " +
                                          std::to_string(o.leaf_depth) + " plies " + std::to_string(o.max_explicit_ply);
                CHECK_MSG(par.search.score == seq.search.score, where + ": " + std::to_string(par.search.score) +
                                                                    " vs " + std::to_string(seq.search.score));
                CHECK_MSG(par.search.best == seq.search.best,
                          where + ": " + move_to_uci(par.search.best) + " vs " + move_to_uci(seq.search.best));
                CHECK_MSG(!par.search.aborted, where);
                if (!seq.search.best.is_none()) CHECK_MSG(!par.pv.empty() && par.pv.front() == par.search.best, where);
                ++compared;
            }
            pv = seq.pv;
        }
    }
    CHECK(compared > 500);
}

TEST_CASE(parallel_ybwc_with_game_history) {
    // Repetitions through the game history and the explicit nodes.
    struct Game {
        const char* fen;
        const char* moves;
    };
    const Game games[] = {
        {"6k1/5ppp/8/8/8/8/5PPP/3QR1K1 w - - 0 1", "g1h1 g8h8 h1g1 h8g8 g1h1 g8h8 h1g1"},
        {"7k/8/8/8/8/1q6/8/K2Q4 w - - 0 1", "d1h5 h8g7 h5g5 g7h7 g5h5 h7g7 h5g5 g7h7"},
        {kStartFen, "g1f3 g8f6 f3g1 f6g8"},
    };
    CpuTaskRunner runner(kOptions);
    for (const Game& g : games) {
        Position root;
        GameHistory history;
        std::string error;
        REQUIRE_MSG(build_game(testutil::parse_or_die(g.fen), g.moves, root, history, error), error);
        for (int depth = 1; depth <= kMaxDepth; ++depth) {
            const FixedDepthResult seq =
                cpu_search_fixed(root, history, depth, -kScoreInfinite, kScoreInfinite, kOptions);
            for (const YbwcOptions& o : {YbwcOptions{0, 8, false}, YbwcOptions{1, 3, true}}) {
                const FixedDepthResult par = search_ybwc(runner, root, history, depth, o);
                CHECK_MSG(par.search.score == seq.search.score,
                          std::string(g.moves) + " depth " + std::to_string(depth) + ": " +
                              std::to_string(par.search.score) + " vs " + std::to_string(seq.search.score));
                CHECK_MSG(par.search.best == seq.search.best, std::string(g.moves) + " depth " + std::to_string(depth));
            }
        }
    }
}

TEST_CASE(parallel_ybwc_rules) {
    // Higher scores win, exact or not.
    CHECK(better_child(10, false, 5, 9, true, 1));
    CHECK(!better_child(8, true, 0, 9, false, 1));
    // Equal scores: an exact score beats a bound, even from a later move...
    CHECK(better_child(9, true, 4, 9, false, 1));
    // ...a bound never replaces an equal best...
    CHECK(!better_child(9, false, 0, 9, true, 4));
    CHECK(!better_child(9, false, 0, 9, false, 4));
    // ...and between exact scores the earlier move wins.
    CHECK(better_child(9, true, 2, 9, true, 4));
    CHECK(!better_child(9, true, 6, 9, true, 4));
    // Node types.
    CHECK(child_type(YbwcNodeType::Pv, 0) == YbwcNodeType::Pv);
    CHECK(child_type(YbwcNodeType::Pv, 3) == YbwcNodeType::Cut);
    CHECK(child_type(YbwcNodeType::Cut, 0) == YbwcNodeType::All);
    CHECK(child_type(YbwcNodeType::Cut, 7) == YbwcNodeType::All);
    CHECK(child_type(YbwcNodeType::All, 0) == YbwcNodeType::Cut);
}

TEST_CASE(parallel_ybwc_splits_quiescence) {
    // Capture-heavy positions whose quiescence searches run past a small
    // budget, so quiescence tasks get split into explicit quiescence nodes.
    const char* fens[] = {
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
        "r2q1rk1/pP1p2pp/Q4n2/bbp1p3/Np6/1B3NBn/pPPP1PPP/R3K2R b KQ - 0 1",
    };
    CpuTaskRunner runner(kOptions);
    size_t quiescence_nodes = 0;
    for (const char* fen : fens) {
        const Position p = testutil::parse_or_die(fen);
        for (int depth = 0; depth <= 3; ++depth) {
            const FixedDepthResult seq =
                cpu_search_fixed(p, GameHistory{}, depth, -kScoreInfinite, kScoreInfinite, kOptions);
            for (const bool waves : {false, true}) {
                YbwcStats stats;
                const FixedDepthResult par =
                    search_ybwc(runner, p, GameHistory{}, depth, YbwcOptions{0, 8, waves, 1025}, {}, &stats);
                const std::string where = std::string(fen) + " depth " + std::to_string(depth);
                CHECK_MSG(par.search.score == seq.search.score, where + ": " + std::to_string(par.search.score) +
                                                                    " vs " + std::to_string(seq.search.score));
                CHECK_MSG(par.search.best == seq.search.best, where);
                quiescence_nodes += stats.explicit_quiescence;
            }
        }
    }
    CHECK(quiescence_nodes > 100);
}

TEST_CASE(parallel_ybwc_is_deterministic) {
    CpuTaskRunner runner(kOptions);
    const Position p = testutil::parse_or_die(testutil::test_fens()[1]);
    YbwcStats s1;
    YbwcStats s2;
    const FixedDepthResult a = search_ybwc(runner, p, GameHistory{}, 4, YbwcOptions{1, 8}, {}, &s1);
    const FixedDepthResult b = search_ybwc(runner, p, GameHistory{}, 4, YbwcOptions{1, 8}, {}, &s2);
    CHECK_EQ(a.search.nodes, b.search.nodes);
    CHECK(a.pv == b.pv);
    CHECK_EQ(s1.rounds, s2.rounds);
    CHECK_EQ(s1.tasks, s2.tasks);
    CHECK(s1.rounds > 1);
    CHECK(s1.explicit_nodes > 1);
}

TEST_CASE(parallel_runner_stop) {
    CpuTaskRunner runner(kOptions);
    runner.request_stop();
    const FixedDepthResult r = search_ybwc(runner, testutil::parse_or_die(kStartFen), GameHistory{}, 8, YbwcOptions{});
    CHECK(r.search.aborted);
    runner.clear_stop();
    CHECK(!search_ybwc(runner, testutil::parse_or_die(kStartFen), GameHistory{}, 3, YbwcOptions{}).search.aborted);
}

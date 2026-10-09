// GPU search: every kernel variant must return exactly what the CPU search
// returns for the same task (score, best move, node count), and the
// parallel searches must match the sequential search.

#include <chrono>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "core/search.h"
#include "gpu/search_gpu.h"
#include "gpu/variants.h"
#include "gpu_search_util.h"
#include "gpu_test_env.h"
#include "host/host_tables.h"
#include "host/search_cpu.h"
#include "host/split_search.h"
#include "host/ybwc.h"
#include "gpu/ybwc_device.h"
#include "host/uci_move.h"
#include "test_harness.h"
#include "test_util.h"

using namespace cb;

namespace {

using gpu_search_util::compiled_variants;
using gpu_search_util::host_attacks;

// Tasks from the test positions: roots (on the PV, with a PV hint) and
// positions one and two plies down linked to their ancestors through the
// history, with assorted windows and depths.
struct TaskSet {
    std::vector<TaskSpec> tasks;
    std::vector<HistoryEntry> history;
    std::vector<Move> pv;  // hint for root tasks: the root's first legal move, then nothing
};

TaskSet make_tasks(bool quick) {
    static const int kWindows[][2] = {
        {-kScoreInfinite, kScoreInfinite}, {-50, 50}, {0, 1}, {-1, 0}, {100, 400}, {-400, -100}, {-30000, 30000}};
    const int max_depth = quick ? 2 : 3;
    const size_t fens = quick ? 6 : testutil::test_fens().size();
    TaskSet s;
    int window = 0;
    const auto next_window = [&](TaskSpec& t) {
        t.alpha = kWindows[window % 7][0];
        t.beta = kWindows[window % 7][1];
        ++window;
    };
    for (size_t f = 0; f < fens; ++f) {
        const Position root = testutil::parse_or_die(testutil::test_fens()[f]);
        const QuadBoard b = to_board<QuadBoard>(root);
        for (int depth = 0; depth <= max_depth; ++depth) {
            TaskSpec t;
            t.board = to_board<QuadBoard>(root);
            t.depth = depth;
            next_window(t);
            t.flags = depth % 2 ? kTaskOnPv : 0;
            s.tasks.push_back(t);
        }
        // One and two plies down, through the first few moves.
        HistoryEntry e0;
        e0.key = compute_key(b, host_attacks(), host_search_tables().zobrist);
        e0.prev = -1;
        e0.ply = 0;
        e0.repeated = 0;
        s.history.push_back(e0);
        const int root_entry = int(s.history.size()) - 1;
        MoveList moves;
        generate_legal(b, host_attacks(), moves);
        for (int i = 0; i < moves.size && i < 3; ++i) {
            QuadBoard child = b;
            make_move(child, moves.moves[i]);
            TaskSpec t1;
            t1.board = child;
            t1.depth = (i % max_depth) + 1;
            t1.ply = 1;
            t1.history = root_entry;
            next_window(t1);
            s.tasks.push_back(t1);
            HistoryEntry e1;
            e1.key = compute_key(child, host_attacks(), host_search_tables().zobrist);
            e1.prev = root_entry;
            e1.ply = 1;
            e1.repeated = 0;
            s.history.push_back(e1);
            const int child_entry = int(s.history.size()) - 1;
            MoveList grand;
            generate_legal(child, host_attacks(), grand);
            for (int j = 0; j < grand.size && j < 2; ++j) {
                QuadBoard g = child;
                make_move(g, grand.moves[j]);
                TaskSpec t2;
                t2.board = g;
                t2.depth = max_depth - j;
                t2.ply = 2;
                t2.history = child_entry;
                next_window(t2);
                s.tasks.push_back(t2);
            }
        }
    }
    return s;
}

// The CPU's result for each task, with the same history and PV.
template <class Eval>
std::vector<SearchResult> cpu_results(const TaskSet& s) {
    using Level = SearchLevel<StagedPicker::State>;
    auto stack = std::make_unique<LocalStack<Level, kMaxPly + 1>>();
    StagedOrdering::Workspace ws;
    NoStop control;
    const SearchEnv<HqAttacks> env{host_attacks(), &host_search_tables(), s.history.empty() ? nullptr : s.history.data(),
                                   s.pv.empty() ? nullptr : s.pv.data(), int(s.pv.size())};
    std::vector<SearchResult> out;
    for (const TaskSpec& g : s.tasks) {
        SearchTask<QuadBoard> t;
        t.board = g.board;
        t.alpha = short(g.alpha);
        t.beta = short(g.beta);
        t.depth = static_cast<signed char>(g.depth);
        t.ply = u8(g.ply);
        t.flags = g.flags;
        t.history = g.history;
        out.push_back(search_task<QuadBoard, Eval, StagedOrdering, LocalStack<Level, kMaxPly + 1>, NoStop, false>(
            t, env, *stack, ws, control, nullptr));
    }
    return out;
}

template <class Eval>
void check_tasks_match_cpu(EvalKind eval, AttacksKind attacks = AttacksKind::Hq) {
    const TaskSet s = make_tasks(gpu_test_env::quick());
    const std::vector<SearchResult> expected = cpu_results<Eval>(s);
    int variants = 0;
    for (const BoardKind board : compiled_boards()) {
        for (GpuSearchOptions o : compiled_variants(board, eval)) {
            o.attacks = attacks;
            ++variants;
            GpuSearcher searcher(gpu_test_env::context(), o);
            const std::vector<SearchResult> got = searcher.run(s.tasks, s.history, 0, s.pv);
            REQUIRE(got.size() == expected.size());
            for (size_t i = 0; i < got.size(); ++i) {
                const std::string where = describe(o) + " task " + std::to_string(i) + " (" +
                                          format_fen(to_position(s.tasks[i].board)) + " depth " +
                                          std::to_string(s.tasks[i].depth) + ")";
                CHECK_MSG(got[i].score == expected[i].score, where + ": score " + std::to_string(got[i].score) +
                                                                 " vs " + std::to_string(expected[i].score));
                CHECK_MSG(got[i].best == expected[i].best,
                          where + ": best " + move_to_uci(got[i].best) + " vs " + move_to_uci(expected[i].best));
                CHECK_MSG(got[i].nodes == expected[i].nodes, where + ": nodes " + std::to_string(got[i].nodes) +
                                                                 " vs " + std::to_string(expected[i].nodes));
                CHECK_MSG(!got[i].aborted, where + ": aborted");
            }
        }
    }
    CHECK(variants > 0);
}

}  // namespace

#if CHESSGPU_WITH_PESTO_EVAL
TEST_CASE(gpu_search_tasks_match_cpu_pesto) { check_tasks_match_cpu<PestoEval>(EvalKind::Pesto); }
#endif
#if CHESSGPU_WITH_MATERIAL_EVAL
TEST_CASE(gpu_search_tasks_match_cpu_material) { check_tasks_match_cpu<MaterialEval>(EvalKind::Material); }
#endif
#if CHESSGPU_WITH_TERMS_EVAL
TEST_CASE(gpu_search_tasks_match_cpu_terms) { check_tasks_match_cpu<PestoTermsEval>(EvalKind::Terms); }
#endif
#if CHESSGPU_WITH_NNUE_EVAL
// The NNUE (core/nnue.h) with the engine's default network: integer
// arithmetic, so every variant equals the CPU exactly.
TEST_CASE(gpu_search_tasks_match_cpu_nnue) { check_tasks_match_cpu<NnueEval>(EvalKind::Nnue); }
#endif
#if CHESSGPU_WITH_MAGIC_SEARCH_ATTACKS && CHESSGPU_WITH_TERMS_EVAL
// Magic bitboards in the search kernels, with the eval whose mobility uses them.
TEST_CASE(gpu_search_tasks_match_cpu_magic) { check_tasks_match_cpu<PestoTermsEval>(EvalKind::Terms, AttacksKind::Magic); }
#endif

TEST_CASE(gpu_search_independent_matches_cpu) {
    const bool quick = gpu_test_env::quick();
    const size_t fens = quick ? 4 : testutil::test_fens().size();
    const int max_depth = quick ? 2 : 4;
    std::vector<GpuSearchOptions> variants;
    for (const ExecModel model : compiled_models()) {
        GpuSearchOptions o;
        o.board = is_compiled(BoardKind::Quad) ? BoardKind::Quad : compiled_boards().front();
        o.eval = is_compiled(EvalKind::Pesto) ? EvalKind::Pesto : EvalKind::Material;
        o.model = model;
        o.ordering = is_compiled(OrderingKind::Staged) ? OrderingKind::Staged : OrderingKind::List;
        o.stack = is_compiled(StackKind::Local) ? StackKind::Local : StackKind::Shared;
        if (model == ExecModel::Warp) o.block_size = 64;
        variants.push_back(o);
    }
    for (const GpuSearchOptions& o : variants) {
        GpuSearcher searcher(gpu_test_env::context(), o);
        for (size_t f = 0; f < fens; ++f) {
            const Position p = testutil::parse_or_die(testutil::test_fens()[f]);
            std::vector<Move> pv;
            for (int depth = 1; depth <= max_depth; ++depth) {
                const FixedDepthResult cpu = cpu_search_fixed(p, GameHistory{}, depth, -kScoreInfinite, kScoreInfinite,
                                                              {BoardKind::Quad, o.eval, OrderingKind::Staged}, pv);
                // Split 0 searches on one GPU thread: slow, so only to depth 3.
                for (int split = depth <= 3 ? 0 : 1; split <= std::min(depth, 2); ++split) {
                    const FixedDepthResult gpu = search_independent(searcher, p, GameHistory{}, depth, split, pv);
                    const std::string where = describe(o) + " " + testutil::test_fens()[f] + " depth " +
                                              std::to_string(depth) + " split " + std::to_string(split);
                    CHECK_MSG(gpu.search.score == cpu.search.score, where + ": score " +
                                                                        std::to_string(gpu.search.score) + " vs " +
                                                                        std::to_string(cpu.search.score));
                    CHECK_MSG(gpu.search.best == cpu.search.best, where + ": best " + move_to_uci(gpu.search.best) +
                                                                      " vs " + move_to_uci(cpu.search.best));
                    CHECK_MSG(!gpu.search.aborted, where);
                    // With no split, the GPU runs the CPU's exact search.
                    if (split == 0) CHECK_MSG(gpu.search.nodes == cpu.search.nodes, where + ": nodes");
                }
                pv = cpu.pv;
            }
        }
    }
}

namespace {

struct YbwcCase {
    size_t fen;
    int depth;
    YbwcOptions options;
    FixedDepthResult cpu;
    YbwcStats cpu_stats;
};

// YBWC with the CPU runner on a set of positions, depths and options.
std::vector<YbwcCase> cpu_ybwc_cases(EvalKind eval, size_t fens, int max_depth) {
    CpuTaskRunner cpu({BoardKind::Quad, eval, OrderingKind::Staged});
    std::vector<YbwcCase> cases;
    for (size_t f = 0; f < fens; ++f) {
        const Position p = testutil::parse_or_die(testutil::test_fens()[f]);
        for (int depth = 1; depth <= max_depth; ++depth) {
            for (const YbwcOptions& y : {YbwcOptions{1, 8}, YbwcOptions{2, 3}}) {
                YbwcCase c{f, depth, y, {}, {}};
                c.cpu = search_ybwc(cpu, p, GameHistory{}, depth, y, {}, &c.cpu_stats);
                cases.push_back(c);
            }
        }
    }
    return cases;
}

void check_ybwc_cases(GpuSearcher& gpu, const std::vector<YbwcCase>& cases) {
    for (const YbwcCase& c : cases) {
        const Position p = testutil::parse_or_die(testutil::test_fens()[c.fen]);
        YbwcStats gs;
        const FixedDepthResult g = search_ybwc(gpu, p, GameHistory{}, c.depth, c.options, {}, &gs);
        const std::string where = describe(gpu.options()) + " " + testutil::test_fens()[c.fen] + " depth " +
                                  std::to_string(c.depth) + " leaf " + std::to_string(c.options.leaf_depth);
        CHECK_MSG(g.search.score == c.cpu.search.score, where + ": score");
        CHECK_MSG(g.search.best == c.cpu.search.best, where + ": best");
        CHECK_MSG(g.search.nodes == c.cpu.search.nodes,
                  where + ": nodes " + std::to_string(g.search.nodes) + " vs " + std::to_string(c.cpu.search.nodes));
        CHECK_MSG(g.pv == c.cpu.pv, where + ": pv");
        CHECK_MSG(gs.rounds == c.cpu_stats.rounds && gs.tasks == c.cpu_stats.tasks, where + ": rounds/tasks");
    }
}

}  // namespace

TEST_CASE(gpu_ybwc_matches_cpu_runner) {
    // YBWC is deterministic given the task results, and GPU tasks return
    // exactly what CPU tasks return: the whole search must be identical.
    // Every compiled variant on a few positions; the default variant deeper.
    const bool quick = gpu_test_env::quick();
    for (const EvalKind eval : {EvalKind::Material, EvalKind::Pesto, EvalKind::Terms, EvalKind::Nnue}) {
        if (!is_compiled(eval)) continue;
        const std::vector<YbwcCase> small = cpu_ybwc_cases(eval, quick ? 2 : 6, 3);
        for (const BoardKind board : compiled_boards()) {
            for (const GpuSearchOptions& o : compiled_variants(board, eval)) {
                GpuSearcher gpu(gpu_test_env::context(), o);
                check_ybwc_cases(gpu, small);
            }
        }
    }
    if (quick) return;
    GpuSearchOptions o;
    o.board = is_compiled(BoardKind::Quad) ? BoardKind::Quad : compiled_boards().front();
    o.eval = is_compiled(EvalKind::Pesto) ? EvalKind::Pesto : EvalKind::Material;
    o.model = compiled_models().front();
    if (o.model == ExecModel::Warp) o.block_size = 64;
    GpuSearcher gpu(gpu_test_env::context(), o);
    check_ybwc_cases(gpu, cpu_ybwc_cases(o.eval, testutil::test_fens().size(), 5));
}

TEST_CASE(gpu_ybwc_device_matches_sequential) {
    // Nondeterministic, but the score and the best move are exactly the
    // sequential search's in every run.
    const bool quick = gpu_test_env::quick();
    const size_t fens = quick ? 3 : testutil::test_fens().size();
    const int max_depth = quick ? 3 : 5;
    const int runs = quick ? 1 : 2;
    const YbwcOptions options[] = {{1, 8, false, 0}, {2, 3, true, 0}, {3, 8, false, 1500}, {8, 0, true, 1100}};
    std::vector<GpuSearchOptions> variants;
    for (const BoardKind board : compiled_boards())
        for (const EvalKind eval : {EvalKind::Material, EvalKind::Pesto, EvalKind::Terms, EvalKind::Nnue})
            if (is_compiled(eval))
                for (const GpuSearchOptions& o : compiled_variants(board, eval)) variants.push_back(o);
    if (quick) {
        // Under compute-sanitizer (~100x slower): one board and eval, each
        // way of using memory (local stack, shared window, warp) once.
        std::vector<GpuSearchOptions> some;
        for (const GpuSearchOptions& o : variants)
            if (o.board == variants.front().board && o.eval == variants.front().eval) some.push_back(o);
        variants = some;
    }
    for (const GpuSearchOptions& o : variants) {
        const bool full = !quick && o.board == variants.front().board && o.eval == variants.front().eval;
        for (const int lanes : {1, 4}) {
            if (o.model == ExecModel::Warp && lanes != 1) continue;
            YbwcDeviceLimits limits;
            limits.lanes = lanes;
            GpuYbwcDevice device(gpu_test_env::context(), o, limits);
            for (size_t f = 0; f < (full ? fens : std::min<size_t>(fens, 3)); ++f) {
                const Position p = testutil::parse_or_die(testutil::test_fens()[f]);
                std::vector<Move> pv;
                for (int depth = 1; depth <= (full ? max_depth : 3); ++depth) {
                    const FixedDepthResult seq = cpu_search_fixed(p, GameHistory{}, depth, -kScoreInfinite,
                                                                  kScoreInfinite, {BoardKind::Quad, o.eval}, pv);
                    for (const YbwcOptions& y : options) {
                        for (int run = 0; run < runs; ++run) {
                            YbwcDeviceStats s;
                            const FixedDepthResult r = device.search(p, GameHistory{}, depth, y, pv, &s);
                            const std::string where = describe(o) + " lanes " + std::to_string(lanes) + " " +
                                                      testutil::test_fens()[f] + " depth " + std::to_string(depth) +
                                                      " leaf " + std::to_string(y.leaf_depth) + " budget " +
                                                      std::to_string(y.node_budget);
                            CHECK_MSG(!r.search.aborted && !s.overflow, where + ": aborted");
                            CHECK_MSG(r.search.score == seq.search.score, where + ": score " +
                                                                              std::to_string(r.search.score) + " vs " +
                                                                              std::to_string(seq.search.score));
                            CHECK_MSG(r.search.best == seq.search.best,
                                      where + ": best " + move_to_uci(r.search.best) + " vs " +
                                          move_to_uci(seq.search.best));
                        }
                    }
                    pv = seq.pv;
                }
            }
        }
    }
}

TEST_CASE(gpu_ybwc_device_splits_quiescence) {
    // As parallel_ybwc_splits_quiescence: split quiescence tasks, exact results.
    const char* fens[] = {
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
        "r2q1rk1/pP1p2pp/Q4n2/bbp1p3/Np6/1B3NBn/pPPP1PPP/R3K2R b KQ - 0 1",
    };
    for (const ExecModel model : compiled_models()) {
        GpuSearchOptions o;
        o.board = is_compiled(BoardKind::Quad) ? BoardKind::Quad : compiled_boards().front();
        o.eval = is_compiled(EvalKind::Pesto) ? EvalKind::Pesto : EvalKind::Material;
        o.model = model;
        if (model == ExecModel::Warp) o.block_size = 64;
        GpuYbwcDevice device(gpu_test_env::context(), o);
        for (const char* fen : fens) {
            const Position p = testutil::parse_or_die(fen);
            for (int depth = 0; depth <= (gpu_test_env::quick() ? 1 : 3); ++depth) {
                const FixedDepthResult seq =
                    cpu_search_fixed(p, GameHistory{}, depth, -kScoreInfinite, kScoreInfinite, {BoardKind::Quad, o.eval});
                for (const bool waves : {false, true}) {
                    const FixedDepthResult r = device.search(p, GameHistory{}, depth, YbwcOptions{0, 8, waves, 1025});
                    const std::string where = describe(o) + " " + fen + " depth " + std::to_string(depth);
                    CHECK_MSG(r.search.score == seq.search.score, where);
                    CHECK_MSG(r.search.best == seq.search.best, where);
                }
            }
        }
    }
}

TEST_CASE(gpu_ybwc_device_game_history_and_stop) {
    Position root;
    GameHistory history;
    std::string error;
    REQUIRE_MSG(build_game(testutil::parse_or_die("6k1/5ppp/8/8/8/8/5PPP/3QR1K1 w - - 0 1"),
                           "g1h1 g8h8 h1g1 h8g8 g1h1 g8h8 h1g1", root, history, error),
                error);
    GpuSearchOptions o;
    o.board = is_compiled(BoardKind::Quad) ? BoardKind::Quad : compiled_boards().front();
    o.eval = is_compiled(EvalKind::Pesto) ? EvalKind::Pesto : EvalKind::Material;
    o.model = compiled_models().front();
    if (o.model == ExecModel::Warp) o.block_size = 64;
    GpuYbwcDevice device(gpu_test_env::context(), o);
    for (int depth = 1; depth <= (gpu_test_env::quick() ? 3 : 5); ++depth) {
        const FixedDepthResult seq =
            cpu_search_fixed(root, history, depth, -kScoreInfinite, kScoreInfinite, {BoardKind::Quad, o.eval});
        const FixedDepthResult r = device.search(root, history, depth, YbwcOptions{1, 8, false, 0});
        CHECK_MSG(r.search.score == seq.search.score, "depth " + std::to_string(depth));
        CHECK_MSG(r.search.best == seq.search.best, "depth " + std::to_string(depth));
    }
    // Stopped before it starts, and while it runs.
    device.request_stop();
    CHECK(device.search(testutil::parse_or_die(kStartFen), GameHistory{}, 12, YbwcOptions{}).search.aborted);
    device.clear_stop();
    const auto start = std::chrono::steady_clock::now();
    device.set_deadline(start + std::chrono::milliseconds(300));
    CHECK(device.search(testutil::parse_or_die(kStartFen), GameHistory{}, 14, YbwcOptions{}).search.aborted);
    const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    CHECK_MSG(seconds < 5.0, "deadline stop after " + std::to_string(seconds) + " s");
    device.set_deadline(std::chrono::steady_clock::time_point::max());
    CHECK(!device.search(testutil::parse_or_die(kStartFen), GameHistory{}, 3, YbwcOptions{}).search.aborted);
}

TEST_CASE(gpu_ybwc_device_overflow_is_reported) {
    // Pools far too small: the search must report it and come back aborted.
    GpuSearchOptions o;
    o.board = is_compiled(BoardKind::Quad) ? BoardKind::Quad : compiled_boards().front();
    o.eval = is_compiled(EvalKind::Pesto) ? EvalKind::Pesto : EvalKind::Material;
    o.model = compiled_models().front();
    if (o.model == ExecModel::Warp) o.block_size = 64;
    YbwcDeviceLimits limits;
    limits.max_nodes = 64;
    limits.max_moves = 1024;
    GpuYbwcDevice device(gpu_test_env::context(), o, limits);
    YbwcDeviceStats s;
    const FixedDepthResult r =
        device.search(testutil::parse_or_die(testutil::test_fens()[1]), GameHistory{}, 6, YbwcOptions{1, 8, false, 0}, {}, &s);
    CHECK(r.search.aborted);
    CHECK(s.overflow);
}

TEST_CASE(gpu_search_with_game_history) {
    // Repetitions through the game history: the GPU must apply the same rule.
    Position root;
    GameHistory history;
    std::string error;
    REQUIRE_MSG(build_game(testutil::parse_or_die("6k1/5ppp/8/8/8/8/5PPP/3QR1K1 w - - 0 1"),
                           "g1h1 g8h8 h1g1 h8g8 g1h1 g8h8 h1g1", root, history, error),
                error);
    GpuSearchOptions o;
    o.board = is_compiled(BoardKind::Quad) ? BoardKind::Quad : compiled_boards().front();
    o.eval = is_compiled(EvalKind::Pesto) ? EvalKind::Pesto : EvalKind::Material;
    o.model = compiled_models().front();
    if (o.model == ExecModel::Warp) o.block_size = 64;
    GpuSearcher searcher(gpu_test_env::context(), o);
    for (int depth = 1; depth <= (gpu_test_env::quick() ? 2 : 4); ++depth) {
        const FixedDepthResult cpu =
            cpu_search_fixed(root, history, depth, -kScoreInfinite, kScoreInfinite, {BoardKind::Quad, o.eval});
        for (int split = 0; split <= std::min(depth, 2); ++split) {
            const FixedDepthResult gpu = search_independent(searcher, root, history, depth, split);
            CHECK_MSG(gpu.search.score == cpu.search.score,
                      "depth " + std::to_string(depth) + " split " + std::to_string(split) + ": " +
                          std::to_string(gpu.search.score) + " vs " + std::to_string(cpu.search.score));
        }
    }
}

TEST_CASE(gpu_search_stop_flag) {
    GpuSearchOptions o;
    o.board = is_compiled(BoardKind::Quad) ? BoardKind::Quad : compiled_boards().front();
    o.eval = is_compiled(EvalKind::Pesto) ? EvalKind::Pesto : EvalKind::Material;
    for (const ExecModel model : compiled_models()) {
        o.model = model;
        o.block_size = model == ExecModel::Warp ? 64 : 128;
        GpuSearcher searcher(gpu_test_env::context(), o);
        TaskSpec t;
        t.board = to_board<QuadBoard>(testutil::parse_or_die(testutil::test_fens()[1]));
        t.depth = 12;  // far too deep to finish unless stopped
        searcher.request_stop();
        const std::vector<SearchResult> stopped = searcher.run({t}, {}, 0, {});
        REQUIRE(stopped.size() == 1);
        CHECK(stopped[0].aborted);
        CHECK_EQ(stopped[0].nodes, u64(1));  // the first node polls
        searcher.clear_stop();
        t.depth = 2;
        const std::vector<SearchResult> normal = searcher.run({t}, {}, 0, {});
        CHECK(!normal[0].aborted);
        CHECK(!normal[0].best.is_none());
    }
}

TEST_CASE(gpu_search_stop_while_running) {
    // The stop flag must reach a kernel that is already running.
    GpuSearchOptions o;
    o.board = is_compiled(BoardKind::Quad) ? BoardKind::Quad : compiled_boards().front();
    o.eval = is_compiled(EvalKind::Pesto) ? EvalKind::Pesto : EvalKind::Material;
    o.model = compiled_models().front();
    if (o.model == ExecModel::Warp) o.block_size = 64;
    GpuSearcher searcher(gpu_test_env::context(), o);
    TaskSpec t;
    t.board = to_board<QuadBoard>(testutil::parse_or_die(testutil::test_fens()[1]));
    t.depth = 20;  // hours on one GPU thread
    std::vector<SearchResult> results;
    const auto start = std::chrono::steady_clock::now();
    std::thread runner([&] { results = searcher.run({t}, {}, 0, {}); });
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    searcher.request_stop();
    runner.join();
    const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    REQUIRE(results.size() == 1);
    CHECK(results[0].aborted);
    CHECK(results[0].nodes > kStopPollInterval);  // it did run before being stopped
    CHECK_MSG(seconds < 5.0, "stopping took " + std::to_string(seconds) + " s");
    searcher.clear_stop();
}

namespace gpu_search_kernels {
// Spins until *flag becomes nonzero or timeout_ns passes; reports how long it waited.
__global__ void spin_until_flag(const volatile int* flag, unsigned long long timeout_ns, unsigned long long* waited) {
    unsigned long long start, now;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(start));
    do {
        asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(now));
    } while (*flag == 0 && now - start < timeout_ns);
    *waited = now - start;
}
}  // namespace gpu_search_kernels

TEST_CASE(gpu_async_copy_reaches_running_kernel) {
    // What stopping a search relies on (also under WDDM): a copy queued on a
    // second stream lands while a kernel keeps running and polling.
    cudaStream_t a, b;
    REQUIRE(cudaStreamCreateWithFlags(&a, cudaStreamNonBlocking) == cudaSuccess);
    REQUIRE(cudaStreamCreateWithFlags(&b, cudaStreamNonBlocking) == cudaSuccess);
    unsigned long long* waited = nullptr;
    int* d_flag = nullptr;
    int* h_one = nullptr;
    REQUIRE(cudaMallocManaged(&waited, sizeof(unsigned long long)) == cudaSuccess);
    REQUIRE(cudaMalloc(&d_flag, sizeof(int)) == cudaSuccess);
    REQUIRE(cudaHostAlloc(&h_one, sizeof(int), cudaHostAllocDefault) == cudaSuccess);
    *h_one = 1;
    REQUIRE(cudaMemset(d_flag, 0, sizeof(int)) == cudaSuccess);
    REQUIRE(cudaDeviceSynchronize() == cudaSuccess);
    gpu_search_kernels::spin_until_flag<<<1, 1, 0, a>>>(d_flag, 5000000000ull, waited);
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    CHECK(cudaMemcpyAsync(d_flag, h_one, sizeof(int), cudaMemcpyHostToDevice, b) == cudaSuccess);
    CHECK(cudaStreamSynchronize(b) == cudaSuccess);
    CHECK(cudaStreamSynchronize(a) == cudaSuccess);
    const double waited_ms = double(*waited) / 1e6;
    CHECK_MSG(waited_ms >= 50 && waited_ms < 1000, "kernel waited " + std::to_string(waited_ms) + " ms");
    cudaFreeHost(h_one);
    cudaFree(d_flag);
    cudaFree(waited);
    cudaStreamDestroy(a);
    cudaStreamDestroy(b);
}

TEST_CASE(gpu_search_deadline) {
    GpuSearchOptions o;
    o.board = is_compiled(BoardKind::Quad) ? BoardKind::Quad : compiled_boards().front();
    o.eval = is_compiled(EvalKind::Pesto) ? EvalKind::Pesto : EvalKind::Material;
    o.model = compiled_models().front();
    if (o.model == ExecModel::Warp) o.block_size = 64;
    GpuSearcher searcher(gpu_test_env::context(), o);
    TaskSpec t;
    t.board = to_board<QuadBoard>(testutil::parse_or_die(testutil::test_fens()[1]));
    t.depth = 20;
    const auto start = std::chrono::steady_clock::now();
    searcher.set_deadline(start + std::chrono::milliseconds(300));
    const std::vector<SearchResult> r = searcher.run({t}, {}, 0, {});
    const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    REQUIRE(r.size() == 1);
    CHECK(r[0].aborted);
    CHECK_MSG(seconds > 0.25 && seconds < 3.0, "deadline stop after " + std::to_string(seconds) + " s");
    searcher.set_deadline(std::chrono::steady_clock::time_point::max());
    t.depth = 2;
    CHECK(!searcher.run({t}, {}, 0, {})[0].aborted);
}

TEST_CASE(gpu_search_rejects_invalid_variants) {
    GpuSearchOptions o;
    o.model = ExecModel::Thread;
    o.ordering = OrderingKind::List;
    o.stack = StackKind::Shared;
    bool threw = false;
    try {
        validate(o);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    CHECK(threw);
    o = GpuSearchOptions{};
    o.block_size = 100;
    threw = false;
    try {
        validate(o);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    CHECK(threw);
    for (const EvalKind e : {EvalKind::Material, EvalKind::Pesto, EvalKind::Terms, EvalKind::Nnue}) {
        if (is_compiled(e)) continue;
        o = GpuSearchOptions{};
        o.eval = e;
        threw = false;
        try {
            validate(o);
        } catch (const std::invalid_argument&) {
            threw = true;
        }
        CHECK(threw);
    }
}

TEST_CASE(gpu_ybwc_device_magic_attacks) {
    // The GPU-resident YBWC with magic bitboards in its kernels and explicit
    // nodes: the sequential search's score and best move, in every compiled
    // variant (a few positions and depths), as with hyperbola quintessence.
    if (!is_compiled_for_search(AttacksKind::Magic)) return;
    const bool quick = gpu_test_env::quick();
    const size_t fens = quick ? 2 : 5;
    const int max_depth = quick ? 3 : 4;
    for (const BoardKind board : compiled_boards()) {
        for (const EvalKind eval : {EvalKind::Pesto, EvalKind::Terms}) {
            if (!is_compiled(eval) || (quick && board != compiled_boards().front())) continue;
            for (GpuSearchOptions o : compiled_variants(board, eval)) {
                o.attacks = AttacksKind::Magic;
                GpuYbwcDevice device(gpu_test_env::context(), o);
                for (size_t f = 0; f < fens; ++f) {
                    const Position p = testutil::parse_or_die(testutil::test_fens()[f]);
                    std::vector<Move> pv;
                    for (int depth = 1; depth <= max_depth; ++depth) {
                        const FixedDepthResult seq = cpu_search_fixed(p, GameHistory{}, depth, -kScoreInfinite,
                                                                      kScoreInfinite, {BoardKind::Quad, eval}, pv);
                        for (const YbwcOptions& y : {YbwcOptions{1, 8, false, 0}, YbwcOptions{2, 8, true, 1100}}) {
                            YbwcDeviceStats s;
                            const FixedDepthResult r = device.search(p, GameHistory{}, depth, y, pv, &s);
                            const std::string where = describe(o) + " " + testutil::test_fens()[f] + " depth " +
                                                      std::to_string(depth) + " leaf " + std::to_string(y.leaf_depth);
                            CHECK_MSG(!r.search.aborted && r.search.score == seq.search.score &&
                                          r.search.best == seq.search.best,
                                      where + ": " + std::to_string(r.search.score) + " " + move_to_uci(r.search.best) +
                                          " vs " + std::to_string(seq.search.score) + " " +
                                          move_to_uci(seq.search.best));
                        }
                        pv = seq.pv;
                    }
                }
            }
        }
    }
}

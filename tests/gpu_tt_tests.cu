// The transposition table on the GPU (core/tt.h, gpu/device_tt.h):
// - one task on one worker runs exactly the CPU's table operations, so with
//   equal tables (same size, same generation) the results match the CPU
//   search exactly: score, best move and node count, in every variant;
// - batches of concurrent tasks sharing a table: tt-order keeps every score
//   exact, and cutoffs keep scores exact to depth 3 (see tests/test_tt.cpp);
// - the GPU-resident YBWC uses the table at its explicit nodes too, with the
//   same guarantees.

#include <string>
#include <vector>

#include "gpu/device_tt.h"
#include "gpu/search_gpu.h"
#include "gpu/variants.h"
#include "gpu/ybwc_device.h"
#include "gpu_search_util.h"
#include "gpu_test_env.h"
#include "host/fen.h"
#include "host/search_cpu.h"
#include "host/split_search.h"
#include "host/task_runner.h"
#include "host/tt_storage.h"
#include "host/uci_move.h"
#include "host/ybwc.h"
#include "test_harness.h"
#include "test_util.h"

using namespace cb;
using gpu_search_util::compiled_variants;
using gpu_search_util::quick_skip;
using gpu_search_util::score_of_move;

namespace {

struct TTCombo {
    FeatureKind features;
    TTLayoutKind layout;
};

// Compiled feature sets with a table, each with every compiled layout.
std::vector<TTCombo> compiled_tt_combos() {
    std::vector<TTCombo> v;
    for (const FeatureKind f : {FeatureKind::TT, FeatureKind::TTNoQS, FeatureKind::TTOrder})
        for (const TTLayoutKind l : {TTLayoutKind::Packed64, TTLayoutKind::Wide128})
            if (is_compiled(f, l)) v.push_back({f, l});
    return v;
}

std::string name(const TTCombo& c) { return std::string(name_of(c.features)) + " " + name_of(c.layout); }

GpuSearchOptions with(GpuSearchOptions o, const TTCombo& c) {
    o.features = c.features;
    o.tt_layout = c.layout;
    return o;
}

CpuSearchOptions cpu_with(const GpuSearchOptions& o, const TTCombo& c, const TTDesc& tt) {
    CpuSearchOptions cpu{BoardKind::Quad, o.eval, OrderingKind::Staged};
    cpu.features = c.features;
    cpu.tt_layout = c.layout;
    cpu.tt = tt;
    return cpu;
}

TaskSpec root_task(const std::string& fen, int depth) {
    TaskSpec t;
    t.board = to_board<QuadBoard>(testutil::parse_or_die(fen));
    t.depth = depth;
    return t;
}

constexpr size_t kTableMb = 16;  // same size on both sides: same buckets

}  // namespace

TEST_CASE(gpu_tt_single_tasks_match_cpu) {
    // Tasks one at a time (one worker each), tables shared across the tasks
    // and compared after all of them: the GPU and the CPU must have done the
    // same probes and stores in the same order.
    const bool quick = gpu_test_env::quick();
    const size_t fens = quick ? 3 : testutil::test_fens().size();
    const int depth = quick ? 2 : 3;
    DeviceTT dtt(kTableMb);
    HostTT htt(kTableMb);
    int checked = 0;
    for (const BoardKind board : compiled_boards()) {
        for (const EvalKind eval : {EvalKind::Pesto, EvalKind::Material}) {
            if (!is_compiled(eval) || (quick && eval != EvalKind::Pesto)) continue;
            const std::vector<GpuSearchOptions> variants = compiled_variants(board, eval);
            const std::vector<TTCombo> combos = compiled_tt_combos();
            for (size_t v = 0; v < variants.size(); ++v) {
                for (size_t k = 0; k < combos.size(); ++k) {
                    if (quick_skip(quick, v, k)) continue;
                    const TTCombo& c = combos[k];
                    const GpuSearchOptions o = with(variants[v], c);
                    GpuSearcher gpu(gpu_test_env::context(), o);
                    dtt.clear();
                    htt.clear();
                    gpu.set_tt(dtt.desc());
                    CpuTaskRunner cpu(cpu_with(o, c, htt.desc()));
                    for (size_t f = 0; f < fens; ++f) {
                        for (int d = 1; d <= depth; ++d) {
                            const std::vector<TaskSpec> task{root_task(testutil::test_fens()[f], d)};
                            const SearchResult g = gpu.run(task, {}, 0, {})[0];
                            const SearchResult r = cpu.run(task, {}, 0, {})[0];
                            const std::string where = describe(o) + " " + testutil::test_fens()[f] + " depth " +
                                                      std::to_string(d);
                            CHECK_MSG(g.score == r.score && g.best == r.best && g.nodes == r.nodes,
                                      where + ": " + std::to_string(g.score) + " " + move_to_uci(g.best) + " " +
                                          std::to_string(g.nodes) + " vs " + std::to_string(r.score) + " " +
                                          move_to_uci(r.best) + " " + std::to_string(r.nodes));
                            ++checked;
                        }
                    }
                }
            }
        }
    }
    CHECK(checked > 0);
}

TEST_CASE(gpu_tt_batches_keep_scores) {
    // Every test position at once, sharing one table, in increasing depth
    // (so no entry is deeper than a later search needs): tt-order at every
    // depth, cutoffs to depth 3, all exactly the CPU's score without a table.
    const bool quick = gpu_test_env::quick();
    const int max_depth = quick ? 2 : 4;
    std::vector<std::vector<int>> expected(size_t(max_depth) + 1);
    for (int d = 1; d <= max_depth; ++d)
        for (const std::string& fen : testutil::test_fens())
            expected[size_t(d)].push_back(cpu_search_fixed(testutil::parse_or_die(fen), GameHistory{}, d,
                                                           -kScoreInfinite, kScoreInfinite,
                                                           {BoardKind::Quad, EvalKind::Pesto, OrderingKind::Staged})
                                              .search.score);
    DeviceTT dtt(kTableMb);
    const BoardKind board = is_compiled(BoardKind::Quad) ? BoardKind::Quad : compiled_boards().front();
    const std::vector<GpuSearchOptions> variants = compiled_variants(board, EvalKind::Pesto);
    const std::vector<TTCombo> combos = compiled_tt_combos();
    for (size_t v = 0; v < variants.size(); ++v) {
        for (size_t k = 0; k < combos.size(); ++k) {
            if (quick_skip(quick, v, k)) continue;
            const TTCombo& c = combos[k];
            const GpuSearchOptions o = with(variants[v], c);
            GpuSearcher gpu(gpu_test_env::context(), o);
            dtt.clear();
            gpu.set_tt(dtt.desc());
            const int top = c.features == FeatureKind::TTOrder ? max_depth : std::min(max_depth, 3);
            for (int d = 1; d <= top; ++d) {
                std::vector<TaskSpec> tasks;
                for (const std::string& fen : testutil::test_fens()) tasks.push_back(root_task(fen, d));
                for (int pass = 0; pass < 2; ++pass) {  // the second pass finds the first's entries
                    const std::vector<SearchResult> got = gpu.run(tasks, {}, 0, {});
                    for (size_t i = 0; i < tasks.size(); ++i)
                        CHECK_MSG(got[i].score == expected[size_t(d)][i],
                                  describe(o) + " " + testutil::test_fens()[i] + " depth " + std::to_string(d) +
                                      " pass " + std::to_string(pass) + ": " + std::to_string(got[i].score) +
                                      " vs " + std::to_string(expected[size_t(d)][i]));
                }
            }
        }
    }
}

TEST_CASE(gpu_tt_ybwc_device) {
    // The GPU-resident YBWC with a table at its explicit nodes and in its
    // tasks, iterative deepening on one table: exact scores with tt-order
    // at every depth and with cutoffs to depth 3; the best move scores the same.
    const bool quick = gpu_test_env::quick();
    const size_t fens = quick ? 2 : testutil::test_fens().size();
    const int max_depth = quick ? 3 : 5;
    const YbwcOptions options[] = {{0, 8, true, 0}, {1, 8, false, 0}, {2, 8, true, 2048}};
    DeviceTT dtt(kTableMb);
    std::vector<GpuSearchOptions> variants;
    for (const BoardKind board : compiled_boards())
        for (const GpuSearchOptions& o : compiled_variants(board, EvalKind::Pesto))
            if (!quick || board == compiled_boards().front()) variants.push_back(o);
    const std::vector<TTCombo> combos = compiled_tt_combos();
    for (size_t v = 0; v < variants.size(); ++v) {
        const GpuSearchOptions& base = variants[v];
        for (size_t k = 0; k < combos.size(); ++k) {
            const TTCombo& c = combos[k];
            if (quick_skip(quick, v, k)) continue;
            if (c.features == FeatureKind::TTNoQS && base.model == ExecModel::Warp) continue;  // time
            const GpuSearchOptions o = with(base, c);
            GpuYbwcDevice device(gpu_test_env::context(), o);
            for (size_t f = 0; f < fens; ++f) {
                const Position p = testutil::parse_or_die(testutil::test_fens()[f]);
                for (const YbwcOptions& y : options) {
                    dtt.clear();
                    dtt.new_search();
                    device.set_tt(dtt.desc());
                    std::vector<Move> pv;
                    const int top = c.features == FeatureKind::TTOrder ? max_depth : 3;
                    for (int d = 1; d <= top; ++d) {
                        const FixedDepthResult seq = cpu_search_fixed(p, GameHistory{}, d, -kScoreInfinite,
                                                                      kScoreInfinite, {BoardKind::Quad, o.eval}, pv);
                        YbwcDeviceStats s;
                        const FixedDepthResult r = device.search(p, GameHistory{}, d, y, pv, &s);
                        const std::string where = describe(o) + " " + testutil::test_fens()[f] + " depth " +
                                                  std::to_string(d) + " leaf " + std::to_string(y.leaf_depth);
                        CHECK_MSG(!r.search.aborted && !s.overflow, where + ": aborted");
                        CHECK_MSG(r.search.score == seq.search.score, where + ": score " +
                                                                          std::to_string(r.search.score) + " vs " +
                                                                          std::to_string(seq.search.score));
                        if (r.search.best != seq.search.best)
                            CHECK_MSG(!r.search.best.is_none() &&
                                          score_of_move(p, r.search.best, d, o.eval) == seq.search.score,
                                      where + ": best " + move_to_uci(r.search.best) + " doesn't score " +
                                          std::to_string(seq.search.score));
                        pv = seq.pv;
                    }
                }
            }
        }
    }
}

TEST_CASE(gpu_tt_host_coordinated_searches) {
    // Independent subtrees and YBWC with CPU rounds hand their tasks to a
    // GPU runner that shares one table: exact to depth 3.
    const size_t fens = gpu_test_env::quick() ? 2 : testutil::test_fens().size();
    DeviceTT dtt(kTableMb);
    GpuSearchOptions o;
    o.board = is_compiled(BoardKind::Quad) ? BoardKind::Quad : compiled_boards().front();
    o.model = compiled_models().front();
    if (o.model == ExecModel::Warp) o.block_size = 64;
    for (const TTCombo& c : compiled_tt_combos()) {
        if (c.features != FeatureKind::TT) continue;
        GpuSearcher gpu(gpu_test_env::context(), with(o, c));
        for (size_t f = 0; f < fens; ++f) {
            const Position p = testutil::parse_or_die(testutil::test_fens()[f]);
            dtt.clear();
            gpu.set_tt(dtt.desc());
            for (int d = 1; d <= 3; ++d) {
                const int seq = cpu_search_fixed(p, GameHistory{}, d, -kScoreInfinite, kScoreInfinite,
                                                 {BoardKind::Quad, o.eval})
                                    .search.score;
                const FixedDepthResult ind = search_independent(gpu, p, GameHistory{}, d, std::min(d, 2), {});
                CHECK_MSG(ind.search.score == seq, name(c) + " independent " + testutil::test_fens()[f] + " depth " +
                                                       std::to_string(d));
                const FixedDepthResult yb = search_ybwc(gpu, p, GameHistory{}, d, YbwcOptions{1, 8}, {});
                CHECK_MSG(yb.search.score == seq,
                          name(c) + " ybwc " + testutil::test_fens()[f] + " depth " + std::to_string(d));
            }
        }
    }
}

TEST_CASE(gpu_tt_prunes) {
    // Deeper, where the table pays: fewer nodes for the same depth.
    if (gpu_test_env::quick()) return;
    GpuSearchOptions o;
    o.board = is_compiled(BoardKind::Quad) ? BoardKind::Quad : compiled_boards().front();
    o.model = compiled_models().front();
    if (o.model == ExecModel::Warp) o.block_size = 64;
    DeviceTT dtt(256);
    GpuYbwcDevice plain(gpu_test_env::context(), o);
    for (const TTCombo& c : compiled_tt_combos()) {
        if (c.features == FeatureKind::TTOrder) continue;
        GpuYbwcDevice device(gpu_test_env::context(), with(o, c));
        u64 with_table = 0, without = 0;
        for (const std::string& fen : testutil::test_fens()) {
            const Position p = testutil::parse_or_die(fen);
            dtt.clear();
            dtt.new_search();
            device.set_tt(dtt.desc());
            std::vector<Move> pv, pv_plain;
            for (int d = 1; d <= 6; ++d) {
                const FixedDepthResult r = device.search(p, GameHistory{}, d, YbwcOptions{}, pv);
                const FixedDepthResult b = plain.search(p, GameHistory{}, d, YbwcOptions{}, pv_plain);
                with_table += r.search.nodes;
                without += b.search.nodes;
                CHECK(!r.search.aborted && !r.search.best.is_none() == !b.search.best.is_none());
                pv = r.pv;
                pv_plain = b.pv;
            }
        }
        CHECK_MSG(with_table < without, name(c) + ": " + std::to_string(with_table) + " nodes vs " +
                                            std::to_string(without));
    }
}

TEST_CASE(gpu_tt_explicit_nodes_use_the_table) {
    // Explicit nodes store their results and use them like search_task's
    // nodes: the same search again (no PV) is answered by the root and its
    // children alone, but a search one ply deeper isn't (stored depths fall
    // short), and its result stays exact.
    DeviceTT dtt(kTableMb);
    GpuSearchOptions base;
    base.board = is_compiled(BoardKind::Quad) ? BoardKind::Quad : compiled_boards().front();
    base.model = compiled_models().front();
    if (base.model == ExecModel::Warp) base.block_size = 64;
    for (const TTCombo& c : compiled_tt_combos()) {
        if (c.features == FeatureKind::TTOrder) continue;
        GpuYbwcDevice device(gpu_test_env::context(), with(base, c));
        for (size_t f = 0; f < (gpu_test_env::quick() ? 2 : 8); ++f) {
            const Position p = testutil::parse_or_die(testutil::test_fens()[f]);
            const int moves = count_legal(to_board<QuadBoard>(p), gpu_search_util::host_attacks());
            if (moves == 0) continue;
            dtt.clear();
            device.set_tt(dtt.desc());
            const YbwcOptions explicit_all{0, 8, true, 0};  // every node of depth 1 or more is explicit
            const FixedDepthResult first = device.search(p, GameHistory{}, 2, explicit_all, {});
            YbwcDeviceStats s;
            const FixedDepthResult again = device.search(p, GameHistory{}, 2, explicit_all, {}, &s);
            const std::string where = name(c) + " " + testutil::test_fens()[f];
            CHECK_MSG(again.search.score == first.search.score && !again.search.best.is_none(), where);
            CHECK_MSG(s.tasks == 0 && s.explicit_nodes == unsigned(moves + 1),
                      where + ": " + describe(s));
            const FixedDepthResult deeper = device.search(p, GameHistory{}, 3, explicit_all, {}, &s);
            CHECK_MSG(s.tasks > 0, where + ": depth 3 answered from depth-2 entries: " + describe(s));
            const int seq = cpu_search_fixed(p, GameHistory{}, 3, -kScoreInfinite, kScoreInfinite, {}).search.score;
            CHECK_MSG(deeper.search.score == seq, where + ": depth 3 score");
        }
    }
}

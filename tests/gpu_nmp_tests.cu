// Null-move pruning on the GPU (feature set tt-nmp; see tests/test_nmp.cpp
// for the CPU side):
// - one task on one worker runs exactly the CPU's search, null moves
//   included: same score, best move and node count in every variant, for
//   root tasks and for tasks below the root (whose flags say whether their
//   root may pass);
// - the GPU-resident YBWC passes at explicit nodes too (the null move as a
//   child of its own, searched as a task or as an explicit node), keeps
//   mates exact, never passes with only kings and pawns, and prunes;
// - the host-coordinated searches run their tasks with null moves.
//
// Of 10 bugs planted one at a time in the explicit nodes' null moves, these
// tests catch 7. Not caught: the null search with the node's own window
// and the moves after a failed pass with alpha = beta - 1 (both identical to
// the right code under the zero windows that make the GPU deterministic),
// and bypassing null_move_score (tested on the CPU) at the call site.

#include <algorithm>
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

namespace {

constexpr size_t kTableMb = 16;  // same size on both sides: same buckets

std::vector<TTLayoutKind> nmp_layouts() {
    std::vector<TTLayoutKind> v;
    for (const TTLayoutKind l : {TTLayoutKind::Packed64, TTLayoutKind::Wide128})
        if (is_compiled(FeatureKind::TTNmp, l)) v.push_back(l);
    return v;
}

GpuSearchOptions with_nmp(GpuSearchOptions o, TTLayoutKind layout) {
    o.features = FeatureKind::TTNmp;
    o.tt_layout = layout;
    return o;
}

GpuSearchOptions default_variant() {
    GpuSearchOptions o;
    o.board = is_compiled(BoardKind::Quad) ? BoardKind::Quad : compiled_boards().front();
    o.model = compiled_models().front();
    if (o.model == ExecModel::Warp) o.block_size = 64;
    return o;
}

// Positions where null moves happen, then the usual test positions.
std::vector<std::string> positions() {
    std::vector<std::string> v = {
        "r1bq1rk1/pp2bppp/2n1pn2/3p4/2PP4/2N1PN2/PP3PPP/R2QKB1R w KQ - 0 8",
        "2r2rk1/1bqnbppp/pp1ppn2/8/2PNP3/1PN1BP2/P2Q2PP/2RR1BK1 w - - 0 15",
        "r2q1rk1/1p1nbppp/p2pbn2/4p3/4P3/1NN1BP2/PPPQ2PP/2KR1B1R w - - 0 11",
    };
    for (const std::string& fen : testutil::test_fens()) v.push_back(fen);
    return v;
}

TaskSpec task_for(const Position& p, int depth, int alpha, int beta, int ply, u8 flags) {
    TaskSpec t;
    t.board = to_board<QuadBoard>(p);
    t.depth = depth;
    t.alpha = alpha;
    t.beta = beta;
    t.ply = ply;
    t.flags = flags;
    return t;
}

bool is_legal_move(const Position& p, Move m) {
    MoveList list;
    generate_legal(to_board<QuadBoard>(p), gpu_search_util::host_attacks(), list);
    for (int i = 0; i < list.size; ++i)
        if (list.moves[i] == m) return true;
    return false;
}

}  // namespace

TEST_CASE(gpu_nmp_single_tasks_match_cpu) {
    // Tasks one at a time (one worker each), tables shared across the tasks
    // on both sides, as gpu_tt_single_tasks_match_cpu.
    const bool quick = gpu_test_env::quick();
    const std::vector<std::string> fens = positions();
    const size_t count = quick ? 3 : fens.size();
    const int depth = quick ? 3 : 4;
    DeviceTT dtt(kTableMb);
    HostTT htt(kTableMb);
    int checked = 0;
    for (const BoardKind board : compiled_boards()) {
        for (const EvalKind eval : {EvalKind::Pesto, EvalKind::Material}) {
            if (!is_compiled(eval) || (quick && eval != EvalKind::Pesto)) continue;
            const std::vector<GpuSearchOptions> variants = compiled_variants(board, eval);
            const std::vector<TTLayoutKind> layouts = nmp_layouts();
            for (size_t v = 0; v < variants.size(); ++v) {
                for (size_t k = 0; k < layouts.size(); ++k) {
                    if (quick_skip(quick, v, k)) continue;
                    const TTLayoutKind layout = layouts[k];
                    const GpuSearchOptions o = with_nmp(variants[v], layout);
                    GpuSearcher gpu(gpu_test_env::context(), o);
                    dtt.clear();
                    htt.clear();
                    gpu.set_tt(dtt.desc());
                    CpuSearchOptions c{BoardKind::Quad, eval, OrderingKind::Staged};
                    c.features = FeatureKind::TTNmp;
                    c.tt_layout = layout;
                    c.tt = htt.desc();
                    CpuTaskRunner cpu(c);
                    for (size_t f = 0; f < count; ++f) {
                        const Position p = testutil::parse_or_die(fens[f]);
                        std::vector<TaskSpec> tasks;
                        for (int d = 1; d <= depth; ++d)
                            tasks.push_back(task_for(p, d, -kScoreInfinite, kScoreInfinite, 0, 0));
                        // Below the root, with a window the static evaluation
                        // beats: the task root passes unless its flags forbid it.
                        if (!make_check_info(to_board<QuadBoard>(p), gpu_search_util::host_attacks()).in_check()) {
                            const QuadBoard b = to_board<QuadBoard>(p);
                            const int beta =
                                (eval == EvalKind::Pesto
                                     ? PestoEval::evaluate(compute_accumulator<PestoEval>(b, host_search_tables().eval),
                                                           state::side(b.state()))
                                     : MaterialEval::evaluate(
                                           compute_accumulator<MaterialEval>(b, host_search_tables().eval),
                                           state::side(b.state()))) -
                                50;
                            for (const u8 flags : {u8(0), kTaskAfterNull, kTaskOnPv})
                                tasks.push_back(task_for(p, depth, beta - 1, beta, 1, flags));
                        }
                        for (const TaskSpec& t : tasks) {
                            const SearchResult g = gpu.run({t}, {}, 0, {})[0];
                            const SearchResult r = cpu.run({t}, {}, 0, {})[0];
                            const std::string where = describe(o) + " " + fens[f] + " depth " +
                                                      std::to_string(t.depth) + " ply " + std::to_string(t.ply) +
                                                      " flags " + std::to_string(t.flags);
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

TEST_CASE(gpu_nmp_ybwc_device) {
    // Iterative deepening on one table, with several YBWC settings (leaf
    // depth 0: every node of depth 1 or more explicit, so null moves are
    // explicit nodes too). Every search finishes with a legal move, and
    // explicit nodes pass and cut off; without the feature they never pass.
    const bool quick = gpu_test_env::quick();
    const std::vector<std::string> fens = positions();
    const size_t count = quick ? 2 : 8;
    struct Setting {
        YbwcOptions y;
        int max_depth;
    };
    const Setting settings[] = {{{2, 8, true, 2048}, quick ? 4 : 6}, {{0, 8, true, 0}, quick ? 3 : 5},
                                {{1, 8, false, 0}, quick ? 3 : 5}};
    DeviceTT dtt(kTableMb);
    std::vector<GpuSearchOptions> variants;
    for (const BoardKind board : compiled_boards())
        for (const GpuSearchOptions& o : compiled_variants(board, EvalKind::Pesto))
            if (!quick || board == compiled_boards().front()) variants.push_back(o);
    const std::vector<TTLayoutKind> layouts = nmp_layouts();
    for (size_t v = 0; v < variants.size(); ++v) {
        for (size_t k = 0; k < layouts.size(); ++k) {
            if (quick_skip(quick, v, k)) continue;
            const GpuSearchOptions o = with_nmp(variants[v], layouts[k]);
            GpuYbwcDevice device(gpu_test_env::context(), o);
            unsigned tries = 0, cuts = 0;
            for (size_t f = 0; f < count; ++f) {
                const Position p = testutil::parse_or_die(fens[f]);
                const bool has_moves = count_legal(to_board<QuadBoard>(p), gpu_search_util::host_attacks()) > 0;
                for (const Setting& s : settings) {
                    dtt.clear();
                    device.set_tt(dtt.desc());
                    std::vector<Move> pv;
                    for (int d = 1; d <= s.max_depth; ++d) {
                        dtt.new_search();
                        YbwcDeviceStats stats;
                        const FixedDepthResult r = device.search(p, GameHistory{}, d, s.y, pv, &stats);
                        const std::string where = describe(o) + " " + fens[f] + " depth " + std::to_string(d) +
                                                  " leaf " + std::to_string(s.y.leaf_depth);
                        CHECK_MSG(!r.search.aborted && !stats.overflow, where + ": aborted");
                        CHECK_MSG(has_moves ? is_legal_move(p, r.search.best) : r.search.best.is_none(),
                                  where + ": best " + move_to_uci(r.search.best));
                        CHECK_MSG(stats.null_cuts <= stats.null_tries, where + ": " + describe(stats));
                        tries += stats.null_tries;
                        cuts += stats.null_cuts;
                        pv = r.pv;
                    }
                }
            }
            CHECK_MSG(tries > 0 && cuts > 0, describe(o) + ": " + std::to_string(tries) + " null moves at explicit "
                                                 "nodes, " + std::to_string(cuts) + " cutoffs");
        }
    }
    // The table alone never passes.
    GpuSearchOptions o = default_variant();
    o.features = FeatureKind::TT;
    if (is_compiled(o.features, o.tt_layout)) {
        GpuYbwcDevice device(gpu_test_env::context(), o);
        dtt.clear();
        device.set_tt(dtt.desc());
        YbwcDeviceStats stats;
        device.search(testutil::parse_or_die(fens[0]), GameHistory{}, quick ? 4 : 6, YbwcOptions{}, {}, &stats);
        CHECK_MSG(stats.null_tries == 0 && stats.null_cuts == 0, describe(stats));
    }
}

TEST_CASE(gpu_nmp_zero_window_matches_cpu) {
    // With a zero window, every node at a given ply has the same window, in
    // whatever order results arrive: the explicit nodes of the GPU-resident
    // YBWC then pass exactly where the CPU search does, with the same null
    // searches, so each search's outcome (above beta or below alpha) is the
    // CPU's, to depth 3 (where the table is exact, see gpu_tt_ybwc_device).
    // Leaf depth 0 makes every node of depth 1 or more explicit; a tiny node
    // budget turns quiescence tasks, null moves included, into explicit
    // nodes too. Each window is searched without a PV and with the CPU's
    // (nodes on it never pass), from positions that include checks at the
    // root's children (no node in check may pass).
    const bool quick = gpu_test_env::quick();
    std::vector<std::string> fens = {
        "3qk3/8/8/8/8/8/8/R2QK2R b KQ - 0 1",  // black checks a much stronger white
        "4k3/8/8/8/8/5q2/8/R3K2R b KQ - 0 1",
        // Quiescence trees past 1024 nodes (where the node budget is checked), so the budget splits them.
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
    };
    for (const std::string& fen : positions()) fens.push_back(fen);
    const size_t count = quick ? 3 : fens.size();
    const YbwcOptions all_settings[] = {{0, 8, true, 0}, {0, 8, true, 16}, {1, 8, false, 0}, {2, 8, true, 2048}};
    const std::vector<YbwcOptions> settings(all_settings, all_settings + (quick ? 2 : 4));
    DeviceTT dtt(kTableMb);
    HostTT htt(kTableMb);
    unsigned tries = 0, explicit_null_children = 0;
    const std::vector<TTLayoutKind> layouts = nmp_layouts();
    for (size_t k = 0; k < layouts.size(); ++k) {
        if (quick && k > 0) break;
        GpuYbwcDevice device(gpu_test_env::context(), with_nmp(default_variant(), layouts[k]));
        CpuSearchOptions c{BoardKind::Quad, EvalKind::Pesto, OrderingKind::Staged};
        c.features = FeatureKind::TTNmp;
        c.tt_layout = layouts[k];
        c.tt = htt.desc();
        for (size_t f = 0; f < count; ++f) {
            const Position p = testutil::parse_or_die(fens[f]);
            for (int d = 1; d <= 3; ++d) {
                htt.clear();
                const FixedDepthResult full = cpu_search_fixed(p, GameHistory{}, d, -kScoreInfinite, kScoreInfinite, c);
                const int s = full.search.score;
                // Zero windows on both sides of the score and 30 cp away from it.
                for (const int alpha : {s - 31, s - 30, s - 1, s, s + 29, s + 30}) {
                    const int beta = alpha + 1;
                    if (alpha <= -kScoreInfinite || beta >= kScoreInfinite) continue;
                    for (const std::vector<Move>& pv : {std::vector<Move>{}, full.pv}) {
                        htt.clear();
                        const int expected = cpu_search_fixed(p, GameHistory{}, d, alpha, beta, c, pv).search.score;
                        for (const YbwcOptions& y : settings) {
                            dtt.clear();
                            device.set_tt(dtt.desc());
                            YbwcDeviceStats st;
                            const FixedDepthResult r = device.search(p, GameHistory{}, d, y, pv, &st, alpha, beta);
                            const std::string where = fens[f] + " " + name_of(layouts[k]) + " depth " +
                                                      std::to_string(d) + " window " + std::to_string(alpha) + " " +
                                                      std::to_string(beta) + (pv.empty() ? "" : " pv") + " leaf " +
                                                      std::to_string(y.leaf_depth) + " budget " +
                                                      std::to_string(y.node_budget);
                            CHECK_MSG(!r.search.aborted && (r.search.score >= beta) == (expected >= beta),
                                      where + ": " + std::to_string(r.search.score) + " vs " +
                                          std::to_string(expected) + " (" + describe(st) + ")");
                            tries += st.null_tries;
                            if (y.node_budget == 16) explicit_null_children += st.over_budget;
                        }
                    }
                }
            }
        }
    }
    CHECK_MSG(tries > 0 && explicit_null_children > 0,
              std::to_string(tries) + " null moves at explicit nodes, " + std::to_string(explicit_null_children) +
                  " tasks split by the budget");
}

TEST_CASE(gpu_nmp_kings_and_pawns_never_pass) {
    // No side may pass: to depth 3 the GPU-resident YBWC with tt-nmp scores
    // exactly as the CPU search without a table (as with tt alone, see
    // gpu_tt_ybwc_device), and no explicit node passes at any depth.
    const char* const fens[] = {
        "8/8/4k3/8/2P5/4K3/8/8 w - - 0 1",
        "8/5k2/3p4/1p1P4/1P6/5K2/8/8 w - - 0 1",
        "8/pp3k2/8/2P5/5K2/1P6/8/8 b - - 0 1",
    };
    DeviceTT dtt(kTableMb);
    for (const TTLayoutKind layout : nmp_layouts()) {
        GpuYbwcDevice device(gpu_test_env::context(), with_nmp(default_variant(), layout));
        for (const char* fen : fens) {
            const Position p = testutil::parse_or_die(fen);
            for (const YbwcOptions& y : {YbwcOptions{0, 8, true, 0}, YbwcOptions{}}) {
                dtt.clear();
                device.set_tt(dtt.desc());
                std::vector<Move> pv;
                for (int d = 1; d <= (gpu_test_env::quick() ? 4 : 7); ++d) {
                    dtt.new_search();
                    YbwcDeviceStats s;
                    const FixedDepthResult r = device.search(p, GameHistory{}, d, y, pv, &s);
                    const std::string where = std::string(fen) + " " + name_of(layout) + " depth " +
                                              std::to_string(d) + " leaf " + std::to_string(y.leaf_depth);
                    CHECK_MSG(s.null_tries == 0, where + ": " + describe(s));
                    if (d <= 3) {
                        const int seq = cpu_search_fixed(p, GameHistory{}, d, -kScoreInfinite, kScoreInfinite, {})
                                            .search.score;
                        CHECK_MSG(r.search.score == seq,
                                  where + ": " + std::to_string(r.search.score) + " vs " + std::to_string(seq));
                    }
                    pv = r.pv;
                }
            }
        }
    }
}

TEST_CASE(gpu_nmp_mates_keep_their_distance) {
    // As nmp_mates_keep_their_distance, on the GPU-resident YBWC.
    struct Case {
        const char* fen;
        int depth;
    };
    const Case cases[] = {
        {"6k1/5ppp/8/8/8/8/8/R5K1 w - - 0 1", 3},
        {"k7/8/1K6/8/8/8/8/7R b - - 0 1", 4},
        {"7k/8/6K1/8/8/8/8/6Q1 w - - 0 1", 5},
        {"r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5Q2/PPPP1PPP/RNB1K1NR w KQkq - 4 4", 3},
    };
    DeviceTT dtt(kTableMb);
    for (const TTLayoutKind layout : nmp_layouts()) {
        GpuYbwcDevice device(gpu_test_env::context(), with_nmp(default_variant(), layout));
        for (const Case& c : cases) {
            const Position p = testutil::parse_or_die(c.fen);
            const int mate = cpu_search_fixed(p, GameHistory{}, c.depth, -kScoreInfinite, kScoreInfinite, {})
                                 .search.score;
            REQUIRE(is_mate_score(mate));
            for (const YbwcOptions& y : {YbwcOptions{0, 8, true, 0}, YbwcOptions{}}) {
                dtt.clear();
                device.set_tt(dtt.desc());
                std::vector<Move> pv;
                int score = 0;
                for (int d = 1; d <= c.depth + 3; ++d) {
                    dtt.new_search();
                    const FixedDepthResult r = device.search(p, GameHistory{}, d, y, pv);
                    score = r.search.score;
                    pv = r.pv;
                }
                CHECK_MSG(score == mate, std::string(c.fen) + " " + name_of(layout) + " leaf " +
                                             std::to_string(y.leaf_depth) + ": " + std::to_string(score) + " vs " +
                                             std::to_string(mate));
            }
        }
    }
}

TEST_CASE(gpu_nmp_prunes) {
    // Fewer nodes to the same depth than the table alone.
    if (gpu_test_env::quick()) return;
    GpuSearchOptions base = default_variant();
    GpuSearchOptions tt = base;
    tt.features = FeatureKind::TT;
    DeviceTT dtt(256);
    for (const TTLayoutKind layout : nmp_layouts()) {
        tt.tt_layout = layout;
        if (!is_compiled(tt.features, layout)) continue;
        GpuYbwcDevice with(gpu_test_env::context(), with_nmp(base, layout));
        GpuYbwcDevice without(gpu_test_env::context(), tt);
        u64 nodes_with = 0, nodes_without = 0;
        for (const std::string& fen : positions()) {
            const Position p = testutil::parse_or_die(fen);
            for (GpuYbwcDevice* device : {&with, &without}) {
                dtt.clear();
                device->set_tt(dtt.desc());
                std::vector<Move> pv;
                for (int d = 1; d <= 7; ++d) {
                    dtt.new_search();
                    const FixedDepthResult r = device->search(p, GameHistory{}, d, YbwcOptions{}, pv);
                    (device == &with ? nodes_with : nodes_without) += r.search.nodes;
                    pv = r.pv;
                }
            }
        }
        CHECK_MSG(nodes_with < nodes_without, std::string(name_of(layout)) + ": " + std::to_string(nodes_with) +
                                                  " nodes with null moves vs " + std::to_string(nodes_without));
    }
}

TEST_CASE(gpu_nmp_host_coordinated_searches) {
    // Independent subtrees and YBWC with CPU rounds hand their tasks to a
    // GPU runner with null moves: they finish with legal moves.
    const std::vector<std::string> fens = positions();
    const size_t count = gpu_test_env::quick() ? 2 : 6;
    DeviceTT dtt(kTableMb);
    for (const TTLayoutKind layout : nmp_layouts()) {
        GpuSearcher gpu(gpu_test_env::context(), with_nmp(default_variant(), layout));
        for (size_t f = 0; f < count; ++f) {
            const Position p = testutil::parse_or_die(fens[f]);
            const bool has_moves = count_legal(to_board<QuadBoard>(p), gpu_search_util::host_attacks()) > 0;
            dtt.clear();
            gpu.set_tt(dtt.desc());
            for (int d = 1; d <= 4; ++d) {
                const FixedDepthResult ind = search_independent(gpu, p, GameHistory{}, d, std::min(d, 2), {});
                const FixedDepthResult yb = search_ybwc(gpu, p, GameHistory{}, d, YbwcOptions{1, 8}, {});
                const std::string where = fens[f] + " " + name_of(layout) + " depth " + std::to_string(d);
                CHECK_MSG(!ind.search.aborted && (has_moves ? is_legal_move(p, ind.search.best)
                                                            : ind.search.best.is_none()),
                          where + ": independent");
                CHECK_MSG(!yb.search.aborted &&
                              (has_moves ? is_legal_move(p, yb.search.best) : yb.search.best.is_none()),
                          where + ": ybwc");
            }
        }
    }
}

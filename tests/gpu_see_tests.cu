// SEE and delta pruning in quiescence on the GPU (feature sets
// tt-killers-see and tt-ext-killers-see; see tests/test_see.cpp for the CPU
// side):
// - one task on one worker runs exactly the CPU's search: same score, best
//   move and node count in every variant, at several settings;
// - the GPU-resident YBWC's explicit quiescence nodes (tasks split for
//   running out of node budget) prune too, searches end with legal moves,
//   and with zero windows the outcomes equal the CPU's;
// - where an explicit quiescence node searches a single capture, its
//   fail-soft value (stand pat, delta bounds, the capture's result) equals
//   the CPU's exactly;
// - mates keep their distance.

#include <string>
#include <vector>

#include "core/see.h"
#include "gpu/device_tt.h"
#include "gpu/search_gpu.h"
#include "gpu/variants.h"
#include "gpu/ybwc_device.h"
#include "gpu_search_util.h"
#include "gpu_test_env.h"
#include "host/fen.h"
#include "host/host_tables.h"
#include "host/search_cpu.h"
#include "host/task_runner.h"
#include "host/tt_storage.h"
#include "host/uci_move.h"
#include "test_harness.h"
#include "test_util.h"

using namespace cb;
using gpu_search_util::compiled_variants;
using gpu_search_util::quick_skip;

namespace {

constexpr size_t kTableMb = 16;  // same size on both sides: same buckets

const FeatureKind kSeeSets[] = {FeatureKind::TTKillersSee, FeatureKind::TTExtKillersSee};

struct Combo {
    FeatureKind features;
    TTLayoutKind layout;
};
std::vector<Combo> see_combos() {
    std::vector<Combo> v;
    for (const FeatureKind f : kSeeSets)
        for (const TTLayoutKind l : {TTLayoutKind::Wide128, TTLayoutKind::Packed64})
            if (is_compiled(f, l)) v.push_back({f, l});
    return v;
}

GpuSearchOptions with_combo(GpuSearchOptions o, const Combo& c, const PruneParams& prune = {}) {
    o.features = c.features;
    o.tt_layout = c.layout;
    o.prune = prune;
    return o;
}

CpuSearchOptions cpu_combo(EvalKind eval, const Combo& c, const HostTT& table, const PruneParams& prune = {}) {
    CpuSearchOptions o{BoardKind::Quad, eval, OrderingKind::Staged};
    o.features = c.features;
    o.tt_layout = c.layout;
    o.prune = prune;
    o.tt = table.desc();
    return o;
}

std::string describe(const Combo& c) { return std::string(name_of(c.features)) + "/" + name_of(c.layout); }

GpuSearchOptions default_variant() {
    GpuSearchOptions o;
    o.board = is_compiled(BoardKind::Quad) ? BoardKind::Quad : compiled_boards().front();
    o.model = compiled_models().front();
    if (o.model == ExecModel::Warp) o.block_size = 64;
    return o;
}

std::vector<std::string> positions() {
    std::vector<std::string> v = {
        "1k1r3q/1ppn3p/p4b2/4p3/8/P2N2P1/1PP1R1BP/2K1Q3 w - - 0 1",  // a losing capture (SEE -220)
        "r1bq1rk1/pp2bppp/2n1pn2/3p4/2PP4/2N1PN2/PP3PPP/R2QKB1R w KQ - 0 8",
        "4r3/2pk4/3n1B1Q/pp6/1P4qP/2P3P1/1P4K1/R7 w - - 3 47",
    };
    for (const std::string& fen : testutil::test_fens()) v.push_back(fen);
    return v;
}

bool is_legal_move(const Position& p, Move m) {
    MoveList list;
    generate_legal(to_board<QuadBoard>(p), gpu_search_util::host_attacks(), list);
    for (int i = 0; i < list.size; ++i)
        if (list.moves[i] == m) return true;
    return false;
}

// The default settings, delta pruning alone and SEE pruning alone.
std::vector<PruneParams> settings_to_try(bool quick) {
    PruneParams delta_only;
    delta_only.see_prune = 0;
    PruneParams see_only;
    see_only.delta_margin = 0;
    if (quick) return {PruneParams{}};
    return {PruneParams{}, delta_only, see_only};
}

}  // namespace

TEST_CASE(gpu_see_single_tasks_match_cpu) {
    const bool quick = gpu_test_env::quick();
    const std::vector<std::string> fens = positions();
    const size_t count = quick ? 3 : fens.size();
    const int depth = quick ? 3 : 4;
    DeviceTT dtt(kTableMb);
    HostTT htt(kTableMb);
    int checked = 0;
    for (const PruneParams& prune : settings_to_try(quick)) {
        for (const BoardKind board : compiled_boards()) {
            for (const EvalKind eval : {EvalKind::Pesto, EvalKind::Material}) {
                if (!is_compiled(eval) || (quick && eval != EvalKind::Pesto)) continue;
                const std::vector<GpuSearchOptions> variants = compiled_variants(board, eval);
                const std::vector<Combo> combos = see_combos();
                for (size_t v = 0; v < variants.size(); ++v) {
                    for (size_t k = 0; k < combos.size(); ++k) {
                        if (quick_skip(quick, v, k)) continue;
                        const GpuSearchOptions o = with_combo(variants[v], combos[k], prune);
                        GpuSearcher gpu(gpu_test_env::context(), o);
                        dtt.clear();
                        htt.clear();
                        gpu.set_tt(dtt.desc());
                        CpuTaskRunner cpu(cpu_combo(eval, combos[k], htt, prune));
                        for (size_t f = 0; f < count; ++f) {
                            for (int d = 1; d <= depth; ++d) {
                                TaskSpec t;
                                t.board = to_board<QuadBoard>(testutil::parse_or_die(fens[f]));
                                t.depth = d;
                                const SearchResult g = gpu.run({t}, {}, 0, {})[0];
                                const SearchResult r = cpu.run({t}, {}, 0, {})[0];
                                CHECK_MSG(g.score == r.score && g.best == r.best && g.nodes == r.nodes,
                                          cb::describe(o) + " " + fens[f] + " depth " + std::to_string(d) + ": " +
                                              std::to_string(g.score) + " " + move_to_uci(g.best) + " " +
                                              std::to_string(g.nodes) + " vs " + std::to_string(r.score) + " " +
                                              move_to_uci(r.best) + " " + std::to_string(r.nodes));
                                ++checked;
                            }
                        }
                    }
                }
            }
        }
    }
    CHECK(checked > 0);
}

TEST_CASE(gpu_see_ybwc_device) {
    // Iterative deepening on one table: every search ends with a legal move.
    // A tiny node budget splits tasks down into quiescence, where explicit
    // nodes prune captures too.
    const bool quick = gpu_test_env::quick();
    const std::vector<std::string> fens = positions();
    const size_t count = quick ? 2 : 8;
    struct Setting {
        YbwcOptions y;
        int max_depth;
    };
    const Setting settings[] = {{{2, 8, true, 2048}, quick ? 5 : 7}, {{0, 8, true, 16}, quick ? 3 : 4},
                                {{1, 8, false, 0}, quick ? 4 : 5}};
    DeviceTT dtt(kTableMb);
    std::vector<GpuSearchOptions> variants;
    for (const BoardKind board : compiled_boards())
        for (const GpuSearchOptions& o : compiled_variants(board, EvalKind::Pesto))
            if (!quick || board == compiled_boards().front()) variants.push_back(o);
    const std::vector<Combo> combos = see_combos();
    unsigned pruned = 0;
    for (size_t v = 0; v < variants.size(); ++v) {
        for (size_t k = 0; k < combos.size(); ++k) {
            if (quick_skip(quick, v, k)) continue;
            const GpuSearchOptions o = with_combo(variants[v], combos[k]);
            GpuYbwcDevice device(gpu_test_env::context(), o);
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
                        const std::string where = cb::describe(o) + " " + fens[f] + " depth " + std::to_string(d) +
                                                  " leaf " + std::to_string(s.y.leaf_depth) + " budget " +
                                                  std::to_string(s.y.node_budget);
                        CHECK_MSG(!r.search.aborted && !stats.overflow, where + ": aborted (" + describe(stats) + ")");
                        CHECK_MSG(has_moves ? is_legal_move(p, r.search.best) : r.search.best.is_none(),
                                  where + ": best " + move_to_uci(r.search.best));
                        pruned += stats.qs_pruned;
                        pv = r.pv;
                    }
                }
            }
        }
    }
    // Quick mode's few shallow searches may split no task down to a
    // quiescence node with a capture to prune.
    CHECK_MSG(quick || pruned > 0, "no explicit quiescence node pruned a move");
}

TEST_CASE(gpu_see_zero_window_matches_cpu) {
    // As gpu_ext_zero_window_matches_cpu, with a node budget of 16 among the
    // settings, so explicit quiescence nodes decide too.
    const bool quick = gpu_test_env::quick();
    const std::vector<std::string> fens = positions();
    const size_t count = quick ? 3 : fens.size();
    const YbwcOptions all_settings[] = {{0, 8, true, 16}, {0, 8, true, 0}, {1, 8, false, 0}, {2, 8, true, 2048}};
    const std::vector<YbwcOptions> settings(all_settings, all_settings + (quick ? 2 : 4));
    DeviceTT dtt(kTableMb);
    HostTT htt(kTableMb);
    const std::vector<Combo> combos = see_combos();
    for (size_t k = 0; k < combos.size(); ++k) {
        if (quick && k > 0) break;
        for (const PruneParams& prune : settings_to_try(quick)) {
            GpuYbwcDevice device(gpu_test_env::context(), with_combo(default_variant(), combos[k], prune));
            const CpuSearchOptions c = cpu_combo(EvalKind::Pesto, combos[k], htt, prune);
            for (size_t f = 0; f < count; ++f) {
                const Position p = testutil::parse_or_die(fens[f]);
                for (int d = 1; d <= 3; ++d) {
                    htt.clear();
                    const FixedDepthResult full =
                        cpu_search_fixed(p, GameHistory{}, d, -kScoreInfinite, kScoreInfinite, c);
                    const int s = full.search.score;
                    for (const int alpha : {s - 31, s - 30, s - 1, s, s + 29, s + 30}) {
                        const int beta = alpha + 1;
                        if (alpha <= -kScoreInfinite || beta >= kScoreInfinite) continue;
                        for (const std::vector<Move>& pv : {std::vector<Move>{}, full.pv}) {
                            htt.clear();
                            const int expected =
                                cpu_search_fixed(p, GameHistory{}, d, alpha, beta, c, pv).search.score;
                            for (const YbwcOptions& y : settings) {
                                dtt.clear();
                                device.set_tt(dtt.desc());
                                YbwcDeviceStats st;
                                const FixedDepthResult r =
                                    device.search(p, GameHistory{}, d, y, pv, &st, alpha, beta);
                                CHECK_MSG(!r.search.aborted && (r.search.score >= beta) == (expected >= beta),
                                          fens[f] + " " + describe(combos[k]) + " " + describe_prune(c.features, prune) +
                                              " depth " + std::to_string(d) + " window " + std::to_string(alpha) +
                                              (pv.empty() ? "" : " pv") + " leaf " + std::to_string(y.leaf_depth) +
                                              " budget " + std::to_string(y.node_budget) + ": " +
                                              std::to_string(r.search.score) + " vs " + std::to_string(expected));
                            }
                        }
                    }
                }
            }
        }
    }
}

TEST_CASE(gpu_see_mates_keep_their_distance) {
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
    for (const Combo& c : see_combos()) {
        GpuYbwcDevice device(gpu_test_env::context(), with_combo(default_variant(), c));
        for (const Case& m : cases) {
            const Position p = testutil::parse_or_die(m.fen);
            const int mate =
                cpu_search_fixed(p, GameHistory{}, m.depth, -kScoreInfinite, kScoreInfinite, {}).search.score;
            REQUIRE(is_mate_score(mate));
            for (const YbwcOptions& y : {YbwcOptions{0, 8, true, 16}, YbwcOptions{}}) {
                dtt.clear();
                device.set_tt(dtt.desc());
                std::vector<Move> pv;
                int score = 0;
                for (int d = 1; d <= m.depth + 2; ++d) {
                    dtt.new_search();
                    const FixedDepthResult r = device.search(p, GameHistory{}, d, y, pv);
                    score = r.search.score;
                    pv = r.pv;
                }
                CHECK_MSG(is_mate_score(score) && score >= mate,
                          std::string(m.fen) + " " + describe(c) + " leaf " + std::to_string(y.leaf_depth) + ": " +
                              std::to_string(score) + " vs " + std::to_string(mate));
            }
        }
    }
}

TEST_CASE(gpu_see_explicit_quiescence_values) {
    // A depth-0 search with a node budget of 1 splits its root as soon as the
    // root's search enters a capture, so the root becomes an explicit
    // quiescence node. With the zero window (a, a + 1), a = stand pat + the
    // second-largest gain among the root's captures + the delta margin + a
    // little, delta pruning skips every capture but the largest (whose SEE
    // the choice of positions keeps non-negative), and alpha can't move. The
    // explicit node then searches that one capture, nothing runs in
    // parallel, and its fail-soft value must equal the CPU search's exactly:
    // the stand pat, the skipped captures' delta bounds, the capture's result.
    const HqAttacks& attacks = gpu_search_util::host_attacks();
    std::vector<std::string> fens = positions();
    fens.push_back("4k3/8/8/3q4/4P3/8/1p6/1K6 w - - 0 1");    // exd5 (a queen), Kxb2 (a pawn)
    fens.push_back("4k3/7r/8/3q4/4P3/8/8/4K2R w - - 0 1");    // exd5, Rxh7 (a rook)
    fens.push_back("3k4/8/2n5/3q4/4P3/5P2/8/4K3 w - - 0 1");  // exd5, a defended queen
    DeviceTT dtt(kTableMb);
    HostTT htt(kTableMb);
    int cases = 0, split = 0, pruned = 0;
    for (const Combo& c : see_combos()) {
        GpuYbwcDevice device(gpu_test_env::context(), with_combo(default_variant(), c));
        const CpuSearchOptions cpu = cpu_combo(EvalKind::Pesto, c, htt);
        for (const std::string& fen : fens) {
            const Position p = testutil::parse_or_die(fen);
            const QuadBoard b = to_board<QuadBoard>(p);
            if (side_to_move_in_check(b, attacks)) continue;
            // The largest and second-largest gain among the quiescence moves.
            MoveList list;
            generate_legal(b, attacks, list);
            int best_gain = -1, second_gain = -1;
            Move best_move;
            for (int i = 0; i < list.size; ++i) {
                const Move m = list.moves[i];
                if (!is_quiescence_group(order_key(b, m, Move()))) continue;
                const int g = material_gain(b, m);
                if (g > best_gain) {
                    second_gain = best_gain;
                    best_gain = g;
                    best_move = m;
                } else if (g > second_gain) {
                    second_gain = g;
                }
            }
            if (second_gain < 0 || best_gain - second_gain < 60 || see(b, attacks, best_move) < 0) continue;
            const int stand = evaluate_position<PestoEval>(
                b, compute_accumulator<PestoEval>(b, host_search_tables().eval), state::side(p.state));
            for (const int extra : {0, 10, 40}) {
                const int alpha = stand + second_gain + PruneParams{}.delta_margin + extra;
                if (alpha + 1 >= kScoreMateBound || alpha <= -kScoreMateBound) continue;
                htt.clear();
                const int expected = cpu_search_fixed(p, GameHistory{}, 0, alpha, alpha + 1, cpu).search.score;
                dtt.clear();
                device.set_tt(dtt.desc());
                YbwcDeviceStats st;
                const FixedDepthResult r =
                    device.search(p, GameHistory{}, 0, YbwcOptions{0, 8, true, 1}, {}, &st, alpha, alpha + 1);
                CHECK_MSG(!r.search.aborted && r.search.score == expected,
                          fen + " " + describe(c) + " window " + std::to_string(alpha) + ": " +
                              std::to_string(r.search.score) + " vs " + std::to_string(expected) + " (" +
                              describe(st) + ")");
                ++cases;
                split += st.over_budget > 0 ? 1 : 0;
                pruned += st.qs_pruned > 0 ? 1 : 0;
            }
        }
    }
    CHECK_MSG(cases >= 6 && split > 0 && pruned > 0, std::to_string(cases) + " cases, " + std::to_string(split) +
                                                         " split roots, " + std::to_string(pruned) + " with pruning");
}

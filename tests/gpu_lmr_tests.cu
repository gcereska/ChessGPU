// Late move reductions on the GPU (feature sets tt-lmr and tt-nmp-lmr; see
// tests/test_lmr.cpp for the CPU side):
// - one task on one worker runs exactly the CPU's search, reductions and
//   re-searches included: same score, best move and node count in every
//   variant;
// - the GPU-resident YBWC reduces at explicit nodes too (a reduced child
//   with a zero window, replaced by a full one in the same slot when it
//   beats alpha), and with zero windows its outcomes equal the CPU's;
// - mates keep their distance, the search prunes, and the host-coordinated
//   searches run their tasks with reductions.
//
// Of 11 bugs planted one at a time in the explicit nodes' reductions, these
// tests catch 7 (one by hanging: a re-search that is reduced again never
// ends). Not caught: four window mistakes (re-searching only above beta,
// the reduced child's window, the task's refreshed window, the alpha of the
// re-search), all identical to the right code under the zero windows that
// make the GPU deterministic.

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

// The feature sets with reductions, each with a table layout, compiled ones only.
struct LmrCombo {
    FeatureKind features;
    TTLayoutKind layout;
};

std::vector<LmrCombo> lmr_combos() {
    std::vector<LmrCombo> v;
    for (const FeatureKind f : {FeatureKind::TTLmr, FeatureKind::TTNmpLmr})
        for (const TTLayoutKind l : {TTLayoutKind::Wide128, TTLayoutKind::Packed64})
            if (is_compiled(f, l)) v.push_back({f, l});
    return v;
}

std::string name(const LmrCombo& c) { return std::string(name_of(c.features)) + " " + name_of(c.layout); }

GpuSearchOptions with_lmr(GpuSearchOptions o, const LmrCombo& c, const PruneParams& prune = {}) {
    o.features = c.features;
    o.tt_layout = c.layout;
    o.prune = prune;
    return o;
}

CpuSearchOptions cpu_lmr(EvalKind eval, const LmrCombo& c, const HostTT& table, const PruneParams& prune = {}) {
    CpuSearchOptions o{BoardKind::Quad, eval, OrderingKind::Staged};
    o.features = c.features;
    o.tt_layout = c.layout;
    o.tt = table.desc();
    o.prune = prune;
    return o;
}

GpuSearchOptions default_variant() {
    GpuSearchOptions o;
    o.board = is_compiled(BoardKind::Quad) ? BoardKind::Quad : compiled_boards().front();
    o.model = compiled_models().front();
    if (o.model == ExecModel::Warp) o.block_size = 64;
    return o;
}

std::vector<std::string> positions() {
    std::vector<std::string> v = {
        "r1bq1rk1/pp2bppp/2n1pn2/3p4/2PP4/2N1PN2/PP3PPP/R2QKB1R w KQ - 0 8",
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        "2r2rk1/1bqnbppp/pp1ppn2/8/2PNP3/1PN1BP2/P2Q2PP/2RR1BK1 w - - 0 15",
        "3qk3/8/8/8/8/8/8/R2QK2R b KQ - 0 1",  // checks
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

// Settings that reduce from depth 2 and even the first move, so shallow
// searches reach every path of the reductions.
PruneParams steep() {
    PruneParams p;
    p.lmr_min_depth = 2;
    p.lmr_moves = 0;
    p.lmr_base = 150;
    p.lmr_divisor = 150;
    return p;
}

}  // namespace

TEST_CASE(gpu_lmr_single_tasks_match_cpu) {
    // Tasks one at a time (one worker each), tables shared across the tasks
    // on both sides, as gpu_tt_single_tasks_match_cpu.
    const bool quick = gpu_test_env::quick();
    const std::vector<std::string> fens = positions();
    const size_t count = quick ? 3 : fens.size();
    const int depth = quick ? 3 : 5;
    // Quick: the steep settings alone (they reduce at depth 3 too).
    const std::vector<PruneParams> settings = quick ? std::vector<PruneParams>{steep()}
                                                    : std::vector<PruneParams>{PruneParams{}, steep()};
    DeviceTT dtt(kTableMb);
    HostTT htt(kTableMb);
    int checked = 0;
    for (const BoardKind board : compiled_boards()) {
        for (const EvalKind eval : {EvalKind::Pesto, EvalKind::Material}) {
            if (!is_compiled(eval) || (quick && eval != EvalKind::Pesto)) continue;
            const std::vector<GpuSearchOptions> variants = compiled_variants(board, eval);
            const std::vector<LmrCombo> combos = lmr_combos();
            for (size_t v = 0; v < variants.size(); ++v) {
                for (size_t k = 0; k < combos.size(); ++k) {
                    if (quick_skip(quick, v, k)) continue;
                    for (const PruneParams& prune : settings) {
                        const GpuSearchOptions o = with_lmr(variants[v], combos[k], prune);
                        GpuSearcher gpu(gpu_test_env::context(), o);
                        dtt.clear();
                        htt.clear();
                        gpu.set_tt(dtt.desc());
                        CpuTaskRunner cpu(cpu_lmr(eval, combos[k], htt, prune));
                        for (size_t f = 0; f < count; ++f) {
                            for (int d = 1; d <= depth; ++d) {
                                TaskSpec t;
                                t.board = to_board<QuadBoard>(testutil::parse_or_die(fens[f]));
                                t.depth = d;
                                const SearchResult g = gpu.run({t}, {}, 0, {})[0];
                                const SearchResult r = cpu.run({t}, {}, 0, {})[0];
                                CHECK_MSG(g.score == r.score && g.best == r.best && g.nodes == r.nodes,
                                          describe(o) + " " + fens[f] + " depth " + std::to_string(d) + ": " +
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

TEST_CASE(gpu_lmr_ybwc_device) {
    // Iterative deepening on one table, with several YBWC settings: every
    // search finishes with a legal move, and explicit nodes reduce moves and
    // search some again.
    const bool quick = gpu_test_env::quick();
    const std::vector<std::string> fens = positions();
    const size_t count = quick ? 2 : 8;
    struct Setting {
        YbwcOptions y;
        int max_depth;
    };
    const Setting settings[] = {{{2, 8, true, 2048}, quick ? 4 : 7}, {{0, 8, true, 0}, quick ? 3 : 5},
                                {{1, 8, false, 0}, quick ? 3 : 5}};
    DeviceTT dtt(kTableMb);
    std::vector<GpuSearchOptions> variants;
    for (const BoardKind board : compiled_boards())
        for (const GpuSearchOptions& o : compiled_variants(board, EvalKind::Pesto))
            if (!quick || board == compiled_boards().front()) variants.push_back(o);
    const std::vector<LmrCombo> combos = lmr_combos();
    for (size_t v = 0; v < variants.size(); ++v) {
        for (size_t k = 0; k < combos.size(); ++k) {
            if (quick_skip(quick, v, k)) continue;
            const GpuSearchOptions o = with_lmr(variants[v], combos[k]);
            GpuYbwcDevice device(gpu_test_env::context(), o);
            unsigned reductions = 0, researches = 0;
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
                        CHECK_MSG(stats.lmr_researches <= stats.lmr_reductions, where + ": " + describe(stats));
                        reductions += stats.lmr_reductions;
                        researches += stats.lmr_researches;
                        pv = r.pv;
                    }
                }
            }
            // Quick searches are too small to need a re-search (the zero-window
            // test covers re-searches then).
            CHECK_MSG(reductions > 0 && (quick || researches > 0),
                      describe(o) + ": " + std::to_string(reductions) + " reduced at explicit nodes, " +
                          std::to_string(researches) + " searched again");
        }
    }
}

TEST_CASE(gpu_lmr_zero_window_matches_cpu) {
    // As gpu_nmp_zero_window_matches_cpu: with a zero window every node at a
    // ply has the same window, so explicit nodes reduce and search again
    // exactly where the CPU search does, and the outcomes match to depth 3.
    // The steep settings reduce from depth 2 and even the first move (the
    // eldest child's path), so depth 3 reaches every case.
    const bool quick = gpu_test_env::quick();
    const std::vector<std::string> fens = positions();
    const size_t count = quick ? 3 : fens.size();
    const YbwcOptions all_settings[] = {{0, 8, true, 0}, {0, 8, true, 16}, {1, 8, false, 0}, {2, 8, true, 2048}};
    const std::vector<YbwcOptions> settings(all_settings, all_settings + (quick ? 2 : 4));
    DeviceTT dtt(kTableMb);
    HostTT htt(kTableMb);
    unsigned reductions = 0, researches = 0;
    const std::vector<LmrCombo> combos = lmr_combos();
    for (size_t k = 0; k < combos.size(); ++k) {
        if (quick && k > 0) break;
        for (const PruneParams& prune : quick ? std::vector<PruneParams>{steep()}
                                              : std::vector<PruneParams>{PruneParams{}, steep()}) {
            GpuYbwcDevice device(gpu_test_env::context(), with_lmr(default_variant(), combos[k], prune));
            const CpuSearchOptions c = cpu_lmr(EvalKind::Pesto, combos[k], htt, prune);
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
                            const int expected = cpu_search_fixed(p, GameHistory{}, d, alpha, beta, c, pv).search.score;
                            for (const YbwcOptions& y : settings) {
                                dtt.clear();
                                device.set_tt(dtt.desc());
                                YbwcDeviceStats st;
                                const FixedDepthResult r = device.search(p, GameHistory{}, d, y, pv, &st, alpha, beta);
                                const std::string where = fens[f] + " " + name(combos[k]) + " lmr " +
                                                          std::to_string(prune.lmr_min_depth) + " depth " +
                                                          std::to_string(d) + " window " + std::to_string(alpha) +
                                                          (pv.empty() ? "" : " pv") + " leaf " +
                                                          std::to_string(y.leaf_depth) + " budget " +
                                                          std::to_string(y.node_budget);
                                CHECK_MSG(!r.search.aborted && (r.search.score >= beta) == (expected >= beta),
                                          where + ": " + std::to_string(r.search.score) + " vs " +
                                              std::to_string(expected) + " (" + describe(st) + ")");
                                reductions += st.lmr_reductions;
                                researches += st.lmr_researches;
                            }
                        }
                    }
                }
            }
        }
    }
    CHECK_MSG(reductions > 0 && researches > 0,
              std::to_string(reductions) + " reduced at explicit nodes, " + std::to_string(researches) +
                  " searched again");
}

namespace {

// How many of the root's moves the reduction rules pick at `depth`: in the
// root's move order (order_key, no table or PV move), the late quiet moves
// that don't give check, if the root isn't in check.
int expected_root_reductions(const Position& p, int depth, const PruneParams& prune) {
    const QuadBoard b = to_board<QuadBoard>(p);
    const HqAttacks& a = gpu_search_util::host_attacks();
    if (make_check_info(b, a).in_check()) return 0;
    MoveList list;
    generate_legal(b, a, list);
    std::vector<std::pair<u32, Move>> ordered;
    for (int i = 0; i < list.size; ++i) ordered.emplace_back(order_key(b, list.moves[i], Move()), list.moves[i]);
    std::sort(ordered.begin(), ordered.end(), [](const auto& x, const auto& y) { return x.first > y.first; });
    const LmrTable table = make_lmr_table(prune);
    int count = 0;
    for (size_t i = 0; i < ordered.size(); ++i) {
        const Move m = ordered[i].second;
        if (m.is_capture() || m.is_promotion() || table.reduction(depth, int(i) + 1) == 0) continue;
        QuadBoard child = b;
        make_move(child, m);
        if (!side_to_move_in_check(child, a)) ++count;
    }
    return count;
}

}  // namespace

TEST_CASE(gpu_lmr_explicit_reductions) {
    // The explicit nodes' reduction rules, counted exactly: at depth 3 with
    // leaf depth 2 only the root is explicit, and with a zero window above
    // its score the eldest child fails low and every brother starts at once
    // (the root is a PV node), so the root reduces exactly the moves the
    // rules pick (in check: none; captures, promotions and checks: never;
    // the table decides how late), and nothing beats alpha to be searched
    // again.
    const std::vector<std::string> fens = positions();
    const size_t count = gpu_test_env::quick() ? 4 : fens.size();
    DeviceTT dtt(kTableMb);
    HostTT htt(kTableMb);
    int reduced = 0;
    const std::vector<LmrCombo> combos = lmr_combos();
    for (size_t k = 0; k < combos.size(); ++k) {
        if (gpu_test_env::quick() && k > 0) break;
        for (const PruneParams& prune : {PruneParams{}, steep()}) {
            GpuYbwcDevice device(gpu_test_env::context(), with_lmr(default_variant(), combos[k], prune));
            for (size_t f = 0; f < count; ++f) {
                const Position p = testutil::parse_or_die(fens[f]);
                if (count_legal(to_board<QuadBoard>(p), gpu_search_util::host_attacks()) == 0) continue;
                htt.clear();
                const int s = cpu_search_fixed(p, GameHistory{}, 3, -kScoreInfinite, kScoreInfinite,
                                               cpu_lmr(EvalKind::Pesto, combos[k], htt, prune))
                                  .search.score;
                if (s + 101 >= kScoreMateBound) continue;  // the window must stay below mate scores
                dtt.clear();
                device.set_tt(dtt.desc());
                YbwcDeviceStats st;
                const FixedDepthResult r = device.search(p, GameHistory{}, 3, YbwcOptions{2, 8, false, 0}, {}, &st,
                                                         s + 100, s + 101);
                const int expected = expected_root_reductions(p, 3, prune);
                const std::string where = fens[f] + " " + name(combos[k]) + " lmr from " +
                                          std::to_string(prune.lmr_min_depth) + " after " +
                                          std::to_string(prune.lmr_moves);
                CHECK_MSG(!r.search.aborted && r.search.score <= s + 100, where + ": aborted or above the window");
                CHECK_MSG(st.lmr_reductions == unsigned(expected) && st.lmr_researches == 0,
                          where + ": " + std::to_string(st.lmr_reductions) + " reduced, " +
                              std::to_string(st.lmr_researches) + " searched again; expected " +
                              std::to_string(expected) + " reduced");
                reduced += expected;
            }
        }
    }
    CHECK(reduced > 0);
}

TEST_CASE(gpu_lmr_mates_keep_their_distance) {
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
    for (const LmrCombo& combo : lmr_combos()) {
        GpuYbwcDevice device(gpu_test_env::context(), with_lmr(default_variant(), combo));
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
                CHECK_MSG(score == mate, std::string(c.fen) + " " + name(combo) + " leaf " +
                                             std::to_string(y.leaf_depth) + ": " + std::to_string(score) + " vs " +
                                             std::to_string(mate));
            }
        }
    }
}

TEST_CASE(gpu_lmr_prunes) {
    // Fewer nodes to the same depth than the feature set without reductions.
    if (gpu_test_env::quick()) return;
    DeviceTT dtt(256);
    for (const LmrCombo& combo : lmr_combos()) {
        GpuSearchOptions base = default_variant();
        base.features = combo.features == FeatureKind::TTLmr ? FeatureKind::TT : FeatureKind::TTNmp;
        base.tt_layout = combo.layout;
        if (!is_compiled(base.features, base.tt_layout)) continue;
        GpuYbwcDevice with(gpu_test_env::context(), with_lmr(default_variant(), combo));
        GpuYbwcDevice without(gpu_test_env::context(), base);
        u64 nodes_with = 0, nodes_without = 0;
        for (const std::string& fen : positions()) {
            const Position p = testutil::parse_or_die(fen);
            for (GpuYbwcDevice* device : {&with, &without}) {
                dtt.clear();
                device->set_tt(dtt.desc());
                std::vector<Move> pv;
                for (int d = 1; d <= 8; ++d) {
                    dtt.new_search();
                    const FixedDepthResult r = device->search(p, GameHistory{}, d, YbwcOptions{}, pv);
                    (device == &with ? nodes_with : nodes_without) += r.search.nodes;
                    pv = r.pv;
                }
            }
        }
        CHECK_MSG(nodes_with < nodes_without, name(combo) + ": " + std::to_string(nodes_with) +
                                                  " nodes with reductions vs " + std::to_string(nodes_without));
    }
}

TEST_CASE(gpu_lmr_host_coordinated_searches) {
    const std::vector<std::string> fens = positions();
    const size_t count = gpu_test_env::quick() ? 2 : 6;
    DeviceTT dtt(kTableMb);
    for (const LmrCombo& combo : lmr_combos()) {
        GpuSearcher gpu(gpu_test_env::context(), with_lmr(default_variant(), combo));
        for (size_t f = 0; f < count; ++f) {
            const Position p = testutil::parse_or_die(fens[f]);
            const bool has_moves = count_legal(to_board<QuadBoard>(p), gpu_search_util::host_attacks()) > 0;
            dtt.clear();
            gpu.set_tt(dtt.desc());
            for (int d = 1; d <= 5; ++d) {
                const FixedDepthResult ind = search_independent(gpu, p, GameHistory{}, d, std::min(d, 2), {});
                const FixedDepthResult yb = search_ybwc(gpu, p, GameHistory{}, d, YbwcOptions{1, 8}, {});
                const std::string where = fens[f] + " " + name(combo) + " depth " + std::to_string(d);
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

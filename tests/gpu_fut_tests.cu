// Futility pruning on the GPU (feature sets tt-fut and tt-lmr-fut; see
// tests/test_fut.cpp for the CPU side):
// - one task on one worker runs exactly the CPU's search, pruning included:
//   same score, best move and node count in every variant;
// - the GPU-resident YBWC prunes at explicit nodes too (futile moves never
//   become children), with exactly the rules' count where that is
//   deterministic, and with zero windows its outcomes equal the CPU's;
// - mates keep their distance, the search prunes, and the host-coordinated
//   searches run their tasks with pruning.
//
// Of 8 bugs planted one at a time in the explicit nodes' futility pruning,
// these tests catch all 8.

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

struct FutCombo {
    FeatureKind features;
    TTLayoutKind layout;
};

std::vector<FutCombo> fut_combos() {
    std::vector<FutCombo> v;
    for (const FeatureKind f : {FeatureKind::TTFut, FeatureKind::TTLmrFut})
        for (const TTLayoutKind l : {TTLayoutKind::Wide128, TTLayoutKind::Packed64})
            if (is_compiled(f, l)) v.push_back({f, l});
    return v;
}

std::string name(const FutCombo& c) { return std::string(name_of(c.features)) + " " + name_of(c.layout); }

GpuSearchOptions with_fut(GpuSearchOptions o, const FutCombo& c, const PruneParams& prune = {}) {
    o.features = c.features;
    o.tt_layout = c.layout;
    o.prune = prune;
    return o;
}

CpuSearchOptions cpu_fut(EvalKind eval, const FutCombo& c, const HostTT& table, const PruneParams& prune = {}) {
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
        "3qk3/8/8/8/8/8/8/R2QK2R b KQ - 0 1",  // checks
        "r3k3/8/8/8/8/8/8/4K2R w Kq - 0 1",
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

// Settings that prune deep and much, so shallow searches reach every path.
PruneParams wide() {
    PruneParams p;
    p.fut_depth = 5;
    p.fut_margin = 40;
    return p;
}

}  // namespace

TEST_CASE(gpu_fut_single_tasks_match_cpu) {
    // Tasks one at a time (one worker each), tables shared across the tasks
    // on both sides, as gpu_tt_single_tasks_match_cpu; root tasks off the PV
    // too, where the root itself prunes.
    const bool quick = gpu_test_env::quick();
    const std::vector<std::string> fens = positions();
    const size_t count = quick ? 3 : fens.size();
    const int depth = quick ? 3 : 5;
    const std::vector<PruneParams> settings =
        quick ? std::vector<PruneParams>{wide()} : std::vector<PruneParams>{PruneParams{}, wide()};
    DeviceTT dtt(kTableMb);
    HostTT htt(kTableMb);
    int checked = 0;
    for (const BoardKind board : compiled_boards()) {
        for (const EvalKind eval : {EvalKind::Pesto, EvalKind::Material}) {
            if (!is_compiled(eval) || (quick && eval != EvalKind::Pesto)) continue;
            const std::vector<GpuSearchOptions> variants = compiled_variants(board, eval);
            const std::vector<FutCombo> combos = fut_combos();
            for (size_t v = 0; v < variants.size(); ++v) {
                for (size_t k = 0; k < combos.size(); ++k) {
                    if (quick_skip(quick, v, k)) continue;
                    for (const PruneParams& prune : settings) {
                        const GpuSearchOptions o = with_fut(variants[v], combos[k], prune);
                        GpuSearcher gpu(gpu_test_env::context(), o);
                        dtt.clear();
                        htt.clear();
                        gpu.set_tt(dtt.desc());
                        CpuTaskRunner cpu(cpu_fut(eval, combos[k], htt, prune));
                        for (size_t f = 0; f < count; ++f) {
                            for (int d = 1; d <= depth; ++d) {
                                for (const u8 flags : {kTaskOnPv, u8(0)}) {
                                    TaskSpec t;
                                    t.board = to_board<QuadBoard>(testutil::parse_or_die(fens[f]));
                                    t.depth = d;
                                    t.flags = flags;
                                    const SearchResult g = gpu.run({t}, {}, 0, {})[0];
                                    const SearchResult r = cpu.run({t}, {}, 0, {})[0];
                                    CHECK_MSG(g.score == r.score && g.best == r.best && g.nodes == r.nodes,
                                              describe(o) + " " + fens[f] + " depth " + std::to_string(d) +
                                                  " flags " + std::to_string(flags) + ": " + std::to_string(g.score) +
                                                  " " + move_to_uci(g.best) + " " + std::to_string(g.nodes) +
                                                  " vs " + std::to_string(r.score) + " " + move_to_uci(r.best) + " " +
                                                  std::to_string(r.nodes));
                                    ++checked;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    CHECK(checked > 0);
}

TEST_CASE(gpu_fut_ybwc_device) {
    // Iterative deepening on one table, with several YBWC settings: every
    // search finishes with a legal move, and explicit nodes prune.
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
    const std::vector<FutCombo> combos = fut_combos();
    for (size_t v = 0; v < variants.size(); ++v) {
        for (size_t k = 0; k < combos.size(); ++k) {
            if (quick_skip(quick, v, k)) continue;
            const GpuSearchOptions o = with_fut(variants[v], combos[k], wide());
            GpuYbwcDevice device(gpu_test_env::context(), o);
            unsigned pruned = 0;
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
                        pruned += stats.fut_pruned;
                        pv = r.pv;
                    }
                }
            }
            CHECK_MSG(pruned > 0, describe(o) + ": nothing pruned at explicit nodes");
        }
    }
}

TEST_CASE(gpu_fut_zero_window_matches_cpu) {
    // As gpu_nmp_zero_window_matches_cpu: with a zero window every node at a
    // ply has the same window, so explicit nodes prune exactly as the CPU
    // search does, and the outcomes match to depth 3.
    const bool quick = gpu_test_env::quick();
    const std::vector<std::string> fens = positions();
    const size_t count = quick ? 3 : fens.size();
    const YbwcOptions all_settings[] = {{0, 8, true, 0}, {0, 8, true, 16}, {1, 8, false, 0}, {2, 8, true, 2048}};
    const std::vector<YbwcOptions> settings(all_settings, all_settings + (quick ? 2 : 4));
    DeviceTT dtt(kTableMb);
    HostTT htt(kTableMb);
    unsigned pruned = 0;
    const std::vector<FutCombo> combos = fut_combos();
    for (size_t k = 0; k < combos.size(); ++k) {
        if (quick && k > 0) break;
        for (const PruneParams& prune :
             quick ? std::vector<PruneParams>{wide()} : std::vector<PruneParams>{PruneParams{}, wide()}) {
            GpuYbwcDevice device(gpu_test_env::context(), with_fut(default_variant(), combos[k], prune));
            const CpuSearchOptions c = cpu_fut(EvalKind::Pesto, combos[k], htt, prune);
            for (size_t f = 0; f < count; ++f) {
                const Position p = testutil::parse_or_die(fens[f]);
                for (int d = 1; d <= 3; ++d) {
                    htt.clear();
                    const FixedDepthResult full =
                        cpu_search_fixed(p, GameHistory{}, d, -kScoreInfinite, kScoreInfinite, c);
                    const int s = full.search.score;
                    for (const int alpha : {s - 301, s - 31, s - 1, s, s + 29, s + 300}) {
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
                                const std::string where = fens[f] + " " + name(combos[k]) + " fut " +
                                                          std::to_string(prune.fut_margin) + " depth " +
                                                          std::to_string(d) + " window " + std::to_string(alpha) +
                                                          (pv.empty() ? "" : " pv") + " leaf " +
                                                          std::to_string(y.leaf_depth) + " budget " +
                                                          std::to_string(y.node_budget);
                                CHECK_MSG(!r.search.aborted && (r.search.score >= beta) == (expected >= beta),
                                          where + ": " + std::to_string(r.search.score) + " vs " +
                                              std::to_string(expected) + " (" + describe(st) + ")");
                                pruned += st.fut_pruned;
                            }
                        }
                    }
                }
            }
        }
    }
    CHECK_MSG(pruned > 0, "nothing pruned at explicit nodes");
}

TEST_CASE(gpu_fut_explicit_pruning) {
    // The explicit nodes' rules, counted exactly: at depth 2 with leaf depth
    // 0, the root's children are explicit; with the lowest zero window the
    // root's eldest child (its first move in order) must fail high at once,
    // and that child, where alpha is as high as it gets, prunes exactly its
    // quiet moves that don't give check, unless it's in check. Where that is
    // every move it has (a lone king's replies), the child scores exactly
    // its futility bound, and the root minus that.
    std::vector<std::string> fens = {
        "8/8/4k3/8/8/8/3PK3/8 w - - 0 1",  // first move d3; black has king moves only
        "8/8/8/4k3/8/8/4P3/4K3 w - - 0 1",
    };
    for (const std::string& fen : positions()) fens.push_back(fen);
    const size_t count = gpu_test_env::quick() ? 6 : fens.size();
    const HqAttacks& attacks = gpu_search_util::host_attacks();
    DeviceTT dtt(kTableMb);
    int expected_total = 0, all_futile = 0;
    for (const FutCombo& combo : fut_combos()) {
        if (combo.features != FeatureKind::TTFut) continue;  // reductions don't change one node's pruning
        GpuYbwcDevice device(gpu_test_env::context(), with_fut(default_variant(), combo));
        for (size_t f = 0; f < count; ++f) {
            const Position p = testutil::parse_or_die(fens[f]);
            const QuadBoard b = to_board<QuadBoard>(p);
            MoveList list;
            generate_legal(b, attacks, list);
            if (list.size == 0) continue;
            Move first = list.moves[0];
            for (int i = 1; i < list.size; ++i)
                if (order_key(b, list.moves[i], Move()) > order_key(b, first, Move())) first = list.moves[i];
            QuadBoard child = b;
            make_move(child, first);
            MoveList replies;
            generate_legal(child, attacks, replies);
            // Over before any pruning: no moves, a dead draw, the 50-move rule.
            if (replies.size == 0 || insufficient_material(child) || state::halfmove(child.state()) >= 100) continue;
            int expected = 0;
            if (!make_check_info(child, attacks).in_check())
                for (int i = 0; i < replies.size; ++i) {
                    const Move m = replies.moves[i];
                    if (!m.is_capture() && !m.is_promotion() && !gives_check(child, m, attacks)) ++expected;
                }
            const int alpha = -kScoreMateBound + 1;  // the child's alpha: kScoreMateBound - 2
            dtt.clear();
            device.set_tt(dtt.desc());
            YbwcDeviceStats st;
            const FixedDepthResult r = device.search(p, GameHistory{}, 2, YbwcOptions{0, 8, true, 0}, {}, &st, alpha,
                                                     alpha + 1);
            CHECK_MSG(!r.search.aborted && r.search.score > alpha && st.fut_pruned == unsigned(expected),
                      fens[f] + " " + name(combo) + ": " + std::to_string(st.fut_pruned) + " pruned, expected " +
                          std::to_string(expected) + " (" + describe(st) + ")");
            if (expected == replies.size) {
                const int eval = PestoEval::evaluate(compute_accumulator<PestoEval>(child, host_search_tables().eval),
                                                     state::side(child.state()));
                const int bound = eval + PruneParams{}.fut_margin * 1;  // one ply left at the child
                CHECK_MSG(r.search.score == -bound, fens[f] + " " + name(combo) + ": every reply futile, root " +
                                                        std::to_string(r.search.score) + " vs " +
                                                        std::to_string(-bound));
                ++all_futile;
            }
            expected_total += expected;
        }
    }
    CHECK(expected_total > 0 && all_futile > 0);
}

TEST_CASE(gpu_fut_mates_keep_their_distance) {
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
    for (const FutCombo& combo : fut_combos()) {
        GpuYbwcDevice device(gpu_test_env::context(), with_fut(default_variant(), combo));
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

TEST_CASE(gpu_fut_prunes) {
    // Fewer nodes to the same depth than the feature set without pruning.
    if (gpu_test_env::quick()) return;
    DeviceTT dtt(256);
    for (const FutCombo& combo : fut_combos()) {
        GpuSearchOptions base = default_variant();
        base.features = combo.features == FeatureKind::TTFut ? FeatureKind::TT : FeatureKind::TTLmr;
        base.tt_layout = combo.layout;
        if (!is_compiled(base.features, base.tt_layout)) continue;
        GpuYbwcDevice with(gpu_test_env::context(), with_fut(default_variant(), combo));
        GpuYbwcDevice without(gpu_test_env::context(), base);
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
        CHECK_MSG(nodes_with < nodes_without, name(combo) + ": " + std::to_string(nodes_with) +
                                                  " nodes with pruning vs " + std::to_string(nodes_without));
    }
}

TEST_CASE(gpu_fut_host_coordinated_searches) {
    const std::vector<std::string> fens = positions();
    const size_t count = gpu_test_env::quick() ? 2 : 6;
    DeviceTT dtt(kTableMb);
    for (const FutCombo& combo : fut_combos()) {
        GpuSearcher gpu(gpu_test_env::context(), with_fut(default_variant(), combo));
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

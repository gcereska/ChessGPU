// Killer moves on the GPU (feature sets tt-killers and tt-ext-killers; see
// tests/test_killers.cpp for the CPU side):
// - one task on one worker runs exactly the CPU's search, killers included:
//   same score, best move and node count in every variant;
// - the GPU-resident YBWC's explicit nodes share a killer table by ply: a
//   quiet move ending its parent with a cutoff is stored at the parent's
//   ply (the root's own killer is predictable), explicit nodes find killers
//   among their moves, searches end with legal moves, and with zero windows
//   the outcomes equal the CPU's;
// - mates keep their distance.
// The same checks run for the feature sets that add null moves or late move
// reductions on top of killers (tt-killers-nmp, tt-killers-lmr and their
// tt-ext- versions).

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

const FeatureKind kKillerSets[] = {FeatureKind::TTKillers, FeatureKind::TTExtKillers};
// Null moves and reductions again on top of killers: wide128 only (their
// own techniques have their own GPU tests in both layouts).
const FeatureKind kRetrySets[] = {FeatureKind::TTKillersNmp, FeatureKind::TTKillersLmr, FeatureKind::TTExtKillersNmp,
                                  FeatureKind::TTExtKillersLmr, FeatureKind::TTKillersSeePvs,
                                  FeatureKind::TTKillersSeeQchecks, FeatureKind::TTKillersSeeQchecksNmp};

// Compiled (feature set, layout) pairs with killers.
struct Combo {
    FeatureKind features;
    TTLayoutKind layout;
};
std::vector<Combo> killer_combos() {
    std::vector<Combo> v;
    for (const FeatureKind f : kKillerSets)
        for (const TTLayoutKind l : {TTLayoutKind::Wide128, TTLayoutKind::Packed64})
            if (is_compiled(f, l)) v.push_back({f, l});
    for (const FeatureKind f : kRetrySets)
        if (is_compiled(f, TTLayoutKind::Wide128)) v.push_back({f, TTLayoutKind::Wide128});
    return v;
}

GpuSearchOptions with_combo(GpuSearchOptions o, const Combo& c) {
    o.features = c.features;
    o.tt_layout = c.layout;
    return o;
}

CpuSearchOptions cpu_combo(EvalKind eval, const Combo& c, const HostTT& table) {
    CpuSearchOptions o{BoardKind::Quad, eval, OrderingKind::Staged};
    o.features = c.features;
    o.tt_layout = c.layout;
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
        "r1bq1rk1/pp2bppp/2n1pn2/3p4/2PP4/2N1PN2/PP3PPP/R2QKB1R w KQ - 0 8",
        "2r2rk1/1bqnbppp/pp1ppn2/8/2PNP3/1PN1BP2/P2Q2PP/2RR1BK1 w - - 0 15",
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

}  // namespace

TEST_CASE(gpu_killers_single_tasks_match_cpu) {
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
            const std::vector<Combo> combos = killer_combos();
            for (size_t v = 0; v < variants.size(); ++v) {
                for (size_t k = 0; k < combos.size(); ++k) {
                    if (quick_skip(quick, v, k)) continue;
                    const GpuSearchOptions o = with_combo(variants[v], combos[k]);
                    GpuSearcher gpu(gpu_test_env::context(), o);
                    dtt.clear();
                    htt.clear();
                    gpu.set_tt(dtt.desc());
                    CpuTaskRunner cpu(cpu_combo(eval, combos[k], htt));
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
    CHECK(checked > 0);
}

TEST_CASE(gpu_killers_ybwc_device) {
    // Iterative deepening on one table: every search ends with a legal move;
    // explicit nodes store killers and find them among their moves (none of
    // that without killers).
    const bool quick = gpu_test_env::quick();
    const std::vector<std::string> fens = positions();
    const size_t count = quick ? 2 : 8;
    struct Setting {
        YbwcOptions y;
        int max_depth;
    };
    const Setting settings[] = {{{2, 8, true, 2048}, quick ? 5 : 7}, {{0, 8, true, 0}, quick ? 3 : 4},
                                {{1, 8, false, 0}, quick ? 4 : 5}};
    DeviceTT dtt(kTableMb);
    std::vector<GpuSearchOptions> variants;
    for (const BoardKind board : compiled_boards())
        for (const GpuSearchOptions& o : compiled_variants(board, EvalKind::Pesto))
            if (!quick || board == compiled_boards().front()) variants.push_back(o);
    const std::vector<Combo> combos = killer_combos();
    for (size_t v = 0; v < variants.size(); ++v) {
        for (size_t k = 0; k < combos.size(); ++k) {
            if (quick_skip(quick, v, k)) continue;
            const GpuSearchOptions o = with_combo(variants[v], combos[k]);
            GpuYbwcDevice device(gpu_test_env::context(), o);
            unsigned stores = 0, used = 0, explicit_nodes = 0;
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
                                                  " leaf " + std::to_string(s.y.leaf_depth);
                        CHECK_MSG(!r.search.aborted && !stats.overflow, where + ": aborted (" + describe(stats) + ")");
                        CHECK_MSG(has_moves ? is_legal_move(p, r.search.best) : r.search.best.is_none(),
                                  where + ": best " + move_to_uci(r.search.best));
                        CHECK_MSG(stats.killers_used <= stats.explicit_nodes, where + ": " + describe(stats));
                        stores += stats.killer_stores;
                        used += stats.killers_used;
                        explicit_nodes += stats.explicit_nodes;
                        pv = r.pv;
                    }
                }
            }
            // A move of the wrong side (a killer stored a ply off) is never
            // among a node's moves, so explicit nodes finding killers show the
            // plies line up: a sizable share of them with killers alone; with
            // reductions too, whose explicit trees are narrow, at least some.
            const bool retry = uses_null_move(combos[k].features) || uses_lmr(combos[k].features);
            CHECK_MSG(stores > 0 && used > 0 && (retry || used * 20 > explicit_nodes),
                      cb::describe(o) + ": killers stored " + std::to_string(stores) + ", used at " +
                          std::to_string(used) + " of " + std::to_string(explicit_nodes) + " explicit nodes");
        }
    }
    // Without killers: no table, nothing stored or used.
    GpuSearchOptions plain = default_variant();
    plain.features = FeatureKind::TT;
    if (is_compiled(FeatureKind::TT, TTLayoutKind::Wide128)) {
        GpuYbwcDevice device(gpu_test_env::context(), plain);
        dtt.clear();
        device.set_tt(dtt.desc());
        YbwcDeviceStats stats;
        const FixedDepthResult r =
            device.search(testutil::parse_or_die(fens[0]), GameHistory{}, 5, YbwcOptions{1, 8, true, 0}, {}, &stats);
        CHECK_MSG(!r.search.aborted && stats.explicit_nodes > 1 && stats.killer_stores == 0 &&
                      stats.killers_used == 0 && device.explicit_killers().empty(),
                  describe(stats));
    }
}

TEST_CASE(gpu_killers_root_killer) {
    // With the lowest zero window, the root's eldest child fails high at once
    // (unless it gets mated: the CPU's search, which tries the same move
    // first, then answers with another move) and ends the root: its move
    // becomes the root ply's only killer if it is quiet (the start position's
    // eldest is Nb1-a3, the knight move first by order), and the root ply has
    // none otherwise. Deeper explicit nodes store at their own plies only.
    const HqAttacks& attacks = gpu_search_util::host_attacks();
    DeviceTT dtt(kTableMb);
    HostTT htt(kTableMb);
    int quiet = 0, other = 0;
    for (const Combo& c : killer_combos()) {
        GpuYbwcDevice device(gpu_test_env::context(), with_combo(default_variant(), c));
        const CpuSearchOptions cpu = cpu_combo(EvalKind::Pesto, c, htt);
        std::vector<std::string> fens = positions();
        fens.push_back("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
        for (const std::string& fen : fens) {
            const Position p = testutil::parse_or_die(fen);
            const QuadBoard b = to_board<QuadBoard>(p);
            MoveList list;
            generate_legal(b, attacks, list);
            if (list.size == 0) continue;
            Move eldest = list.moves[0];
            for (int i = 1; i < list.size; ++i)
                if (order_key(b, list.moves[i], Move()) > order_key(b, eldest, Move())) eldest = list.moves[i];
            QuadBoard child = b;
            make_move(child, eldest);
            if (insufficient_material(child) || count_legal(child, attacks) == 0) continue;
            const int alpha = -kScoreMateBound + 1;
            htt.clear();
            if (cpu_search_fixed(p, GameHistory{}, 4, alpha, alpha + 1, cpu).search.best != eldest) continue;
            for (const YbwcOptions& y : {YbwcOptions{1, 8, true, 0}, YbwcOptions{0, 8, false, 0}}) {
                dtt.clear();
                device.set_tt(dtt.desc());
                YbwcDeviceStats st;
                const FixedDepthResult r = device.search(p, GameHistory{}, 4, y, {}, &st, alpha, alpha + 1);
                const std::vector<Killers> k = device.explicit_killers();
                const std::string where = fen + " " + describe(c) + " leaf " + std::to_string(y.leaf_depth) +
                                          ": eldest " + move_to_uci(eldest) + ", root killers " +
                                          move_to_uci(k.empty() ? Move() : k[0].first) + " " +
                                          move_to_uci(k.empty() ? Move() : k[0].second);
                REQUIRE_MSG(!r.search.aborted && k.size() == size_t(kMaxPly + 1), where);
                const bool is_quiet = can_be_killer(eldest);
                CHECK_MSG(k[0].first == (is_quiet ? eldest : Move()) && k[0].second.is_none(), where);
                (is_quiet ? quiet : other) += 1;
            }
        }
    }
    CHECK_MSG(quiet > 0 && other > 0, std::to_string(quiet) + " quiet eldest moves, " + std::to_string(other) +
                                          " others");
}

TEST_CASE(gpu_killers_zero_window_matches_cpu) {
    // As gpu_ext_zero_window_matches_cpu: whatever order killers give, the
    // outcomes with zero windows match the CPU's to depth 3.
    const bool quick = gpu_test_env::quick();
    const std::vector<std::string> fens = positions();
    const size_t count = quick ? 3 : fens.size();
    const YbwcOptions all_settings[] = {{0, 8, true, 0}, {0, 8, true, 16}, {1, 8, false, 0}, {2, 8, true, 2048}};
    const std::vector<YbwcOptions> settings(all_settings, all_settings + (quick ? 2 : 4));
    DeviceTT dtt(kTableMb);
    HostTT htt(kTableMb);
    const std::vector<Combo> combos = killer_combos();
    for (size_t k = 0; k < combos.size(); ++k) {
        if (quick && k > 0) break;
        GpuYbwcDevice device(gpu_test_env::context(), with_combo(default_variant(), combos[k]));
        const CpuSearchOptions c = cpu_combo(EvalKind::Pesto, combos[k], htt);
        for (size_t f = 0; f < count; ++f) {
            const Position p = testutil::parse_or_die(fens[f]);
            for (int d = 1; d <= 3; ++d) {
                htt.clear();
                const FixedDepthResult full = cpu_search_fixed(p, GameHistory{}, d, -kScoreInfinite, kScoreInfinite, c);
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
                            CHECK_MSG(!r.search.aborted && (r.search.score >= beta) == (expected >= beta),
                                      fens[f] + " " + describe(combos[k]) + " depth " + std::to_string(d) +
                                          " window " + std::to_string(alpha) + (pv.empty() ? "" : " pv") +
                                          " leaf " + std::to_string(y.leaf_depth) + ": " +
                                          std::to_string(r.search.score) + " vs " + std::to_string(expected));
                        }
                    }
                }
            }
        }
    }
}

TEST_CASE(gpu_killers_mates_keep_their_distance) {
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
    for (const Combo& c : killer_combos()) {
        GpuYbwcDevice device(gpu_test_env::context(), with_combo(default_variant(), c));
        for (const Case& m : cases) {
            const Position p = testutil::parse_or_die(m.fen);
            const int mate =
                cpu_search_fixed(p, GameHistory{}, m.depth, -kScoreInfinite, kScoreInfinite, {}).search.score;
            REQUIRE(is_mate_score(mate));
            for (const YbwcOptions& y : {YbwcOptions{0, 8, true, 0}, YbwcOptions{}}) {
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
                // Extensions may find a shorter mate, never a longer one.
                CHECK_MSG(is_mate_score(score) && score >= mate,
                          std::string(m.fen) + " " + describe(c) + " leaf " + std::to_string(y.leaf_depth) + ": " +
                              std::to_string(score) + " vs " + std::to_string(mate));
            }
        }
    }
}

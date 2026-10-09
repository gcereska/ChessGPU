// Check extensions on the GPU (feature set tt-ext; see tests/test_ext.cpp
// for the CPU side):
// - one task on one worker runs exactly the CPU's search: same score, best
//   move and node count in every variant;
// - the GPU-resident YBWC extends at explicit nodes too (a checking move's
//   child keeps its parent's depth), and with zero windows its outcomes
//   equal the CPU's;
// - mates keep their distance.
//
// Bugs planted at the explicit nodes (never extending; extending every
// move, which never ends) are both caught.

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

std::vector<TTLayoutKind> ext_layouts() {
    std::vector<TTLayoutKind> v;
    for (const TTLayoutKind l : {TTLayoutKind::Wide128, TTLayoutKind::Packed64})
        if (is_compiled(FeatureKind::TTExt, l)) v.push_back(l);
    return v;
}

GpuSearchOptions with_ext(GpuSearchOptions o, TTLayoutKind layout) {
    o.features = FeatureKind::TTExt;
    o.tt_layout = layout;
    return o;
}

CpuSearchOptions cpu_ext(EvalKind eval, TTLayoutKind layout, const HostTT& table) {
    CpuSearchOptions o{BoardKind::Quad, eval, OrderingKind::Staged};
    o.features = FeatureKind::TTExt;
    o.tt_layout = layout;
    o.tt = table.desc();
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
        "3qk3/8/8/8/8/8/8/R2QK2R b KQ - 0 1",  // many checks
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
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

TEST_CASE(gpu_ext_single_tasks_match_cpu) {
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
            const std::vector<TTLayoutKind> layouts = ext_layouts();
            for (size_t v = 0; v < variants.size(); ++v) {
                for (size_t k = 0; k < layouts.size(); ++k) {
                    if (quick_skip(quick, v, k)) continue;
                    const GpuSearchOptions o = with_ext(variants[v], layouts[k]);
                    GpuSearcher gpu(gpu_test_env::context(), o);
                    dtt.clear();
                    htt.clear();
                    gpu.set_tt(dtt.desc());
                    CpuTaskRunner cpu(cpu_ext(eval, layouts[k], htt));
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
    CHECK(checked > 0);
}

TEST_CASE(gpu_ext_ybwc_device) {
    // Iterative deepening on one table: every search finishes with a legal
    // move, also where the explicit tree grows through series of checks.
    const bool quick = gpu_test_env::quick();
    const std::vector<std::string> fens = positions();
    const size_t count = quick ? 2 : 8;
    struct Setting {
        YbwcOptions y;
        int max_depth;
    };
    const Setting settings[] = {{{2, 8, true, 2048}, quick ? 4 : 7}, {{0, 8, true, 0}, quick ? 3 : 4},
                                {{1, 8, false, 0}, quick ? 3 : 5}};
    DeviceTT dtt(kTableMb);
    std::vector<GpuSearchOptions> variants;
    for (const BoardKind board : compiled_boards())
        for (const GpuSearchOptions& o : compiled_variants(board, EvalKind::Pesto))
            if (!quick || board == compiled_boards().front()) variants.push_back(o);
    const std::vector<TTLayoutKind> layouts = ext_layouts();
    for (size_t v = 0; v < variants.size(); ++v) {
        for (size_t k = 0; k < layouts.size(); ++k) {
            if (quick_skip(quick, v, k)) continue;
            const GpuSearchOptions o = with_ext(variants[v], layouts[k]);
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
                        const std::string where = describe(o) + " " + fens[f] + " depth " + std::to_string(d) +
                                                  " leaf " + std::to_string(s.y.leaf_depth);
                        CHECK_MSG(!r.search.aborted && !stats.overflow, where + ": aborted (" + describe(stats) + ")");
                        CHECK_MSG(has_moves ? is_legal_move(p, r.search.best) : r.search.best.is_none(),
                                  where + ": best " + move_to_uci(r.search.best));
                        pv = r.pv;
                    }
                }
            }
        }
    }
}

TEST_CASE(gpu_ext_zero_window_matches_cpu) {
    // As gpu_nmp_zero_window_matches_cpu: explicit nodes extend exactly as
    // the CPU search does, so with zero windows the outcomes match to depth 3.
    const bool quick = gpu_test_env::quick();
    const std::vector<std::string> fens = positions();
    const size_t count = quick ? 3 : fens.size();
    const YbwcOptions all_settings[] = {{0, 8, true, 0}, {0, 8, true, 16}, {1, 8, false, 0}, {2, 8, true, 2048}};
    const std::vector<YbwcOptions> settings(all_settings, all_settings + (quick ? 2 : 4));
    DeviceTT dtt(kTableMb);
    HostTT htt(kTableMb);
    const std::vector<TTLayoutKind> layouts = ext_layouts();
    for (size_t k = 0; k < layouts.size(); ++k) {
        if (quick && k > 0) break;
        GpuYbwcDevice device(gpu_test_env::context(), with_ext(default_variant(), layouts[k]));
        const CpuSearchOptions c = cpu_ext(EvalKind::Pesto, layouts[k], htt);
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
                                      fens[f] + " " + name_of(layouts[k]) + " depth " + std::to_string(d) +
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

TEST_CASE(gpu_ext_explicit_extensions) {
    // An explicit node's checking children keep its depth: at depth 2 with
    // leaf depth 1, a root move's child has depth 1 (a task) unless the move
    // gives check (depth 2: explicit). With the lowest zero window the
    // root's eldest child fails high at once, so the root is the only
    // explicit node unless its first move gives check (then the eldest is
    // explicit too, and maybe some of its own checking replies).
    const HqAttacks& attacks = gpu_search_util::host_attacks();
    DeviceTT dtt(kTableMb);
    int extended = 0, plain = 0;
    for (const TTLayoutKind layout : ext_layouts()) {
        GpuYbwcDevice device(gpu_test_env::context(), with_ext(default_variant(), layout));
        for (const std::string& fen : positions()) {
            const Position p = testutil::parse_or_die(fen);
            const QuadBoard b = to_board<QuadBoard>(p);
            MoveList list;
            generate_legal(b, attacks, list);
            if (list.size == 0) continue;
            Move first = list.moves[0];
            for (int i = 1; i < list.size; ++i)
                if (order_key(b, list.moves[i], Move()) > order_key(b, first, Move())) first = list.moves[i];
            QuadBoard child = b;
            make_move(child, first);
            if (insufficient_material(child) || count_legal(child, attacks) == 0) continue;
            const bool check = side_to_move_in_check(child, attacks);
            const int alpha = -kScoreMateBound + 1;
            dtt.clear();
            device.set_tt(dtt.desc());
            YbwcDeviceStats st;
            const FixedDepthResult r =
                device.search(p, GameHistory{}, 2, YbwcOptions{1, 8, true, 0}, {}, &st, alpha, alpha + 1);
            CHECK_MSG(!r.search.aborted && (check ? st.explicit_nodes >= 2 : st.explicit_nodes == 1),
                      fen + " " + name_of(layout) + ": eldest " + move_to_uci(first) + (check ? " checks" : "") +
                          ", " + describe(st));
            (check ? extended : plain) += 1;
        }
    }
    CHECK_MSG(extended > 0 && plain > 0, std::to_string(extended) + " checking eldest moves, " + std::to_string(plain) +
                                             " others");
}

TEST_CASE(gpu_ext_mates_keep_their_distance) {
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
    for (const TTLayoutKind layout : ext_layouts()) {
        GpuYbwcDevice device(gpu_test_env::context(), with_ext(default_variant(), layout));
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
                for (int d = 1; d <= c.depth + 2; ++d) {
                    dtt.new_search();
                    const FixedDepthResult r = device.search(p, GameHistory{}, d, y, pv);
                    score = r.search.score;
                    pv = r.pv;
                }
                // Extensions may find a shorter mate than the plain search's, never a longer one.
                CHECK_MSG(is_mate_score(score) && score >= mate,
                          std::string(c.fen) + " " + name_of(layout) + " leaf " + std::to_string(y.leaf_depth) +
                              ": " + std::to_string(score) + " vs " + std::to_string(mate));
            }
        }
    }
}

// Late move reductions (core/search.h; feature sets tt-lmr and tt-nmp-lmr):
// the reduction table, and the search with it.
//
// What the search must guarantee:
// - its reductions and re-searches match a recursive implementation exactly
//   (score and node count; without a table, through test-only feature sets),
//   alone and with null moves, with every setting and window;
// - with a table, exactly the search without reductions when no move is
//   ever reduced (tt-lmr is tt then, tt-nmp-lmr is tt-nmp);
// - mates keep their exact distance; fewer nodes to the same depth; every
//   board and ordering visits the same tree.

#include <algorithm>
#include <stdexcept>
#include <string>
#include <vector>

#include "core/search.h"
#include "host/fen.h"
#include "host/search_cpu.h"
#include "host/tt_storage.h"
#include "recursive_search_ref.h"
#include "test_harness.h"
#include "test_util.h"

using namespace cb;

namespace {

#ifdef NDEBUG
constexpr int kRefDepth = 6;
constexpr int kNodesDepth = 7;
#else
constexpr int kRefDepth = 4;
constexpr int kNodesDepth = 5;
#endif

// Without a table, so the flat search can be compared with the recursive
// reference (tests/recursive_search_ref.h).
struct FeaturesLmrOnly {
    static constexpr const char* kName = "lmr-only";
    static constexpr bool kTT = false;
    static constexpr bool kTTCutoffs = false;
    static constexpr bool kTTQuiescence = false;
    static constexpr bool kNullMove = false;
    static constexpr bool kLmr = true;
    static constexpr bool kFutility = false;
    static constexpr bool kCheckExt = false;
    static constexpr bool kKillers = false;
    static constexpr bool kQPrune = false;
};

struct FeaturesNullLmr {
    static constexpr const char* kName = "null-lmr";
    static constexpr bool kTT = false;
    static constexpr bool kTTCutoffs = false;
    static constexpr bool kTTQuiescence = false;
    static constexpr bool kNullMove = true;
    static constexpr bool kLmr = true;
    static constexpr bool kFutility = false;
    static constexpr bool kCheckExt = false;
    static constexpr bool kKillers = false;
    static constexpr bool kQPrune = false;
};

std::vector<std::string> positions() {
    std::vector<std::string> v = {
        "r1bq1rk1/pp2bppp/2n1pn2/3p4/2PP4/2N1PN2/PP3PPP/R2QKB1R w KQ - 0 8",
        "2r2rk1/1bqnbppp/pp1ppn2/8/2PNP3/1PN1BP2/P2Q2PP/2RR1BK1 w - - 0 15",
        "r2q1rk1/1p1nbppp/p2pbn2/4p3/4P3/1NN1BP2/PPPQ2PP/2KR1B1R w - - 0 11",
        "8/5k2/3p4/1p1P4/1P6/5K2/8/8 w - - 0 1",  // kings and pawns: reductions, no null moves
    };
    for (const std::string& fen : testutil::test_fens()) v.push_back(fen);
    return v;
}

std::string settings_text(const PruneParams& p) {
    return "lmr " + std::to_string(p.lmr_base) + "/" + std::to_string(p.lmr_divisor) + " from " +
           std::to_string(p.lmr_min_depth) + " after " + std::to_string(p.lmr_moves);
}

// The flat search against the reference on every position, in iterative
// deepening (each depth ordered by the previous one's PV), with the full
// window and with zero windows around the score.
template <class Board, class Eval, class Features, bool kNullMove>
void check_against_recursive(const PruneParams& prune, int max_depth = kRefDepth) {
    u64 reductions = 0, researches = 0;
    for (const std::string& fen : positions()) {
        const Position p = testutil::parse_or_die(fen);
        std::vector<Move> pv;
        for (int depth = 1; depth <= max_depth; ++depth) {
            const testref::Outcome flat = testref::flat_search<Board, Eval, Features>(
                p, depth, -kScoreInfinite, kScoreInfinite, 0, kTaskOnPv, prune, pv);
            auto rec = testref::reference<Board, Eval, kNullMove, true>(p, prune, pv);
            const int score = rec.search(depth, -kScoreInfinite, kScoreInfinite, 0, 0, true, false);
            const std::string where = fen + " depth " + std::to_string(depth) + " " + settings_text(prune);
            CHECK_MSG(flat.score == score && flat.nodes == rec.nodes,
                      where + ": " + std::to_string(flat.score) + "/" + std::to_string(flat.nodes) + " vs " +
                          std::to_string(score) + "/" + std::to_string(rec.nodes));
            reductions += rec.reductions;
            researches += rec.researches;
            for (const int alpha : {score - 1, score}) {
                if (alpha <= -kScoreInfinite || alpha + 1 >= kScoreInfinite) continue;
                const testref::Outcome zero =
                    testref::flat_search<Board, Eval, Features>(p, depth, alpha, alpha + 1, 0, kTaskOnPv, prune, pv);
                auto zrec = testref::reference<Board, Eval, kNullMove, true>(p, prune, pv);
                const int zscore = zrec.search(depth, alpha, alpha + 1, 0, 0, true, false);
                CHECK_MSG(zero.score == zscore && zero.nodes == zrec.nodes,
                          where + " window " + std::to_string(alpha) + ": " + std::to_string(zero.score) + "/" +
                              std::to_string(zero.nodes) + " vs " + std::to_string(zscore) + "/" +
                              std::to_string(zrec.nodes));
            }
            pv = flat.pv;
        }
    }
    // Not vacuous: moves were reduced, and some searched again.
    CHECK_MSG(reductions > 0 && researches > 0 && researches < reductions,
              std::to_string(reductions) + " reductions, " + std::to_string(researches) + " re-searches");
}

const TTLayoutKind kLayouts[] = {TTLayoutKind::Packed64, TTLayoutKind::Wide128};

// Each feature set with reductions, and the one it adds them to.
struct LmrSet {
    FeatureKind with;
    FeatureKind without;
};
const LmrSet kSets[] = {{FeatureKind::TTLmr, FeatureKind::TT}, {FeatureKind::TTNmpLmr, FeatureKind::TTNmp}};

CpuSearchOptions with_table(FeatureKind f, TTLayoutKind layout, const HostTT& table,
                            BoardKind board = BoardKind::Quad, OrderingKind ordering = OrderingKind::Staged) {
    CpuSearchOptions o{board, EvalKind::Pesto, ordering};
    o.features = f;
    o.tt_layout = layout;
    o.tt = table.desc();
    return o;
}

FixedDepthResult search(const Position& p, int depth, const CpuSearchOptions& o, const std::vector<Move>& pv = {}) {
    return cpu_search_fixed(p, GameHistory{}, depth, -kScoreInfinite, kScoreInfinite, o, pv);
}

}  // namespace

TEST_CASE(lmr_table) {
    const PruneParams d;  // 0.75 + ln(depth) ln(move) / 2.25, from depth 3, after 3 moves
    const LmrTable t = make_lmr_table(d);
    for (int depth = 0; depth < 64; ++depth)
        for (int m = 0; m < 64; ++m)
            if (depth < 3 || m <= 3) CHECK_MSG(t.r[depth][m] == 0, std::to_string(depth) + " " + std::to_string(m));
    CHECK_EQ(t.reduction(3, 4), 1);    // 0.75 + 1.10 * 1.39 / 2.25 = 1.43, and at most depth - 2
    CHECK_EQ(t.reduction(8, 10), 2);   // 0.75 + 2.08 * 2.30 / 2.25 = 2.88
    CHECK_EQ(t.reduction(8, 30), 3);   // 0.75 + 2.08 * 3.40 / 2.25 = 3.89
    CHECK_EQ(t.reduction(20, 63), 6);  // 0.75 + 3.00 * 4.14 / 2.25 = 6.27
    CHECK_EQ(t.reduction(3, 63), 1);   // capped at depth - 2
    CHECK_EQ(t.reduction(0, 30), 0);   // quiescence
    CHECK_EQ(t.reduction(-5, 30), 0);
    CHECK_EQ(t.reduction(99, 99), t.reduction(63, 63));  // both capped at 63
    for (int depth = 1; depth < 64; ++depth)
        for (int m = 1; m < 64; ++m) {
            CHECK(t.r[depth][m] <= std::max(0, depth - 2));
            if (m > 1) CHECK(t.r[depth][m] >= t.r[depth][m - 1]);  // later moves: no smaller reduction
        }
    PruneParams none = d;
    none.lmr_divisor = 0;  // invalid: no reductions at all
    const LmrTable z = make_lmr_table(none);
    for (int depth = 0; depth < 64; ++depth)
        for (int m = 0; m < 64; ++m) CHECK(z.r[depth][m] == 0);
}

TEST_CASE(lmr_settings) {
    CpuSearchOptions o{BoardKind::Quad, EvalKind::Pesto, OrderingKind::Staged};
    o.features = FeatureKind::TTNmpLmr;
    CHECK_EQ(describe(o), std::string("quad/pesto/staged tt-nmp-lmr wide128 null 3+d/6 from 2 lmr 75/225 from 3 after 3"));
    o.features = FeatureKind::TTLmr;
    CHECK_EQ(describe(o), std::string("quad/pesto/staged tt-lmr wide128 lmr 75/225 from 3 after 3"));
    const auto rejects = [](u8 min_depth, u8 moves, u8 divisor) {
        PruneParams p;
        p.lmr_min_depth = min_depth;
        p.lmr_moves = moves;
        p.lmr_divisor = divisor;
        try {
            validate_prune(p);
        } catch (const std::invalid_argument&) {
            return true;
        }
        return false;
    };
    CHECK(!rejects(3, 3, 225));
    CHECK(!rejects(1, 0, 1));
    CHECK(!rejects(20, 63, 255));
    CHECK(rejects(0, 3, 225));
    CHECK(rejects(21, 3, 225));
    CHECK(rejects(3, 64, 225));
    CHECK(rejects(3, 3, 0));
}

TEST_CASE(lmr_search_matches_recursive_quad_pesto) {
    check_against_recursive<QuadBoard, PestoEval, FeaturesLmrOnly, false>(PruneParams{});
}

TEST_CASE(lmr_search_matches_recursive_std_material) {
    check_against_recursive<StdBoard, MaterialEval, FeaturesLmrOnly, false>(PruneParams{});
}

TEST_CASE(lmr_with_null_moves_matches_recursive) {
    check_against_recursive<QuadBoard, PestoEval, FeaturesNullLmr, true>(PruneParams{});
}

TEST_CASE(lmr_search_matches_recursive_settings) {
    PruneParams steep;  // R = 1 + ln ln / 1.5, from depth 2, after the first move
    steep.lmr_min_depth = 2;
    steep.lmr_moves = 1;
    steep.lmr_base = 100;
    steep.lmr_divisor = 150;
    check_against_recursive<QuadBoard, PestoEval, FeaturesNullLmr, true>(steep);
    PruneParams mild;  // R = 0.5 + ln ln / 3, from depth 4, after 6 moves
    mild.lmr_min_depth = 4;
    mild.lmr_moves = 6;
    mild.lmr_base = 50;
    mild.lmr_divisor = 255;
    check_against_recursive<QuadBoard, PestoEval, FeaturesLmrOnly, false>(mild);
    PruneParams first;  // even the first move may be reduced
    first.lmr_moves = 0;
    first.lmr_base = 200;
    check_against_recursive<QuadBoard, PestoEval, FeaturesLmrOnly, false>(first, kRefDepth - 1);
}

TEST_CASE(lmr_exact_when_nothing_is_reduced) {
    // With reductions out of reach, the search without them, each search
    // from an empty table.
    HostTT table(16);
    PruneParams never;
    never.lmr_min_depth = 20;
    for (const LmrSet& set : kSets) {
        for (const TTLayoutKind layout : kLayouts) {
            for (const std::string& fen : positions()) {
                const Position p = testutil::parse_or_die(fen);
                std::vector<Move> pv;
                for (int depth = 1; depth <= kNodesDepth - 1; ++depth) {
                    table.clear();
                    const FixedDepthResult base = search(p, depth, with_table(set.without, layout, table), pv);
                    table.clear();
                    CpuSearchOptions o = with_table(set.with, layout, table);
                    o.prune = never;
                    const FixedDepthResult lmr = search(p, depth, o, pv);
                    CHECK_MSG(lmr.search.score == base.search.score && lmr.search.nodes == base.search.nodes &&
                                  lmr.search.best == base.search.best,
                              fen + " depth " + std::to_string(depth) + " " + name_of(set.with) + " " +
                                  name_of(layout));
                    pv = base.pv;
                }
            }
        }
    }
}

TEST_CASE(lmr_prunes) {
    // Iterative deepening on one table per position, as the engine searches.
    HostTT table(64);
    for (const LmrSet& set : kSets) {
        u64 with_lmr = 0, without = 0;
        for (const std::string& fen : positions()) {
            const Position p = testutil::parse_or_die(fen);
            for (const FeatureKind f : {set.without, set.with}) {
                table.clear();
                std::vector<Move> pv;
                u64 total = 0;
                for (int depth = 1; depth <= kNodesDepth; ++depth) {
                    table.new_search();
                    const FixedDepthResult r = search(p, depth, with_table(f, TTLayoutKind::Wide128, table), pv);
                    total += r.search.nodes;
                    pv = r.pv;
                }
                (f == set.without ? without : with_lmr) += total;
            }
        }
        CHECK_MSG(with_lmr < without, std::string(name_of(set.with)) + ": " + std::to_string(with_lmr) +
                                          " nodes with reductions vs " + std::to_string(without));
    }
}

TEST_CASE(lmr_mates_keep_their_distance) {
    // A mate the plain search finds at `depth` is found with the same
    // distance a few plies deeper with reductions (a reduced move that
    // matters gets searched again).
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
    HostTT table(16);
    for (const Case& c : cases) {
        const Position p = testutil::parse_or_die(c.fen);
        const int mate = search(p, c.depth, {BoardKind::Quad, EvalKind::Pesto, OrderingKind::Staged}).search.score;
        REQUIRE(is_mate_score(mate));
        for (const LmrSet& set : kSets) {
            for (const TTLayoutKind layout : kLayouts) {
                table.clear();
                std::vector<Move> pv;
                int score = 0;
                for (int depth = 1; depth <= c.depth + 3; ++depth) {
                    table.new_search();
                    const FixedDepthResult r = search(p, depth, with_table(set.with, layout, table), pv);
                    score = r.search.score;
                    pv = r.pv;
                }
                CHECK_MSG(score == mate, std::string(c.fen) + " " + name_of(set.with) + " " + name_of(layout) +
                                             ": " + std::to_string(score) + " vs " + std::to_string(mate));
            }
        }
    }
}

TEST_CASE(lmr_variants_agree) {
    // Every board and ordering: the same tree with reductions and a table.
    HostTT table(16);
    for (const LmrSet& set : kSets) {
        for (const std::string& fen : positions()) {
            const Position p = testutil::parse_or_die(fen);
            table.clear();
            const FixedDepthResult first =
                search(p, kNodesDepth - 1, with_table(set.with, TTLayoutKind::Wide128, table));
            for (const BoardKind board : {BoardKind::Quad, BoardKind::Std}) {
                for (const OrderingKind ordering : {OrderingKind::Staged, OrderingKind::List}) {
                    table.clear();
                    const FixedDepthResult r = search(
                        p, kNodesDepth - 1, with_table(set.with, TTLayoutKind::Wide128, table, board, ordering));
                    CHECK_MSG(r.search.score == first.search.score && r.search.nodes == first.search.nodes &&
                                  r.search.best == first.search.best && r.pv == first.pv,
                              fen + " " + name_of(set.with) + " " + name_of(board) + "/" + name_of(ordering));
                }
            }
        }
    }
}

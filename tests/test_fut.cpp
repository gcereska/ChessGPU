// Futility pruning (core/search.h; feature sets tt-fut and tt-lmr-fut): the
// bound, and the search with it.
//
// What the search must guarantee:
// - its pruning matches a recursive implementation exactly (score and node
//   count; without a table, through test-only feature sets), alone and with
//   late move reductions, with every setting and window;
// - mates keep their exact distance; fewer nodes to the same depth; every
//   board and ordering visits the same tree.
//
// Of 13 bugs planted one at a time in the futility code, these tests catch
// all but two that can't change a search: counting futile moves for the
// reductions (where one quiet move is futile all are, except checks, which
// are never reduced), and letting a re-search be pruned (a reduced move
// passed the same test with the same alpha).

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
constexpr int kRefDepth = 5;  // futile moves cost a gives-check test each in the reference: slower than reductions
constexpr int kNodesDepth = 7;
#else
constexpr int kRefDepth = 4;
constexpr int kNodesDepth = 5;
#endif

// Without a table, so the flat search can be compared with the recursive
// reference (tests/recursive_search_ref.h).
struct FeaturesFutOnly {
    static constexpr const char* kName = "fut-only";
    static constexpr bool kTT = false;
    static constexpr bool kTTCutoffs = false;
    static constexpr bool kTTQuiescence = false;
    static constexpr bool kNullMove = false;
    static constexpr bool kLmr = false;
    static constexpr bool kFutility = true;
    static constexpr bool kCheckExt = false;
    static constexpr bool kKillers = false;
    static constexpr bool kQPrune = false;
};

struct FeaturesLmrFut {
    static constexpr const char* kName = "lmr-fut";
    static constexpr bool kTT = false;
    static constexpr bool kTTCutoffs = false;
    static constexpr bool kTTQuiescence = false;
    static constexpr bool kNullMove = false;
    static constexpr bool kLmr = true;
    static constexpr bool kFutility = true;
    static constexpr bool kCheckExt = false;
    static constexpr bool kKillers = false;
    static constexpr bool kQPrune = false;
};

std::vector<std::string> positions() {
    std::vector<std::string> v = {
        "r1bq1rk1/pp2bppp/2n1pn2/3p4/2PP4/2N1PN2/PP3PPP/R2QKB1R w KQ - 0 8",
        "2r2rk1/1bqnbppp/pp1ppn2/8/2PNP3/1PN1BP2/P2Q2PP/2RR1BK1 w - - 0 15",
        "3qk3/8/8/8/8/8/8/R2QK2R b KQ - 0 1",       // checks
        "8/5k2/3p4/1p1P4/1P6/5K2/8/8 w - - 0 1",     // kings and pawns
        "r3k3/8/8/8/8/8/8/4K2R w Kq - 0 1",          // rook endgame: few moves, quiet ones futile when behind
    };
    for (const std::string& fen : testutil::test_fens()) v.push_back(fen);
    return v;
}

std::string settings_text(const PruneParams& p) {
    return "fut " + std::to_string(p.fut_margin) + " to " + std::to_string(p.fut_depth);
}

// The flat search against the reference on every position, in iterative
// deepening, with the full window and with zero windows around the score
// and far from it (where much is futile).
template <class Board, class Eval, class Features, bool kLmr>
void check_against_recursive(const PruneParams& prune) {
    u64 futile = 0;
    for (const std::string& fen : positions()) {
        const Position p = testutil::parse_or_die(fen);
        std::vector<Move> pv;
        for (int depth = 1; depth <= kRefDepth; ++depth) {
            const testref::Outcome flat = testref::flat_search<Board, Eval, Features>(
                p, depth, -kScoreInfinite, kScoreInfinite, 0, kTaskOnPv, prune, pv);
            auto rec = testref::reference<Board, Eval, false, kLmr, true>(p, prune, pv);
            const int score = rec.search(depth, -kScoreInfinite, kScoreInfinite, 0, 0, true, false);
            const std::string where = fen + " depth " + std::to_string(depth) + " " + settings_text(prune);
            CHECK_MSG(flat.score == score && flat.nodes == rec.nodes,
                      where + ": " + std::to_string(flat.score) + "/" + std::to_string(flat.nodes) + " vs " +
                          std::to_string(score) + "/" + std::to_string(rec.nodes));
            futile += rec.futile;
            for (const int alpha : {score - 1, score, score + 300}) {
                if (alpha <= -kScoreInfinite || alpha + 1 >= kScoreInfinite) continue;
                const testref::Outcome zero =
                    testref::flat_search<Board, Eval, Features>(p, depth, alpha, alpha + 1, 0, 0, prune, pv);
                auto zrec = testref::reference<Board, Eval, false, kLmr, true>(p, prune, pv);
                const int zscore = zrec.search(depth, alpha, alpha + 1, 0, 0, false, false);
                CHECK_MSG(zero.score == zscore && zero.nodes == zrec.nodes,
                          where + " window " + std::to_string(alpha) + ": " + std::to_string(zero.score) + "/" +
                              std::to_string(zero.nodes) + " vs " + std::to_string(zscore) + "/" +
                              std::to_string(zrec.nodes));
                futile += zrec.futile;
            }
            pv = flat.pv;
        }
    }
    CHECK_MSG(futile > 0, "no move was futile");
}

const TTLayoutKind kLayouts[] = {TTLayoutKind::Packed64, TTLayoutKind::Wide128};

struct FutSet {
    FeatureKind with;
    FeatureKind without;
};
const FutSet kSets[] = {{FeatureKind::TTFut, FeatureKind::TT}, {FeatureKind::TTLmrFut, FeatureKind::TTLmr}};

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

TEST_CASE(fut_bound) {
    const PruneParams d;  // 150 per ply, up to 3 plies
    CHECK_EQ(futility_bound(d, 20, 1, 0, 100), 170);
    CHECK_EQ(futility_bound(d, 20, 3, 0, 100), 470);
    CHECK_EQ(futility_bound(d, 20, 4, 0, 100), kScoreInfinite);  // too many plies left
    CHECK_EQ(futility_bound(d, 20, 0, 0, 100), kScoreInfinite);  // quiescence
    CHECK_EQ(futility_bound(d, 20, 2, level_flags::kInCheck, 100), kScoreInfinite);
    CHECK_EQ(futility_bound(d, 20, 2, level_flags::kOnPv, 100), kScoreInfinite);
    CHECK_EQ(futility_bound(d, 20, 2, level_flags::kStore | level_flags::kAfterNull, 100), 320);  // other flags: no matter
    CHECK_EQ(futility_bound(d, 20, 2, 0, mate_in(5)), kScoreInfinite);    // alpha a mate
    CHECK_EQ(futility_bound(d, 20, 2, 0, mated_in(5)), kScoreInfinite);
    CHECK_EQ(futility_bound(d, 20, 2, 0, kScoreMateBound), 320);          // the last score that isn't a mate
    PruneParams wide = d;
    wide.fut_depth = 5;
    wide.fut_margin = 0;
    CHECK_EQ(futility_bound(wide, -40, 5, 0, 100), -40);
}

TEST_CASE(fut_settings) {
    CpuSearchOptions o{BoardKind::Quad, EvalKind::Pesto, OrderingKind::Staged};
    o.features = FeatureKind::TTFut;
    CHECK_EQ(describe(o), std::string("quad/pesto/staged tt-fut wide128 fut 150 to 3"));
    o.features = FeatureKind::TTLmrFut;
    CHECK_EQ(describe(o), std::string("quad/pesto/staged tt-lmr-fut wide128 lmr 75/225 from 3 after 3 fut 150 to 3"));
    const auto rejects = [](u8 depth) {
        PruneParams p;
        p.fut_depth = depth;
        try {
            validate_prune(p);
        } catch (const std::invalid_argument&) {
            return true;
        }
        return false;
    };
    CHECK(!rejects(1));
    CHECK(!rejects(10));
    CHECK(rejects(0));
    CHECK(rejects(11));
}

TEST_CASE(fut_search_matches_recursive_quad_pesto) {
    check_against_recursive<QuadBoard, PestoEval, FeaturesFutOnly, false>(PruneParams{});
}

TEST_CASE(fut_search_matches_recursive_std_material) {
    check_against_recursive<StdBoard, MaterialEval, FeaturesFutOnly, false>(PruneParams{});
}

TEST_CASE(fut_with_reductions_matches_recursive) {
    check_against_recursive<QuadBoard, PestoEval, FeaturesLmrFut, true>(PruneParams{});
}

TEST_CASE(fut_search_matches_recursive_settings) {
    PruneParams tight;  // a small margin, deep
    tight.fut_depth = 5;
    tight.fut_margin = 40;
    check_against_recursive<QuadBoard, PestoEval, FeaturesFutOnly, false>(tight);
    PruneParams frontier;  // the last ply only, a big margin
    frontier.fut_depth = 1;
    frontier.fut_margin = 255;
    check_against_recursive<QuadBoard, PestoEval, FeaturesLmrFut, true>(frontier);
}

TEST_CASE(fut_prunes) {
    // Iterative deepening on one table per position, as the engine searches.
    HostTT table(64);
    for (const FutSet& set : kSets) {
        u64 with_fut = 0, without = 0;
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
                (f == set.without ? without : with_fut) += total;
            }
        }
        CHECK_MSG(with_fut < without, std::string(name_of(set.with)) + ": " + std::to_string(with_fut) +
                                          " nodes with futility pruning vs " + std::to_string(without));
    }
}

TEST_CASE(fut_mates_keep_their_distance) {
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
        for (const FutSet& set : kSets) {
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

TEST_CASE(fut_variants_agree) {
    HostTT table(16);
    for (const FutSet& set : kSets) {
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

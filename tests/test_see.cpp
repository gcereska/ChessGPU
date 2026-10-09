// Static exchange evaluation and quiescence pruning (core/see.h,
// core/search.h; feature sets tt-killers-see and tt-ext-killers-see).
//
// What must hold:
// - see() equals the exchange evaluated recursively on the reference board
//   (tests/reference_see.h) for every capture and promotion in the test
//   trees, on both boards, and gives the known values in hand-checked
//   examples (x-rays, a king that can't recapture, en passant, promotions);
// - material_gain() is the victim plus a promotion's gain;
// - the search with delta and SEE pruning matches the recursive reference
//   exactly (score and node count; no table, through test-only feature
//   sets), at several settings, alone and with every other technique, with
//   zero windows;
// - quiescence pruning saves nodes; mates keep their distance; every board
//   and ordering visits the same tree; settings are described and checked.
//
// Of 15 bugs planted on the CPU side (in see(), material_gain(), the pruning
// rules and where the search applies them), 13 are caught. The 2 others
// remove or weaken the rule that a king can't capture onto a defended
// square, which can't change any result: the king is worth 20000, so the
// exchange never continues into a king capture that loses it.

#include <stdexcept>
#include <string>
#include <vector>

#include "core/see.h"
#include "host/fen.h"
#include "host/search_cpu.h"
#include "host/tt_storage.h"
#include "host/uci_move.h"
#include "recursive_search_ref.h"
#include "reference_see.h"
#include "test_harness.h"
#include "test_util.h"

using namespace cb;
using testref::attacks;

namespace {

int see_of(const char* fen, const char* uci) {
    const Position p = testutil::parse_or_die(fen);
    const QuadBoard b = to_board<QuadBoard>(p);
    const Move m = parse_uci_move(b, attacks(), uci);
    CHECK_MSG(!m.is_none(), std::string(fen) + " " + uci + ": not a legal move");
    if (m.is_none()) return kScoreInfinite;
    const int v = see(b, attacks(), m);
    CHECK_MSG(v == refsee::see(p, m), std::string(fen) + " " + uci + ": " + std::to_string(v) + " vs reference " +
                                          std::to_string(refsee::see(p, m)));
    return v;
}

}  // namespace

TEST_CASE(see_examples) {
    // An undefended pawn.
    CHECK_EQ(see_of("1k1r4/1pp4p/p7/4p3/8/P5P1/1PP4P/2K1R3 w - - 0 1", "e1e5"), 100);
    // Knight takes pawn into N, B and an x-rayed queen, against R and an
    // x-rayed queen: black's knight recaptures, white stops after its rook
    // would be taken: 100 - 320.
    CHECK_EQ(see_of("1k1r3q/1ppn3p/p4b2/4p3/8/P2N2P1/1PP1R1BP/2K1Q3 w - - 0 1", "d3e5"), -220);
    // The king may recapture only an undefended pawn.
    CHECK_EQ(see_of("8/8/8/3k4/4p3/3P4/8/6K1 w - - 0 1", "d3e4"), 0);
    CHECK_EQ(see_of("8/8/8/3k4/4p3/3P4/8/4R1K1 w - - 0 1", "d3e4"), 100);
    // En passant takes a pawn.
    CHECK_EQ(see_of("4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 2", "e5d6"), 100);
    // Promotions: the gain, then maybe the new queen lost.
    CHECK_EQ(see_of("4k3/1P6/8/8/8/8/8/4K3 w - - 0 1", "b7b8q"), 800);
    CHECK_EQ(see_of("r3k3/1P6/8/8/8/8/8/4K3 w - - 0 1", "b7b8q"), -100);
    CHECK_EQ(see_of("r3k3/1P6/8/8/8/8/8/4K3 w - - 0 1", "b7a8q"), 1300);
    // A queen taking a defended pawn loses the queen.
    CHECK_EQ(see_of("4k3/8/2p5/3p4/8/8/8/3QK3 w - - 0 1", "d1d5"), -800);
    // Batteries: two rooks against one defender.
    CHECK_EQ(see_of("3rk3/8/8/3p4/8/8/3R4/3RK3 w - - 0 1", "d2d5"), 100);
}

TEST_CASE(see_matches_reference) {
    // Every capture and promotion at every node of the test trees.
    long long checked = 0, nonzero_losses = 0;
    for (const std::string& fen : testutil::test_fens()) {
        long long budget = testutil::kTreeBudget / 4;
        auto visit = [&](const Position& p, const std::vector<Move>& moves) {
            const QuadBoard qb = to_board<QuadBoard>(p);
            const StdBoard sb = to_board<StdBoard>(p);
            for (const Move m : moves) {
                if (!m.is_capture() && !m.is_promotion()) continue;
                const int expected = refsee::see(p, m);
                const int q = see(qb, attacks(), m);
                const int s = see(sb, attacks(), m);
                if (q != expected || s != expected) {
                    CHECK_MSG(false, format_fen(p) + " " + move_to_uci(m) + ": quad " + std::to_string(q) + ", std " +
                                         std::to_string(s) + ", reference " + std::to_string(expected));
                }
                const int gain = material_gain(qb, m);
                CHECK_MSG(q <= gain, format_fen(p) + " " + move_to_uci(m) + ": more than the material taken");
                if (q < 0) ++nonzero_losses;
                ++checked;
            }
        };
        testutil::walk_tree(testutil::parse_or_die(fen), 3, budget, visit);
    }
    CHECK_MSG(checked > 20000 && nonzero_losses > 1000,
              std::to_string(checked) + " captures, " + std::to_string(nonzero_losses) + " losing");
}

TEST_CASE(see_material_gain) {
    const Position p = testutil::parse_or_die("r3k2r/pPpp1ppp/8/3Pp3/8/8/PPP2PPP/R3K2R w KQkq e6 0 1");
    const QuadBoard b = to_board<QuadBoard>(p);
    CHECK_EQ(material_gain(b, parse_uci_move(b, attacks(), "d5e6")), 100);   // en passant
    CHECK_EQ(material_gain(b, parse_uci_move(b, attacks(), "b7a8q")), 1300);  // rook and a queen for the pawn
    CHECK_EQ(material_gain(b, parse_uci_move(b, attacks(), "b7b8q")), 800);
    CHECK_EQ(material_gain(b, parse_uci_move(b, attacks(), "b7b8n")), 220);
    CHECK_EQ(material_gain(b, parse_uci_move(b, attacks(), "a2a3")), 0);
}

// ---------------------------------------------------------------------------
// Quiescence pruning in the search
// ---------------------------------------------------------------------------

namespace {

#ifdef NDEBUG
constexpr int kRefDepth = 5;
constexpr int kNodesDepth = 6;
#else
constexpr int kRefDepth = 3;
constexpr int kNodesDepth = 4;
#endif

struct FeaturesSeeOnly {
    static constexpr const char* kName = "see-only";
    static constexpr bool kTT = false;
    static constexpr bool kTTCutoffs = false;
    static constexpr bool kTTQuiescence = false;
    static constexpr bool kNullMove = false;
    static constexpr bool kLmr = false;
    static constexpr bool kFutility = false;
    static constexpr bool kCheckExt = false;
    static constexpr bool kKillers = false;
    static constexpr bool kQPrune = true;
};

// Every technique at once, no table.
struct FeaturesEverything {
    static constexpr const char* kName = "everything";
    static constexpr bool kTT = false;
    static constexpr bool kTTCutoffs = false;
    static constexpr bool kTTQuiescence = false;
    static constexpr bool kNullMove = true;
    static constexpr bool kLmr = true;
    static constexpr bool kFutility = true;
    static constexpr bool kCheckExt = true;
    static constexpr bool kKillers = true;
    static constexpr bool kQPrune = true;
};

std::vector<std::string> search_positions() {
    std::vector<std::string> v = {
        "r1bq1rk1/pp2bppp/2n1pn2/3p4/2PP4/2N1PN2/PP3PPP/R2QKB1R w KQ - 0 8",
        "2r2rk1/1bqnbppp/pp1ppn2/8/2PNP3/1PN1BP2/P2Q2PP/2RR1BK1 w - - 0 15",
        "1k1r3q/1ppn3p/p4b2/4p3/8/P2N2P1/1PP1R1BP/2K1Q3 w - - 0 1",  // a losing capture (SEE -220)
        "4r3/2pk4/3n1B1Q/pp6/1P4qP/2P3P1/1P4K1/R7 w - - 3 47",
    };
    for (const std::string& fen : testutil::test_fens()) v.push_back(fen);
    return v;
}

std::string describe_settings(const PruneParams& p) {
    return "delta " + std::to_string(p.delta_margin) + (p.see_prune ? " see" : " no see");
}

template <class Board, class Features, bool kNull, bool kLmr, bool kFut, bool kExt, bool kKillers>
void check_against_recursive(const PruneParams& prune, int max_depth = kRefDepth) {
    u64 delta = 0, losing = 0;
    for (const std::string& fen : search_positions()) {
        const Position p = testutil::parse_or_die(fen);
        std::vector<Move> pv;
        for (int depth = 1; depth <= max_depth; ++depth) {
            const testref::Outcome flat = testref::flat_search<Board, PestoEval, Features>(
                p, depth, -kScoreInfinite, kScoreInfinite, 0, kTaskOnPv, prune, pv);
            auto rec = testref::reference<Board, PestoEval, kNull, kLmr, kFut, kExt, kKillers, true>(p, prune, pv);
            const int score = rec.search(depth, -kScoreInfinite, kScoreInfinite, 0, 0, true, false);
            const std::string where =
                fen + " depth " + std::to_string(depth) + " " + Features::kName + " " + describe_settings(prune);
            CHECK_MSG(flat.score == score && flat.nodes == rec.nodes,
                      where + ": " + std::to_string(flat.score) + "/" + std::to_string(flat.nodes) + " vs " +
                          std::to_string(score) + "/" + std::to_string(rec.nodes));
            delta += rec.delta_pruned;
            losing += rec.see_pruned;
            for (const int alpha : {score - 101, score - 1, score, score + 100}) {
                if (alpha <= -kScoreInfinite || alpha + 1 >= kScoreInfinite) continue;
                const testref::Outcome zero = testref::flat_search<Board, PestoEval, Features>(
                    p, depth, alpha, alpha + 1, 0, 0, prune, pv);
                auto zrec =
                    testref::reference<Board, PestoEval, kNull, kLmr, kFut, kExt, kKillers, true>(p, prune, pv);
                const int zscore = zrec.search(depth, alpha, alpha + 1, 0, 0, false, false);
                CHECK_MSG(zero.score == zscore && zero.nodes == zrec.nodes,
                          where + " window " + std::to_string(alpha) + ": " + std::to_string(zero.score) + "/" +
                              std::to_string(zero.nodes) + " vs " + std::to_string(zscore) + "/" +
                              std::to_string(zrec.nodes));
            }
            pv = flat.pv;
        }
    }
    if (prune.delta_margin > 0) CHECK_MSG(delta > 0, "no move was delta pruned (" + describe_settings(prune) + ")");
    if (prune.see_prune) CHECK_MSG(losing > 0, "no move was SEE pruned (" + describe_settings(prune) + ")");
}

CpuSearchOptions with_table(FeatureKind f, const HostTT& table, const PruneParams& prune = {},
                            BoardKind board = BoardKind::Quad, OrderingKind ordering = OrderingKind::Staged) {
    CpuSearchOptions o{board, EvalKind::Pesto, ordering};
    o.features = f;
    o.prune = prune;
    o.tt = table.desc();
    return o;
}

FixedDepthResult search_fixed(const Position& p, int depth, const CpuSearchOptions& o,
                              const std::vector<Move>& pv = {}) {
    return cpu_search_fixed(p, GameHistory{}, depth, -kScoreInfinite, kScoreInfinite, o, pv);
}

struct SeeSet {
    FeatureKind with;
    FeatureKind without;
};
const SeeSet kSeeSets[] = {{FeatureKind::TTKillersSee, FeatureKind::TTKillers},
                           {FeatureKind::TTExtKillersSee, FeatureKind::TTExtKillers}};

}  // namespace

TEST_CASE(see_search_matches_recursive) {
    check_against_recursive<QuadBoard, FeaturesSeeOnly, false, false, false, false, false>(PruneParams{});
    check_against_recursive<StdBoard, FeaturesSeeOnly, false, false, false, false, false>(PruneParams{});
}

TEST_CASE(see_search_settings_match_recursive) {
    // Delta pruning alone (the default, a tight and a loose margin), SEE pruning alone.
    PruneParams delta_only;
    delta_only.see_prune = 0;
    PruneParams tight = delta_only;
    tight.delta_margin = 1;
    PruneParams loose = delta_only;
    loose.delta_margin = 900;
    PruneParams see_only;
    see_only.delta_margin = 0;
    for (const PruneParams& p : {delta_only, tight, loose, see_only})
        check_against_recursive<QuadBoard, FeaturesSeeOnly, false, false, false, false, false>(p, kRefDepth - 1);
}

TEST_CASE(see_with_everything_matches_recursive) {
    check_against_recursive<QuadBoard, FeaturesEverything, true, true, true, true, true>(PruneParams{});
}

TEST_CASE(see_saves_quiescence_nodes) {
    // Iterative deepening on one table per position, as the engine searches.
    HostTT table(64);
    for (const SeeSet& set : kSeeSets) {
        u64 with = 0, without = 0;
        for (const std::string& fen : search_positions()) {
            const Position p = testutil::parse_or_die(fen);
            for (const FeatureKind f : {set.without, set.with}) {
                table.clear();
                std::vector<Move> pv;
                u64 total = 0;
                for (int depth = 1; depth <= kNodesDepth; ++depth) {
                    table.new_search();
                    const FixedDepthResult r = search_fixed(p, depth, with_table(f, table), pv);
                    total += r.search.nodes;
                    pv = r.pv;
                }
                (f == set.without ? without : with) += total;
            }
        }
        CHECK_MSG(with < without, std::string(name_of(set.with)) + ": " + std::to_string(with) +
                                      " nodes with quiescence pruning vs " + std::to_string(without));
    }
}

TEST_CASE(see_mates_keep_their_distance) {
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
        const int mate =
            search_fixed(p, c.depth, {BoardKind::Quad, EvalKind::Pesto, OrderingKind::Staged}).search.score;
        REQUIRE(is_mate_score(mate));
        for (const SeeSet& set : kSeeSets) {
            table.clear();
            std::vector<Move> pv;
            int score = 0;
            for (int depth = 1; depth <= c.depth + 2; ++depth) {
                table.new_search();
                const FixedDepthResult r = search_fixed(p, depth, with_table(set.with, table), pv);
                score = r.search.score;
                pv = r.pv;
            }
            CHECK_MSG(is_mate_score(score) && score >= mate,
                      std::string(c.fen) + " " + name_of(set.with) + ": " + std::to_string(score) + " vs " +
                          std::to_string(mate));
        }
    }
}

TEST_CASE(see_variants_agree) {
    HostTT table(16);
    for (const SeeSet& set : kSeeSets) {
        for (const std::string& fen : search_positions()) {
            const Position p = testutil::parse_or_die(fen);
            table.clear();
            const FixedDepthResult first = search_fixed(p, kRefDepth, with_table(set.with, table));
            for (const BoardKind board : {BoardKind::Quad, BoardKind::Std}) {
                for (const OrderingKind ordering : {OrderingKind::Staged, OrderingKind::List}) {
                    table.clear();
                    const FixedDepthResult r =
                        search_fixed(p, kRefDepth, with_table(set.with, table, PruneParams{}, board, ordering));
                    CHECK_MSG(r.search.score == first.search.score && r.search.nodes == first.search.nodes &&
                                  r.search.best == first.search.best && r.pv == first.pv,
                              fen + " " + name_of(set.with) + " " + name_of(board) + "/" + name_of(ordering));
                }
            }
        }
    }
}

TEST_CASE(see_settings) {
    CpuSearchOptions o{BoardKind::Quad, EvalKind::Pesto, OrderingKind::Staged};
    o.features = FeatureKind::TTKillersSee;
    CHECK_EQ(describe(o), std::string("quad/pesto/staged tt-killers-see wide128 delta 200 see"));
    o.prune.delta_margin = 0;
    o.prune.see_prune = 0;
    CHECK_EQ(describe(o), std::string("quad/pesto/staged tt-killers-see wide128 no delta no see"));
    CHECK(uses_qprune(FeatureKind::TTExtKillersSee) && uses_killers(FeatureKind::TTExtKillersSee) &&
          uses_check_extensions(FeatureKind::TTExtKillersSee));
    CHECK(!uses_qprune(FeatureKind::TTKillers) && !uses_check_extensions(FeatureKind::TTKillersSee));
    const auto rejects = [](int margin, int see) {
        PruneParams p;
        p.delta_margin = u16(margin);
        p.see_prune = u8(see);
        try {
            validate_prune(p);
        } catch (const std::invalid_argument&) {
            return true;
        }
        return false;
    };
    CHECK(!rejects(0, 0) && !rejects(kMaxDeltaMargin, 1));
    CHECK(rejects(kMaxDeltaMargin + 1, 1) && rejects(200, 2));
}

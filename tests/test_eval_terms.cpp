// Hand-written evaluation terms (core/eval_terms.h; eval "terms").
//
// What must hold:
// - each term counts what it says in hand-made positions;
// - flipping a position's colors negates every count and keeps the score of
//   the side to move (so neither color is favored), at every node of the
//   test trees, on both boards;
// - the evaluation is PeSTO's tapered score with the weighted counts added
//   to its middlegame and endgame parts;
// - the search with it matches the recursive reference exactly.

#include <string>
#include <vector>

#include "core/eval_terms.h"
#include "core/search.h"
#include "host/fen.h"
#include "host/host_tables.h"
#include "recursive_search_ref.h"
#include "test_harness.h"
#include "test_util.h"

using namespace cb;

namespace {

TermCounts counts_of(const char* fen) { return count_terms(to_board<QuadBoard>(testutil::parse_or_die(fen))); }

int terms_eval(const Position& p) {
    const QuadBoard b = to_board<QuadBoard>(p);
    return evaluate_position<PestoTermsEval>(b, compute_accumulator<PestoTermsEval>(b, host_search_tables().eval),
                                             state::side(p.state));
}

std::string counts_text(const TermCounts& t) {
    std::string s;
    for (int i = 0; i < kTermCount; ++i) s += std::to_string(t.v[i]) + " ";
    return s;
}

}  // namespace

TEST_CASE(terms_hand_made_counts) {
    // Symmetric: every count cancels.
    const TermCounts start = counts_of("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
    for (int i = 0; i < kTermCount; ++i) CHECK_MSG(start.v[i] == 0, "term " + std::to_string(i));

    // Two connected pawns on the second rank, both passed; no shield.
    TermCounts t = counts_of("4k3/8/8/8/8/8/PP6/4K3 w - - 0 1");
    CHECK_EQ(t.v[kTermPassedPawn2], 2);
    CHECK_EQ(t.v[kTermDoubledPawn], 0);
    CHECK_EQ(t.v[kTermIsolatedPawn], 0);
    CHECK_EQ(t.v[kTermKingShield1], 0);

    // Doubled and isolated a-pawns, both passed (ranks 2 and 3).
    t = counts_of("4k3/8/8/8/8/P7/P7/4K3 w - - 0 1");
    CHECK_EQ(t.v[kTermDoubledPawn], 1);
    CHECK_EQ(t.v[kTermIsolatedPawn], 2);
    CHECK_EQ(t.v[kTermPassedPawn2], 1);
    CHECK_EQ(t.v[kTermPassedPawn3], 1);

    // A black pawn ahead on an adjacent file stops a passer; one beside it doesn't.
    t = counts_of("4k3/8/8/4p3/3P4/8/8/4K3 w - - 0 1");
    CHECK_EQ(t.v[kTermPassedPawn4], 0);  // d4 is held by e5; e5 (black's rank 4) is held by d4
    t = counts_of("4k3/8/8/3Pp3/8/8/8/4K3 w - - 0 1");
    CHECK_EQ(t.v[kTermPassedPawn5], 1);   // white d5: e5 is beside it
    CHECK_EQ(t.v[kTermPassedPawn4], -1);  // black e5 (its rank 4): d5 is beside it

    // Bishop pair against one bishop.
    t = counts_of("2b1k3/8/8/8/8/8/8/2B1KB2 w - - 0 1");
    CHECK_EQ(t.v[kTermBishopPair], 1);

    // A lone rook on a1: open file, 10 safe squares (a2-a8, b1-d1; e1 is its own king).
    t = counts_of("4k3/8/8/8/8/8/8/R3K3 w - - 0 1");
    CHECK_EQ(t.v[kTermRookOpenFile], 1);
    CHECK_EQ(t.v[kTermRookMobility], 10);
    // Semi-open: only an enemy pawn on the file.
    t = counts_of("4k3/p7/8/8/8/8/8/R3K3 w - - 0 1");
    CHECK_EQ(t.v[kTermRookSemiOpenFile], 1);
    CHECK_EQ(t.v[kTermRookOpenFile], 0);

    // Knight mobility leaves out squares enemy pawns attack: a knight on d4
    // with a black pawn on d7 (attacking c6 and e6): 8 - 2 = 6.
    t = counts_of("4k3/3p4/8/8/3N4/8/8/4K3 w - - 0 1");
    CHECK_EQ(t.v[kTermKnightMobility], 6);

    // King shield: three pawns in front, one further.
    t = counts_of("4k3/8/8/8/8/5P2/3PP3/4K3 w - - 0 1");
    CHECK_EQ(t.v[kTermKingShield1], 2);
    CHECK_EQ(t.v[kTermKingShield2], 1);
}

TEST_CASE(terms_color_flip) {
    long long nodes = 0;
    for (const std::string& fen : testutil::test_fens()) {
        long long budget = testutil::kTreeBudget / 4;
        auto visit = [&](const Position& p, const std::vector<Move>&) {
            const Position m = testutil::mirror(p);
            const TermCounts a = count_terms(to_board<QuadBoard>(p));
            const TermCounts b = count_terms(to_board<QuadBoard>(m));
            const TermCounts s = count_terms(to_board<StdBoard>(p));
            // The search's attack lookups count the same mobility as the table-free fills.
            const TermCounts h = count_terms(to_board<QuadBoard>(p), HqAttacks(host_attack_tables()));
            const TermCounts g =
                count_terms(to_board<StdBoard>(p), MagicAttacks(host_attack_tables(), host_magic_tables()));
            for (int i = 0; i < kTermCount; ++i) {
                if (a.v[i] != -b.v[i] || a.v[i] != s.v[i] || a.v[i] != h.v[i] || a.v[i] != g.v[i]) {
                    CHECK_MSG(false, format_fen(p) + ": " + counts_text(a) + "| mirrored " + counts_text(b) +
                                         "| std " + counts_text(s) + "| hq " + counts_text(h) + "| magic " +
                                         counts_text(g));
                    break;
                }
            }
            CHECK_MSG(terms_eval(p) == terms_eval(m), format_fen(p));
            ++nodes;
        };
        testutil::walk_tree(testutil::parse_or_die(fen), 3, budget, visit);
    }
    CHECK(nodes > 5000);
}

TEST_CASE(terms_eval_is_pesto_plus_weighted_counts) {
    for (const std::string& fen : testutil::test_fens()) {
        const Position p = testutil::parse_or_die(fen);
        const QuadBoard b = to_board<QuadBoard>(p);
        const PestoEval::Accumulator acc = compute_accumulator<PestoEval>(b, host_search_tables().eval);
        const TermCounts t = count_terms(b);
        int mg = unpack_mg(acc.packed), eg = unpack_eg(acc.packed);
        for (int i = 0; i < kTermCount; ++i) {
            mg += term_weight(i, false) * t.v[i];
            eg += term_weight(i, true) * t.v[i];
        }
        const int phase = acc.phase < 24 ? acc.phase : 24;
        const int white = (mg * phase + eg * (24 - phase)) / 24;
        const int expected = state::side(p.state) == WHITE ? white : -white;
        CHECK_MSG(terms_eval(p) == expected, fen);
    }
}

TEST_CASE(terms_search_matches_recursive) {
    const PruneParams prune;
    for (const std::string& fen : testutil::test_fens()) {
        const Position p = testutil::parse_or_die(fen);
        std::vector<Move> pv;
        for (int depth = 1; depth <= 4; ++depth) {
            const testref::Outcome flat = testref::flat_search<QuadBoard, PestoTermsEval, FeaturesNone>(
                p, depth, -kScoreInfinite, kScoreInfinite, 0, kTaskOnPv, prune, pv);
            auto rec = testref::reference<QuadBoard, PestoTermsEval, false, false>(p, prune, pv);
            const int score = rec.search(depth, -kScoreInfinite, kScoreInfinite, 0, 0, true, false);
            CHECK_MSG(flat.score == score && flat.nodes == rec.nodes,
                      fen + " depth " + std::to_string(depth) + ": " + std::to_string(flat.score) + "/" +
                          std::to_string(flat.nodes) + " vs " + std::to_string(score) + "/" + std::to_string(rec.nodes));
            pv = flat.pv;
        }
    }
}

// Principal variation search (core/search.h, feature flag ft::kPvs; feature
// set tt-killers-see-pvs).
//
// What must hold:
// - the search with PVS matches a recursive implementation exactly (score
//   and node count, full and zero windows; without a table, through
//   test-only feature sets), alone and with every other technique, on both
//   boards, and does search moves with the zero window and again;
// - without a table PVS never changes a score (every window that holds the
//   score gives it) and searches fewer nodes;
// - the engine's set with PVS ends every search with a legal move.

#include <string>
#include <vector>

#include "core/search.h"
#include "host/fen.h"
#include "host/search_cpu.h"
#include "host/tt_storage.h"
#include "host/uci_move.h"
#include "recursive_search_ref.h"
#include "test_harness.h"
#include "test_util.h"

using namespace cb;
using testref::attacks;

namespace {

#ifdef NDEBUG
constexpr int kRefDepth = 5;
constexpr int kNodesDepth = 6;
#else
constexpr int kRefDepth = 3;
constexpr int kNodesDepth = 4;
#endif

// Test-only sets (no table), as the other techniques' tests use.
struct FeaturesPvsOnly : FeatureFlags<ft::kPvs> {
    static constexpr const char* kName = "pvs";
};
struct FeaturesPvsEverything
    : FeatureFlags<ft::kPvs | ft::kNullMove | ft::kLmr | ft::kFutility | ft::kCheckExt | ft::kKillers | ft::kQPrune> {
    static constexpr const char* kName = "pvs-everything";
};
struct FeaturesKillersSee : FeatureFlags<ft::kKillers | ft::kQPrune> {
    static constexpr const char* kName = "killers-see";
};
struct FeaturesKillersSeePvs : FeatureFlags<ft::kKillers | ft::kQPrune | ft::kPvs> {
    static constexpr const char* kName = "killers-see-pvs";
};

std::vector<std::string> positions() {
    std::vector<std::string> v = {
        "r1bq1rk1/pp2bppp/2n1pn2/3p4/2PP4/2N1PN2/PP3PPP/R2QKB1R w KQ - 0 8",
        "2r2rk1/1bqnbppp/pp1ppn2/8/2PNP3/1PN1BP2/P2Q2PP/2RR1BK1 w - - 0 15",
        "4r3/2pk4/3n1B1Q/pp6/1P4qP/2P3P1/1P4K1/R7 w - - 3 47",
    };
    for (const std::string& fen : testutil::test_fens()) v.push_back(fen);
    return v;
}

template <class Board, class Features, bool kNull, bool kLmr, bool kFut, bool kExt, bool kKillers, bool kQPrune>
void check_against_recursive() {
    const PruneParams prune;
    u64 zero = 0, again = 0;
    for (const std::string& fen : positions()) {
        const Position p = testutil::parse_or_die(fen);
        std::vector<Move> pv;
        for (int depth = 1; depth <= kRefDepth; ++depth) {
            const testref::Outcome flat = testref::flat_search<Board, PestoEval, Features>(
                p, depth, -kScoreInfinite, kScoreInfinite, 0, kTaskOnPv, prune, pv);
            auto rec = testref::reference<Board, PestoEval, kNull, kLmr, kFut, kExt, kKillers, kQPrune, true>(p, prune, pv);
            const int score = rec.search(depth, -kScoreInfinite, kScoreInfinite, 0, 0, true, false);
            const std::string where = fen + " depth " + std::to_string(depth) + " " + Features::kName;
            CHECK_MSG(flat.score == score && flat.nodes == rec.nodes,
                      where + ": " + std::to_string(flat.score) + "/" + std::to_string(flat.nodes) + " vs " +
                          std::to_string(score) + "/" + std::to_string(rec.nodes));
            zero += rec.pvs_zero;
            again += rec.pvs_researches;
            for (const int alpha : {score - 1, score}) {
                if (alpha <= -kScoreInfinite || alpha + 1 >= kScoreInfinite) continue;
                const testref::Outcome z = testref::flat_search<Board, PestoEval, Features>(p, depth, alpha, alpha + 1,
                                                                                            0, 0, prune, pv);
                auto zrec =
                    testref::reference<Board, PestoEval, kNull, kLmr, kFut, kExt, kKillers, kQPrune, true>(p, prune, pv);
                const int zscore = zrec.search(depth, alpha, alpha + 1, 0, 0, false, false);
                CHECK_MSG(z.score == zscore && z.nodes == zrec.nodes,
                          where + " window " + std::to_string(alpha) + ": " + std::to_string(z.score) + "/" +
                              std::to_string(z.nodes) + " vs " + std::to_string(zscore) + "/" +
                              std::to_string(zrec.nodes));
            }
            pv = flat.pv;
        }
    }
    CHECK_MSG(zero > 100 && again > 0, std::to_string(zero) + " zero-window searches, " + std::to_string(again) +
                                           " searched again");
}

}  // namespace

TEST_CASE(pvs_matches_recursive) {
    check_against_recursive<QuadBoard, FeaturesPvsOnly, false, false, false, false, false, false>();
    check_against_recursive<StdBoard, FeaturesPvsOnly, false, false, false, false, false, false>();
}

TEST_CASE(pvs_with_everything_matches_recursive) {
    check_against_recursive<QuadBoard, FeaturesPvsEverything, true, true, true, true, true, true>();
}

TEST_CASE(pvs_keeps_scores_and_saves_nodes) {
    // No table: both searches are exact, so the scores agree; PVS's zero
    // windows cut more.
    const PruneParams prune;
    u64 with = 0, without = 0;
    for (const std::string& fen : positions()) {
        const Position p = testutil::parse_or_die(fen);
        std::vector<Move> pv;
        for (int depth = 1; depth <= kNodesDepth; ++depth) {
            const testref::Outcome plain = testref::flat_search<QuadBoard, PestoEval, FeaturesKillersSee>(
                p, depth, -kScoreInfinite, kScoreInfinite, 0, kTaskOnPv, prune, pv);
            const testref::Outcome pvs = testref::flat_search<QuadBoard, PestoEval, FeaturesKillersSeePvs>(
                p, depth, -kScoreInfinite, kScoreInfinite, 0, kTaskOnPv, prune, pv);
            CHECK_MSG(pvs.score == plain.score,
                      fen + " depth " + std::to_string(depth) + ": " + std::to_string(pvs.score) + " vs " +
                          std::to_string(plain.score));
            if (depth == kNodesDepth) {
                with += pvs.nodes;
                without += plain.nodes;
            }
            pv = plain.pv;
        }
    }
    CHECK_MSG(with < without, std::to_string(with) + " nodes with PVS, " + std::to_string(without) + " without");
}

TEST_CASE(pvs_engine_set_plays_legal_moves) {
    HostTT table(16);
    for (const std::string& fen : positions()) {
        const Position p = testutil::parse_or_die(fen);
        CpuSearchOptions o{BoardKind::Quad, EvalKind::Terms, OrderingKind::Staged};
        o.features = FeatureKind::TTKillersSeePvs;
        table.clear();
        o.tt = table.desc();
        const FixedDepthResult r = cpu_search_fixed(p, GameHistory{}, kNodesDepth, -kScoreInfinite, kScoreInfinite, o);
        MoveList list;
        generate_legal(to_board<QuadBoard>(p), attacks(), list);
        bool legal = list.size == 0 && r.search.best.is_none();
        for (int i = 0; i < list.size; ++i) legal = legal || list.moves[i] == r.search.best;
        CHECK_MSG(legal, fen + ": " + move_to_uci(r.search.best));
    }
}

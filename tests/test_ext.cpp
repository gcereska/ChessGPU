// Check extensions (core/search.h; feature set tt-ext).
//
// What the search must guarantee:
// - its extensions match a recursive implementation exactly (score and node
//   count; without a table, through test-only feature sets), alone and with
//   every other technique, with full and zero windows;
// - it finds mates no later than the search without extensions (every line
//   is searched at least as deep), and checking mates sooner;
// - every board and ordering visits the same tree.

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
constexpr int kRefDepth = 5;
#else
constexpr int kRefDepth = 3;
#endif

// Without a table, so the flat search can be compared with the recursive
// reference (tests/recursive_search_ref.h).
struct FeaturesExtOnly {
    static constexpr const char* kName = "ext-only";
    static constexpr bool kTT = false;
    static constexpr bool kTTCutoffs = false;
    static constexpr bool kTTQuiescence = false;
    static constexpr bool kNullMove = false;
    static constexpr bool kLmr = false;
    static constexpr bool kFutility = false;
    static constexpr bool kCheckExt = true;
    static constexpr bool kKillers = false;
    static constexpr bool kQPrune = false;
};

// Every technique at once, no table: their interplay against the reference.
struct FeaturesEverything {
    static constexpr const char* kName = "everything";
    static constexpr bool kTT = false;
    static constexpr bool kTTCutoffs = false;
    static constexpr bool kTTQuiescence = false;
    static constexpr bool kNullMove = true;
    static constexpr bool kLmr = true;
    static constexpr bool kFutility = true;
    static constexpr bool kCheckExt = true;
    static constexpr bool kKillers = false;
    static constexpr bool kQPrune = false;
};

std::vector<std::string> positions() {
    std::vector<std::string> v = {
        "r1bq1rk1/pp2bppp/2n1pn2/3p4/2PP4/2N1PN2/PP3PPP/R2QKB1R w KQ - 0 8",
        "3qk3/8/8/8/8/8/8/R2QK2R b KQ - 0 1",              // many checks
        "4r3/2pk4/3n1B1Q/pp6/1P4qP/2P3P1/1P4K1/R7 w - - 3 47",  // the LMR match's checking refutation
    };
    for (const std::string& fen : testutil::test_fens()) v.push_back(fen);
    return v;
}

template <class Board, class Eval, class Features, bool kNull, bool kLmr, bool kFut>
void check_against_recursive(int max_depth = kRefDepth) {
    const PruneParams prune;
    u64 extensions = 0;
    for (const std::string& fen : positions()) {
        const Position p = testutil::parse_or_die(fen);
        std::vector<Move> pv;
        for (int depth = 1; depth <= max_depth; ++depth) {
            const testref::Outcome flat = testref::flat_search<Board, Eval, Features>(
                p, depth, -kScoreInfinite, kScoreInfinite, 0, kTaskOnPv, prune, pv);
            auto rec = testref::reference<Board, Eval, kNull, kLmr, kFut, true>(p, prune, pv);
            const int score = rec.search(depth, -kScoreInfinite, kScoreInfinite, 0, 0, true, false);
            const std::string where = fen + " depth " + std::to_string(depth) + " " + Features::kName;
            CHECK_MSG(flat.score == score && flat.nodes == rec.nodes,
                      where + ": " + std::to_string(flat.score) + "/" + std::to_string(flat.nodes) + " vs " +
                          std::to_string(score) + "/" + std::to_string(rec.nodes));
            extensions += rec.extensions;
            for (const int alpha : {score - 1, score}) {
                if (alpha <= -kScoreInfinite || alpha + 1 >= kScoreInfinite) continue;
                const testref::Outcome zero =
                    testref::flat_search<Board, Eval, Features>(p, depth, alpha, alpha + 1, 0, 0, prune, pv);
                auto zrec = testref::reference<Board, Eval, kNull, kLmr, kFut, true>(p, prune, pv);
                const int zscore = zrec.search(depth, alpha, alpha + 1, 0, 0, false, false);
                CHECK_MSG(zero.score == zscore && zero.nodes == zrec.nodes,
                          where + " window " + std::to_string(alpha) + ": " + std::to_string(zero.score) + "/" +
                              std::to_string(zero.nodes) + " vs " + std::to_string(zscore) + "/" +
                              std::to_string(zrec.nodes));
            }
            pv = flat.pv;
        }
    }
    CHECK_MSG(extensions > 0, "no move was extended");
}

// The shallowest depth (up to max_depth) at which a search without a table
// scores a mate from p, and the score; 0 if none.
template <class Features>
std::pair<int, int> first_mate(const Position& p, int max_depth) {
    std::vector<Move> pv;
    for (int depth = 1; depth <= max_depth; ++depth) {
        const testref::Outcome o = testref::flat_search<QuadBoard, PestoEval, Features>(
            p, depth, -kScoreInfinite, kScoreInfinite, 0, kTaskOnPv, PruneParams{}, pv);
        if (is_mate_score(o.score)) return {depth, o.score};
        pv = o.pv;
    }
    return {0, 0};
}

}  // namespace

TEST_CASE(ext_search_matches_recursive_quad_pesto) {
    check_against_recursive<QuadBoard, PestoEval, FeaturesExtOnly, false, false, false>();
}

TEST_CASE(ext_search_matches_recursive_std_material) {
    check_against_recursive<StdBoard, MaterialEval, FeaturesExtOnly, false, false, false>();
}

TEST_CASE(ext_with_everything_matches_recursive) {
    check_against_recursive<QuadBoard, PestoEval, FeaturesEverything, true, true, true>();
}

TEST_CASE(ext_finds_mates_no_later) {
    // Every line is searched at least as deep as without extensions, so a
    // mate shows up no later; mates delivered by a series of checks show up
    // sooner.
    const char* const fens[] = {
        "6k1/5ppp/8/8/8/8/8/R5K1 w - - 0 1",           // back-rank mate in 1
        "7k/8/6K1/8/8/8/8/6Q1 w - - 0 1",              // queen and king against king
        "6k1/5p1p/6p1/8/8/8/5PPP/3R2K1 w - - 0 1",     // back rank, quiet start
        "r5k1/5ppp/8/8/8/8/1Q3PPP/6K1 w - - 0 1",      // checks with the queen
        "k7/8/1Q6/8/8/8/8/6RK w - - 0 1",              // a mating net of checks
    };
    int sooner = 0;
    for (const char* fen : fens) {
        const Position p = testutil::parse_or_die(fen);
        const auto [plain_depth, plain_score] = first_mate<FeaturesNone>(p, kRefDepth + 2);
        const auto [ext_depth, ext_score] = first_mate<FeaturesExtOnly>(p, kRefDepth + 2);
        if (plain_depth != 0) {
            CHECK_MSG(ext_depth != 0 && ext_depth <= plain_depth && ext_score >= plain_score,
                      std::string(fen) + ": with extensions at depth " + std::to_string(ext_depth) + " (" +
                          std::to_string(ext_score) + "), without at depth " + std::to_string(plain_depth) + " (" +
                          std::to_string(plain_score) + ")");
        }
        if (ext_depth != 0 && (plain_depth == 0 || ext_depth < plain_depth)) ++sooner;
    }
    CHECK_MSG(sooner > 0, "no mate was found sooner with extensions");
}

TEST_CASE(ext_variants_agree) {
    // Every board and ordering: the same tree with extensions and a table.
    HostTT table(16);
    for (const std::string& fen : positions()) {
        const Position p = testutil::parse_or_die(fen);
        CpuSearchOptions base{BoardKind::Quad, EvalKind::Pesto, OrderingKind::Staged};
        base.features = FeatureKind::TTExt;
        base.tt = table.desc();
        table.clear();
        const FixedDepthResult first = cpu_search_fixed(p, GameHistory{}, kRefDepth, -kScoreInfinite, kScoreInfinite, base);
        for (const BoardKind board : {BoardKind::Quad, BoardKind::Std}) {
            for (const OrderingKind ordering : {OrderingKind::Staged, OrderingKind::List}) {
                CpuSearchOptions o = base;
                o.board = board;
                o.ordering = ordering;
                table.clear();
                const FixedDepthResult r =
                    cpu_search_fixed(p, GameHistory{}, kRefDepth, -kScoreInfinite, kScoreInfinite, o);
                CHECK_MSG(r.search.score == first.search.score && r.search.nodes == first.search.nodes &&
                              r.search.best == first.search.best && r.pv == first.pv,
                          fen + " " + name_of(board) + "/" + name_of(ordering));
            }
        }
    }
}

TEST_CASE(ext_settings) {
    CpuSearchOptions o{BoardKind::Quad, EvalKind::Pesto, OrderingKind::Staged};
    o.features = FeatureKind::TTExt;
    CHECK_EQ(describe(o), std::string("quad/pesto/staged tt-ext wide128"));
    CHECK(uses_check_extensions(FeatureKind::TTExt) && !uses_check_extensions(FeatureKind::TT));
}

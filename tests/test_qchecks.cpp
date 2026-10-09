// Checks in quiescence (feature flag ft::kQChecks; feature sets
// tt-killers-see-qchecks and tt-killers-see-qchecks-nmp; core/move_order.h
// PickMode::QuiescenceChecks).
//
// What must hold:
// - quiet_move_gives_check agrees with gives_check (playing the move) for
//   every quiet move that isn't castling, on both boards;
// - both pickers list the same moves in the same order in
//   QuiescenceChecks mode: the captures, en passant and queen promotions of
//   Quiescence mode, then every quiet move (not castling) that gives check,
//   and nothing else;
// - the staged picker, which only generates the quiet moves that may check
//   (to the squares a piece attacks the king from, or any move of a piece
//   that blocks one of our sliders), finds every kind of check: direct ones
//   and uncovered ones by each piece (pawn pushes single and double), for
//   both colors, against playing each move;
// - the search with quiet checks matches a recursive implementation exactly
//   (score and node count, full and zero windows; without a table), alone,
//   with killers and quiescence pruning, and with every other technique, and
//   does search quiet checks;
// - a quiescence search finds a back-rank mate by a quiet check only with
//   the flag;
// - the engine's sets with quiet checks play legal moves.

#include <memory>
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
#else
constexpr int kRefDepth = 3;
#endif

struct FeaturesQChecksOnly : FeatureFlags<ft::kQChecks> {
    static constexpr const char* kName = "qchecks";
};
struct FeaturesKillersSeeQChecks : FeatureFlags<ft::kKillers | ft::kQPrune | ft::kQChecks> {
    static constexpr const char* kName = "killers-see-qchecks";
};
struct FeaturesQChecksEverything : FeatureFlags<ft::kQChecks | ft::kNullMove | ft::kLmr | ft::kFutility |
                                                ft::kCheckExt | ft::kKillers | ft::kQPrune | ft::kPvs> {
    static constexpr const char* kName = "qchecks-everything";
};

std::vector<std::string> positions() {
    std::vector<std::string> v = {
        "r1bq1rk1/pp2bppp/2n1pn2/3p4/2PP4/2N1PN2/PP3PPP/R2QKB1R w KQ - 0 8",
        "2r2rk1/1bqnbppp/pp1ppn2/8/2PNP3/1PN1BP2/P2Q2PP/2RR1BK1 w - - 0 15",
        "4r3/2pk4/3n1B1Q/pp6/1P4qP/2P3P1/1P4K1/R7 w - - 3 47",
        "6k1/5ppp/8/8/8/8/5PPP/3R2K1 w - - 0 1",
    };
    for (const std::string& fen : testutil::test_fens()) v.push_back(fen);
    return v;
}

template <class Board>
std::vector<Move> staged_sequence(const Board& b, PickMode mode) {
    const CheckInfo ci = make_check_info(b, attacks());
    StagedPicker::State st;
    StagedPicker::init(st, b, attacks(), ci, mode, Killers{Move(E2, E4, DOUBLE_PUSH), Move(G1, F3, QUIET)});
    std::vector<Move> seq;
    for (Move m = StagedPicker::next(st, b, attacks(), ci, Move()); !m.is_none();
         m = StagedPicker::next(st, b, attacks(), ci, Move())) {
        seq.push_back(m);
        if (seq.size() > 300) break;
    }
    return seq;
}

template <class Board>
std::vector<Move> list_sequence(const Board& b, PickMode mode) {
    const auto arena = std::make_unique<MoveArena<4096>>();
    ListPicker::State st;
    ListPicker::init(st, *arena, 0, b, attacks(), mode, Move(), Killers{Move(E2, E4, DOUBLE_PUSH), Move(G1, F3, QUIET)});
    std::vector<Move> seq;
    for (Move m = ListPicker::next(st, *arena); !m.is_none(); m = ListPicker::next(st, *arena)) seq.push_back(m);
    return seq;
}

template <class Board, class Features, bool kNull, bool kLmr, bool kFut, bool kExt, bool kKillers, bool kQPrune,
          bool kPvs>
void check_against_recursive() {
    const PruneParams prune;
    u64 quiet_checks = 0;
    for (const std::string& fen : positions()) {
        const Position p = testutil::parse_or_die(fen);
        std::vector<Move> pv;
        for (int depth = 0; depth <= kRefDepth; ++depth) {
            const testref::Outcome flat = testref::flat_search<Board, PestoEval, Features>(
                p, depth, -kScoreInfinite, kScoreInfinite, 0, kTaskOnPv, prune, pv);
            auto rec = testref::reference<Board, PestoEval, kNull, kLmr, kFut, kExt, kKillers, kQPrune, kPvs, true>(
                p, prune, pv);
            const int score = rec.search(depth, -kScoreInfinite, kScoreInfinite, 0, 0, true, false);
            const std::string where = fen + " depth " + std::to_string(depth) + " " + Features::kName;
            CHECK_MSG(flat.score == score && flat.nodes == rec.nodes,
                      where + ": " + std::to_string(flat.score) + "/" + std::to_string(flat.nodes) + " vs " +
                          std::to_string(score) + "/" + std::to_string(rec.nodes));
            quiet_checks += rec.quiet_checks;
            for (const int alpha : {score - 1, score}) {
                if (alpha <= -kScoreInfinite || alpha + 1 >= kScoreInfinite) continue;
                const testref::Outcome z = testref::flat_search<Board, PestoEval, Features>(p, depth, alpha, alpha + 1,
                                                                                            0, 0, prune, pv);
                auto zrec = testref::reference<Board, PestoEval, kNull, kLmr, kFut, kExt, kKillers, kQPrune, kPvs,
                                               true>(p, prune, pv);
                const int zscore = zrec.search(depth, alpha, alpha + 1, 0, 0, false, false);
                CHECK_MSG(z.score == zscore && z.nodes == zrec.nodes,
                          where + " window " + std::to_string(alpha) + ": " + std::to_string(z.score) + "/" +
                              std::to_string(z.nodes) + " vs " + std::to_string(zscore) + "/" +
                              std::to_string(zrec.nodes));
            }
            pv = flat.pv;
        }
    }
    CHECK_MSG(quiet_checks > 100, std::to_string(quiet_checks) + " quiet checks searched");
}

}  // namespace

TEST_CASE(qchecks_pickers_agree) {
    long long nodes = 0, checks = 0;
    for (const std::string& fen : testutil::test_fens()) {
        long long budget = testutil::kTreeBudget / 4;
        auto visit = [&](const Position& p, const std::vector<Move>&) {
            const QuadBoard q = to_board<QuadBoard>(p);
            if (make_check_info(q, attacks()).in_check()) return;
            ++nodes;
            const std::vector<Move> staged = staged_sequence(q, PickMode::QuiescenceChecks);
            const std::vector<Move> staged_std = staged_sequence(to_board<StdBoard>(p), PickMode::QuiescenceChecks);
            const std::vector<Move> listed = list_sequence(q, PickMode::QuiescenceChecks);
            const std::string where = format_fen(p);
            CHECK_MSG(staged == listed && staged_std == listed, where);
            // Exactly Quiescence mode's moves, then every quiet check.
            std::vector<Move> expected = list_sequence(q, PickMode::Quiescence);
            for (const Move m : list_sequence(q, PickMode::All))
                if (is_quiet_check(q, attacks(), m)) expected.push_back(m);
            CHECK_MSG(listed == expected, where);
            for (const Move m : listed) checks += m.is_capture() || m.is_promotion() ? 0 : 1;
        };
        testutil::walk_tree(testutil::parse_or_die(fen), 3, budget, visit);
    }
    CHECK(nodes > 1000 && checks > 100);
    // Castling that gives check (the rook lands on f1) stays out.
    const QuadBoard castle = to_board<QuadBoard>(testutil::parse_or_die("5k2/8/8/8/8/8/8/4K2R w K - 0 1"));
    REQUIRE(gives_check(castle, Move(E1, G1, KING_CASTLE), attacks()));
    for (const Move m : list_sequence(castle, PickMode::QuiescenceChecks))
        CHECK_MSG(m.flags() != KING_CASTLE, move_to_uci(m));
    CHECK(staged_sequence(castle, PickMode::QuiescenceChecks) == list_sequence(castle, PickMode::QuiescenceChecks));
}

TEST_CASE(qchecks_staged_picker_finds_every_kind_of_check) {
    const std::vector<std::string> fens = {
        "8/6k1/8/8/8/2N5/1B6/4K3 w - - 0 1",       // the knight uncovers the bishop
        "R2N3k/8/8/8/8/8/8/4K3 w - - 0 1",         // the knight uncovers the rook (Nf7: both)
        "4k3/8/8/8/4B3/8/8/K3R3 w - - 0 1",        // the bishop uncovers the rook
        "8/6k1/8/8/3R4/8/1Q6/4K3 w - - 0 1",       // the rook uncovers the queen
        "8/6k1/8/8/8/2K5/1B6/8 w - - 0 1",         // the king uncovers the bishop (not Kd4)
        "4k3/8/8/8/4K3/8/8/4R3 w - - 0 1",         // the king uncovers the rook (not Ke3, Ke5)
        "8/6k1/8/8/8/2P5/1B6/4K3 w - - 0 1",       // a push uncovers the bishop
        "8/8/8/8/8/8/R2P3k/4K3 w - - 0 1",         // single and double pushes uncover the rook
        "4k3/8/8/8/4P3/8/8/KR2R3 w - - 0 1",       // a push that stays in the rook's way (Rb8 checks)
        "8/3k4/8/4P3/8/8/8/4K3 w - - 0 1",         // a push checks
        "8/8/8/3k4/8/8/4P3/4K3 w - - 0 1",         // a double push checks (a single one doesn't)
        "8/8/8/4k3/8/3P4/5P2/R1BQK1N1 w Q - 0 1",  // direct: Nf3 Bb2 Bf4 Ra5 Qh5 d4 f4
    };
    for (const std::string& fen : fens) {
        const Position white = testutil::parse_or_die(fen);
        for (const Position& p : {white, testutil::mirror(white)}) {
            const QuadBoard q = to_board<QuadBoard>(p);
            REQUIRE(!make_check_info(q, attacks()).in_check());
            std::vector<Move> expected = list_sequence(q, PickMode::Quiescence);
            int checks = 0;
            for (const Move m : list_sequence(q, PickMode::All)) {
                if (m.flags() <= DOUBLE_PUSH && gives_check(q, m, attacks())) {
                    expected.push_back(m);
                    ++checks;
                }
            }
            const std::string where = format_fen(p);
            CHECK_MSG(checks > 0, where);
            CHECK_MSG(staged_sequence(q, PickMode::QuiescenceChecks) == expected, where);
            CHECK_MSG(staged_sequence(to_board<StdBoard>(p), PickMode::QuiescenceChecks) == expected, where);
            CHECK_MSG(list_sequence(q, PickMode::QuiescenceChecks) == expected, where);
        }
    }
}

TEST_CASE(qchecks_matches_recursive) {
    check_against_recursive<QuadBoard, FeaturesQChecksOnly, false, false, false, false, false, false, false>();
    check_against_recursive<StdBoard, FeaturesQChecksOnly, false, false, false, false, false, false, false>();
    check_against_recursive<QuadBoard, FeaturesKillersSeeQChecks, false, false, false, false, true, true, false>();
}

TEST_CASE(qchecks_with_everything_matches_recursive) {
    check_against_recursive<QuadBoard, FeaturesQChecksEverything, true, true, true, true, true, true, true>();
}

TEST_CASE(qchecks_find_a_back_rank_mate_in_quiescence) {
    // Depth 0: only quiescence. Rd8 mates; it captures nothing.
    const Position p = testutil::parse_or_die("6k1/5ppp/8/8/8/8/5PPP/3R2K1 w - - 0 1");
    const PruneParams prune;
    const testref::Outcome with =
        testref::flat_search<QuadBoard, PestoEval, FeaturesQChecksOnly>(p, 0, -kScoreInfinite, kScoreInfinite, 0, 0,
                                                                          prune, {});
    const testref::Outcome without = testref::flat_search<QuadBoard, PestoEval, FeaturesNone>(
        p, 0, -kScoreInfinite, kScoreInfinite, 0, 0, prune, {});
    CHECK_MSG(with.score == mate_in(1) && with.best == Move(D1, D8, QUIET),
              std::to_string(with.score) + " " + move_to_uci(with.best));
    CHECK_MSG(!is_mate_score(without.score), std::to_string(without.score));
}

TEST_CASE(qchecks_engine_sets_play_legal_moves) {
    HostTT table(16);
    for (const FeatureKind f : {FeatureKind::TTKillersSeeQchecks, FeatureKind::TTKillersSeeQchecksNmp}) {
        for (const std::string& fen : positions()) {
            const Position p = testutil::parse_or_die(fen);
            CpuSearchOptions o{BoardKind::Quad, EvalKind::Terms, OrderingKind::Staged};
            o.features = f;
            table.clear();
            o.tt = table.desc();
            const FixedDepthResult r = cpu_search_fixed(p, GameHistory{}, kRefDepth + 1, -kScoreInfinite,
                                                        kScoreInfinite, o);
            MoveList list;
            generate_legal(to_board<QuadBoard>(p), attacks(), list);
            bool legal = list.size == 0 && r.search.best.is_none();
            for (int i = 0; i < list.size; ++i) legal = legal || list.moves[i] == r.search.best;
            CHECK_MSG(legal, std::string(name_of(f)) + " " + fen + ": " + move_to_uci(r.search.best));
        }
    }
}

TEST_CASE(qchecks_quiet_check_test_agrees_with_playing_the_move) {
    long long quiet = 0, checks = 0;
    for (const std::string& fen : positions()) {
        long long budget = testutil::kTreeBudget / 2;
        auto visit = [&](const Position& p, const std::vector<Move>& moves) {
            const QuadBoard q = to_board<QuadBoard>(p);
            const StdBoard s = to_board<StdBoard>(p);
            for (const Move m : moves) {
                if (m.flags() > DOUBLE_PUSH) continue;  // captures, castling, promotions
                ++quiet;
                const bool slow = gives_check(q, m, attacks());
                checks += slow ? 1 : 0;
                CHECK_MSG(quiet_move_gives_check(q, m, attacks()) == slow &&
                              quiet_move_gives_check(s, m, attacks()) == slow,
                          format_fen(p) + " " + move_to_uci(m));
            }
        };
        testutil::walk_tree(testutil::parse_or_die(fen), 3, budget, visit);
    }
    CHECK(quiet > 100000 && checks > 1000);
}

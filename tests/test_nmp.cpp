// Null-move pruning (core/search.h; feature set tt-nmp): the pass itself,
// when a node may try one, and the search with it.
//
// What the search must guarantee:
// - its null moves match a recursive implementation exactly (score and node
//   count; without a table, through a test-only feature set), with every
//   setting and for tasks below the root;
// - with a table (tt-nmp), exactly tt's search wherever no null move is
//   allowed: positions with only kings and pawns, and searches no deeper
//   than the minimum null-move depth (the root never passes, and its
//   children are too shallow);
// - mates keep their exact distance; fewer nodes to the same depth; every
//   board and ordering visits the same tree.
//
// Of 17 bugs planted one at a time in the null-move code, these tests catch
// all but one that can't change a search: dropping the flag that forbids a
// pass right after a pass (see level_flags::kAfterNull). nmp_task_root_flags
// tests the flag where it can matter, at task roots.

#include <algorithm>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "core/search.h"
#include "host/fen.h"
#include "host/host_tables.h"
#include "host/position.h"
#include "host/search_cpu.h"
#include "host/tt_storage.h"
#include "host/uci_move.h"
#include "recursive_search_ref.h"
#include "test_harness.h"
#include "test_util.h"

using namespace cb;

namespace {

using testref::attacks;
using testref::tables;

#ifdef NDEBUG
constexpr int kRefDepth = 5;
constexpr int kNodesDepth = 6;
#else
constexpr int kRefDepth = 3;
constexpr int kNodesDepth = 4;
#endif

// Null moves without a table, so the flat search can be compared with the
// recursive reference (tests/recursive_search_ref.h).
struct FeaturesNullOnly {
    static constexpr const char* kName = "null-only";
    static constexpr bool kTT = false;
    static constexpr bool kTTCutoffs = false;
    static constexpr bool kTTQuiescence = false;
    static constexpr bool kNullMove = true;
    static constexpr bool kLmr = false;
    static constexpr bool kFutility = false;
    static constexpr bool kCheckExt = false;
    static constexpr bool kKillers = false;
    static constexpr bool kQPrune = false;
};

// Kings and pawns only: no side may pass.
const char* const kPawnEndgames[] = {
    "8/8/4k3/8/2P5/4K3/8/8 w - - 0 1",
    "8/5k2/3p4/1p1P4/1P6/5K2/8/8 w - - 0 1",
    "8/pp3k2/8/2P5/5K2/1P6/8/8 b - - 0 1",
    "4k3/4p3/4P3/8/8/8/8/4K3 w - - 0 1",
};

// Positions where null moves happen (pieces, quiet enough to pass).
std::vector<std::string> middlegames() {
    std::vector<std::string> v = {
        "r1bq1rk1/pp2bppp/2n1pn2/3p4/2PP4/2N1PN2/PP3PPP/R2QKB1R w KQ - 0 8",
        "2r2rk1/1bqnbppp/pp1ppn2/8/2PNP3/1PN1BP2/P2Q2PP/2RR1BK1 w - - 0 15",
        "r2q1rk1/1p1nbppp/p2pbn2/4p3/4P3/1NN1BP2/PPPQ2PP/2KR1B1R w - - 0 11",
    };
    for (const std::string& fen : testutil::test_fens()) v.push_back(fen);
    return v;
}

using testref::Outcome;

// search_task with null moves and no table.
template <class Board, class Eval>
Outcome flat_search(const Position& p, int depth, int alpha, int beta, int ply, u8 flags, const PruneParams& prune,
                    const std::vector<Move>& pv_hint) {
    return testref::flat_search<Board, Eval, FeaturesNullOnly>(p, depth, alpha, beta, ply, flags, prune, pv_hint);
}

// The recursive reference with null moves only.
template <class Board, class Eval>
using RecursiveNullSearch = testref::RecursiveSearch<Board, Eval, true, false>;

template <class Board, class Eval>
RecursiveNullSearch<Board, Eval> reference(const Position& p, const PruneParams& prune, const std::vector<Move>& pv) {
    return testref::reference<Board, Eval, true, false>(p, prune, pv);
}

// Compares the flat search with the reference on every test position, in
// iterative deepening (each depth ordered by the previous one's PV).
template <class Board, class Eval>
void check_against_recursive(const PruneParams& prune) {
    u64 tries = 0, cuts = 0;
    for (const std::string& fen : middlegames()) {
        const Position p = testutil::parse_or_die(fen);
        std::vector<Move> pv;
        for (int depth = 1; depth <= kRefDepth; ++depth) {
            const Outcome flat =
                flat_search<Board, Eval>(p, depth, -kScoreInfinite, kScoreInfinite, 0, kTaskOnPv, prune, pv);
            RecursiveNullSearch<Board, Eval> rec = reference<Board, Eval>(p, prune, pv);
            const int score = rec.search(depth, -kScoreInfinite, kScoreInfinite, 0, 0, true, false);
            const std::string where = fen + " depth " + std::to_string(depth) + " null " +
                                      std::to_string(prune.null_base) + "+d/" + std::to_string(prune.null_divisor) +
                                      " from " + std::to_string(prune.null_min_depth);
            CHECK_MSG(flat.score == score, where + ": " + std::to_string(flat.score) + " vs " + std::to_string(score));
            CHECK_MSG(flat.nodes == rec.nodes,
                      where + ": nodes " + std::to_string(flat.nodes) + " vs " + std::to_string(rec.nodes));
            tries += rec.null_tries;
            cuts += rec.null_cuts;
            pv = flat.pv;
        }
    }
    // Not vacuous: null moves were tried, and some cut off.
    CHECK_MSG(tries > 0 && cuts > 0 && cuts < tries,
              std::to_string(tries) + " null moves, " + std::to_string(cuts) + " cutoffs");
}

template <class Board>
bool same_pieces(const Board& a, const Board& b) {
    for (const Color c : {WHITE, BLACK}) {
        if (a.color(c) != b.color(c)) return false;
        if ((a.template pieces<PAWN>() & a.color(c)) != (b.template pieces<PAWN>() & b.color(c))) return false;
        if ((a.template pieces<KNIGHT>() & a.color(c)) != (b.template pieces<KNIGHT>() & b.color(c))) return false;
        if ((a.template pieces<BISHOP>() & a.color(c)) != (b.template pieces<BISHOP>() & b.color(c))) return false;
        if ((a.template pieces<ROOK>() & a.color(c)) != (b.template pieces<ROOK>() & b.color(c))) return false;
        if ((a.template pieces<QUEEN>() & a.color(c)) != (b.template pieces<QUEEN>() & b.color(c))) return false;
        if ((a.template pieces<KING>() & a.color(c)) != (b.template pieces<KING>() & b.color(c))) return false;
    }
    return true;
}

const TTLayoutKind kLayouts[] = {TTLayoutKind::Packed64, TTLayoutKind::Wide128};

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

// ---------------------------------------------------------------------------
// The pass and when to try it
// ---------------------------------------------------------------------------

TEST_CASE(nmp_null_move_round_trip) {
    // Passing changes the side to move and nothing else on the board, drops
    // the en passant square, restarts the halfmove clock (so repetition
    // checks never reach back across the pass), keeps the evaluation
    // accumulator, and updates the key to the new position's.
    int checked = 0;
    for (const std::string& fen : testutil::test_fens()) {
        const Position p = testutil::parse_or_die(fen);
        const QuadBoard b = to_board<QuadBoard>(p);
        if (make_check_info(b, attacks()).in_check()) continue;  // never passes in check
        SearchBoard<QuadBoard, PestoEval> sb = make_search_board<PestoEval>(b, attacks(), tables());
        const SearchBoard<QuadBoard, PestoEval> before = sb;
        const Undo undo = make_null_search_move(sb, attacks(), tables());
        const u32 old_state = before.board.state();
        const u32 st = sb.board.state();
        CHECK_MSG(state::side(st) == ~state::side(old_state), fen);
        CHECK_MSG(state::castling(st) == state::castling(old_state), fen);
        CHECK_MSG(!state::has_ep(st), fen);
        CHECK_MSG(state::halfmove(st) == 0, fen);
        CHECK_MSG(same_pieces(sb.board, before.board), fen);
        CHECK_MSG(sb.acc == before.acc, fen);
        CHECK_MSG(sb.key == compute_key(sb.board, attacks(), tables().zobrist), fen);
        CHECK_MSG(sb.key != before.key, fen);
        unmake_null_search_move(sb, undo, before.key);
        CHECK_MSG(sb.board.state() == old_state && sb.key == before.key && same_pieces(sb.board, before.board), fen);
        // The plain board function does the same.
        QuadBoard plain = b;
        const Undo u = make_null_move(plain);
        CHECK_MSG(plain.state() == st, fen);
        unmake_null_move(plain, u);
        CHECK_MSG(plain.state() == old_state, fen);
        ++checked;
    }
    CHECK(checked > 10);
}

TEST_CASE(nmp_allowed_conditions) {
    const Position p = testutil::parse_or_die("r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10");
    const QuadBoard b = to_board<QuadBoard>(p);
    const PruneParams pp;  // null moves from depth 2
    const auto allowed = [&](const QuadBoard& board, Color us, int ply, int depth, u8 flags, bool in_check, int beta) {
        return null_move_allowed<PestoEval>(pp, ply, depth, flags, in_check, beta, board, us,
                                            compute_accumulator<PestoEval>(board, tables().eval));
    };
    const int eval = PestoEval::evaluate(compute_accumulator<PestoEval>(b, tables().eval), WHITE);
    CHECK(allowed(b, WHITE, 1, 2, 0, false, eval));        // static eval at beta: enough
    CHECK(allowed(b, WHITE, 5, 9, 0, false, eval - 300));  // well above beta, deep
    CHECK(!allowed(b, WHITE, 1, 2, 0, false, eval + 1));   // below beta
    CHECK(!allowed(b, WHITE, 0, 2, 0, false, eval));       // the root
    CHECK(!allowed(b, WHITE, 1, 1, 0, false, eval));       // too shallow
    CHECK(!allowed(b, WHITE, 1, 0, 0, false, eval));       // quiescence
    CHECK(!allowed(b, WHITE, 1, 2, level_flags::kOnPv, false, eval));
    CHECK(!allowed(b, WHITE, 1, 2, level_flags::kAfterNull, false, eval));
    CHECK(allowed(b, WHITE, 1, 2, level_flags::kStore | level_flags::kSearchedAny, false, eval));  // other flags don't matter
    CHECK(!allowed(b, WHITE, 1, 2, 0, true, eval));                                                 // in check
    CHECK(!allowed(b, WHITE, 1, 2, 0, false, -kScoreMateBound - 1));  // beta says we're mated
    CHECK(allowed(b, WHITE, 1, 2, 0, false, -kScoreMateBound));       // the last score that isn't a mate
    // Only pawns besides the king: never, for that side.
    const QuadBoard pawns = to_board<QuadBoard>(testutil::parse_or_die("8/5k2/3p4/1p1P4/1P6/5K2/8/6N1 w - - 0 1"));
    CHECK(allowed(pawns, WHITE, 1, 4, 0, false, -1000));   // white has a knight
    CHECK(!allowed(pawns, BLACK, 1, 4, 0, false, -1000));  // black has pawns only
    // The minimum depth is a setting.
    PruneParams deep;
    deep.null_min_depth = 4;
    CHECK(!null_move_allowed<PestoEval>(deep, 1, 3, 0, false, eval, b, WHITE,
                                        compute_accumulator<PestoEval>(b, tables().eval)));
    CHECK(null_move_allowed<PestoEval>(deep, 1, 4, 0, false, eval, b, WHITE,
                                       compute_accumulator<PestoEval>(b, tables().eval)));
}

TEST_CASE(nmp_cutoff_score) {
    // Fail-soft, but a mate found after passing becomes beta.
    CHECK_EQ(null_move_score(120, 100), 120);
    CHECK_EQ(null_move_score(100, 100), 100);
    CHECK_EQ(null_move_score(kScoreMateBound, 100), kScoreMateBound);  // the last score that isn't a mate
    CHECK_EQ(null_move_score(kScoreMateBound + 1, 100), 100);
    CHECK_EQ(null_move_score(mate_in(3), -50), -50);
}

TEST_CASE(nmp_settings) {
    const PruneParams d;
    CHECK_EQ(null_move_reduction(d, 2), 3);
    CHECK_EQ(null_move_reduction(d, 5), 3);
    CHECK_EQ(null_move_reduction(d, 6), 4);
    CHECK_EQ(null_move_reduction(d, 12), 5);
    PruneParams fixed;
    fixed.null_base = 2;
    fixed.null_divisor = 0;
    CHECK_EQ(null_move_reduction(fixed, 20), 2);

    CpuSearchOptions o{BoardKind::Quad, EvalKind::Pesto, OrderingKind::Staged};
    o.features = FeatureKind::TTNmp;
    CHECK_EQ(describe(o), std::string("quad/pesto/staged tt-nmp wide128 null 3+d/6 from 2"));
    o.prune = fixed;
    o.prune.null_min_depth = 3;
    CHECK_EQ(describe(o), std::string("quad/pesto/staged tt-nmp wide128 null 2 from 3"));
    o.features = FeatureKind::TT;
    CHECK_EQ(describe(o), std::string("quad/pesto/staged tt wide128"));

    const auto rejects = [](u8 min_depth, u8 base, u8 divisor) {
        PruneParams p;
        p.null_min_depth = min_depth;
        p.null_base = base;
        p.null_divisor = divisor;
        try {
            validate_prune(p);
        } catch (const std::invalid_argument&) {
            return true;
        }
        return false;
    };
    CHECK(!rejects(2, 3, 6));
    CHECK(!rejects(1, 1, 0));
    CHECK(!rejects(20, 10, 20));
    CHECK(rejects(0, 3, 6));
    CHECK(rejects(21, 3, 6));
    CHECK(rejects(2, 0, 6));
    CHECK(rejects(2, 11, 6));
    CHECK(rejects(2, 3, 21));
    // The search checks them too.
    HostTT table(1);
    CpuSearchOptions bad = with_table(FeatureKind::TTNmp, TTLayoutKind::Packed64, table);
    bad.prune.null_base = 0;
    bool threw = false;
    try {
        search(testutil::parse_or_die(kStartFen), 1, bad);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    CHECK(threw);
}

// ---------------------------------------------------------------------------
// The search
// ---------------------------------------------------------------------------

TEST_CASE(nmp_search_matches_recursive_quad_pesto) { check_against_recursive<QuadBoard, PestoEval>(PruneParams{}); }
TEST_CASE(nmp_search_matches_recursive_std_material) { check_against_recursive<StdBoard, MaterialEval>(PruneParams{}); }

TEST_CASE(nmp_search_matches_recursive_settings) {
    PruneParams fixed2;  // R = 2 everywhere, from depth 1
    fixed2.null_min_depth = 1;
    fixed2.null_base = 2;
    fixed2.null_divisor = 0;
    check_against_recursive<QuadBoard, PestoEval>(fixed2);
    // R = 1 + depth / 2, from depth 2: small enough that the depth term
    // decides whether a null search starts in quiescence at these depths.
    PruneParams steep;
    steep.null_min_depth = 2;
    steep.null_base = 1;
    steep.null_divisor = 2;
    check_against_recursive<QuadBoard, PestoEval>(steep);
}

TEST_CASE(nmp_task_root_flags) {
    // A task below the root may pass at its own root, unless the move into
    // it was a null move or it lies on the previous PV: the task flags
    // carry that, and the reference agrees in each case.
    const PruneParams prune;
    const Position p = testutil::parse_or_die(middlegames()[0]);
    const QuadBoard b = to_board<QuadBoard>(p);
    const int eval = PestoEval::evaluate(compute_accumulator<PestoEval>(b, tables().eval), state::side(b.state()));
    const int beta = eval - 50;  // the pass is tried at the root unless a flag forbids it
    const int depth = 4;
    u64 nodes[3] = {};
    const u8 flags[3] = {0, kTaskAfterNull, kTaskOnPv};
    for (int i = 0; i < 3; ++i) {
        const Outcome flat = flat_search<QuadBoard, PestoEval>(p, depth, beta - 1, beta, 1, flags[i], prune, {});
        RecursiveNullSearch<QuadBoard, PestoEval> rec = reference<QuadBoard, PestoEval>(p, prune, {});
        const int score = rec.search(depth, beta - 1, beta, 1, 0, flags[i] == kTaskOnPv, flags[i] == kTaskAfterNull);
        CHECK_MSG(flat.score == score && flat.nodes == rec.nodes,
                  "flags " + std::to_string(flags[i]) + ": " + std::to_string(flat.score) + "/" +
                      std::to_string(flat.nodes) + " vs " + std::to_string(score) + "/" + std::to_string(rec.nodes));
        nodes[i] = flat.nodes;
        if (i == 0) CHECK(rec.null_tries > 0 && flat.score >= beta);
    }
    CHECK_MSG(nodes[1] != nodes[0] && nodes[2] != nodes[0],
              std::to_string(nodes[0]) + " " + std::to_string(nodes[1]) + " " + std::to_string(nodes[2]));
}

TEST_CASE(nmp_exact_where_no_null_move_is_allowed) {
    // tt-nmp against tt, each search from an empty table.
    HostTT table(16);
    const auto compare = [&](const Position& p, int depth, TTLayoutKind layout, const PruneParams& prune,
                             const std::vector<Move>& pv, const std::string& where) {
        table.clear();
        const FixedDepthResult tt = search(p, depth, with_table(FeatureKind::TT, layout, table), pv);
        table.clear();
        CpuSearchOptions o = with_table(FeatureKind::TTNmp, layout, table);
        o.prune = prune;
        const FixedDepthResult nmp = search(p, depth, o, pv);
        CHECK_MSG(nmp.search.score == tt.search.score && nmp.search.nodes == tt.search.nodes &&
                      nmp.search.best == tt.search.best,
                  where + " depth " + std::to_string(depth) + ": " + std::to_string(nmp.search.score) + "/" +
                      std::to_string(nmp.search.nodes) + " vs " + std::to_string(tt.search.score) + "/" +
                      std::to_string(tt.search.nodes));
        return tt.pv;
    };
    for (const TTLayoutKind layout : kLayouts) {
        // Kings and pawns: no side ever passes.
        for (const char* fen : kPawnEndgames) {
            const Position p = testutil::parse_or_die(fen);
            std::vector<Move> pv;
            for (int depth = 1; depth <= kNodesDepth + 1; ++depth)
                pv = compare(p, depth, layout, PruneParams{}, pv, std::string(fen) + " " + name_of(layout));
        }
        // No deeper than the minimum null-move depth.
        PruneParams deep;
        deep.null_min_depth = 4;
        for (const std::string& fen : middlegames()) {
            const Position p = testutil::parse_or_die(fen);
            std::vector<Move> pv;
            for (int depth = 1; depth <= 2; ++depth) pv = compare(p, depth, layout, PruneParams{}, pv, fen);
            pv.clear();
            for (int depth = 1; depth <= std::min(4, kNodesDepth); ++depth) pv = compare(p, depth, layout, deep, pv, fen);
        }
    }
}

TEST_CASE(nmp_prunes) {
    // Iterative deepening on one table per position, as the engine searches.
    HostTT table(64);
    u64 with_null = 0, without = 0;
    for (const std::string& fen : middlegames()) {
        const Position p = testutil::parse_or_die(fen);
        for (const FeatureKind f : {FeatureKind::TT, FeatureKind::TTNmp}) {
            table.clear();
            std::vector<Move> pv;
            u64 total = 0;
            for (int depth = 1; depth <= kNodesDepth; ++depth) {
                table.new_search();
                const FixedDepthResult r = search(p, depth, with_table(f, TTLayoutKind::Packed64, table), pv);
                total += r.search.nodes;
                pv = r.pv;
            }
            (f == FeatureKind::TT ? without : with_null) += total;
        }
    }
    CHECK_MSG(with_null < without, std::to_string(with_null) + " nodes with null moves vs " + std::to_string(without));
}

TEST_CASE(nmp_mates_keep_their_distance) {
    // A mate the plain search finds at `depth` is found with the same
    // distance a few plies deeper with null moves (a pass never proves a
    // mate, and moves on the PV never pass).
    struct Case {
        const char* fen;
        int depth;
    };
    const Case cases[] = {
        {"6k1/5ppp/8/8/8/8/8/R5K1 w - - 0 1", 3},                                    // mate in 1
        {"k7/8/1K6/8/8/8/8/7R b - - 0 1", 4},                                        // mated in 1
        {"7k/8/6K1/8/8/8/8/6Q1 w - - 0 1", 5},                                       // queen and king
        {"r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5Q2/PPPP1PPP/RNB1K1NR w KQkq - 4 4", 3},  // Qxf7#
    };
    HostTT table(16);
    for (const Case& c : cases) {
        const Position p = testutil::parse_or_die(c.fen);
        const int mate = search(p, c.depth, {BoardKind::Quad, EvalKind::Pesto, OrderingKind::Staged}).search.score;
        REQUIRE(is_mate_score(mate));
        for (const TTLayoutKind layout : kLayouts) {
            table.clear();
            std::vector<Move> pv;
            int score = 0;
            for (int depth = 1; depth <= c.depth + 3; ++depth) {
                table.new_search();
                const FixedDepthResult r = search(p, depth, with_table(FeatureKind::TTNmp, layout, table), pv);
                score = r.search.score;
                pv = r.pv;
            }
            CHECK_MSG(score == mate, std::string(c.fen) + " " + name_of(layout) + ": " + std::to_string(score) +
                                         " vs " + std::to_string(mate));
        }
        // Without a table too.
        std::vector<Move> pv;
        Outcome o;
        for (int depth = 1; depth <= c.depth + 3; ++depth) {
            o = flat_search<QuadBoard, PestoEval>(p, depth, -kScoreInfinite, kScoreInfinite, 0, kTaskOnPv,
                                                  PruneParams{}, pv);
            pv = o.pv;
        }
        CHECK_MSG(o.score == mate, std::string(c.fen) + " without a table: " + std::to_string(o.score));
    }
}

TEST_CASE(nmp_variants_agree) {
    // Every board and ordering: the same tree with null moves and a table.
    HostTT table(16);
    for (const std::string& fen : middlegames()) {
        const Position p = testutil::parse_or_die(fen);
        table.clear();
        const FixedDepthResult first =
            search(p, kNodesDepth - 1, with_table(FeatureKind::TTNmp, TTLayoutKind::Wide128, table));
        for (const BoardKind board : {BoardKind::Quad, BoardKind::Std}) {
            for (const OrderingKind ordering : {OrderingKind::Staged, OrderingKind::List}) {
                table.clear();
                const FixedDepthResult r = search(
                    p, kNodesDepth - 1, with_table(FeatureKind::TTNmp, TTLayoutKind::Wide128, table, board, ordering));
                CHECK_MSG(r.search.score == first.search.score && r.search.nodes == first.search.nodes &&
                              r.search.best == first.search.best && r.pv == first.pv,
                          fen + " " + name_of(board) + "/" + name_of(ordering));
            }
        }
    }
}

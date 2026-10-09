// Board representations and make/unmake, cross-checked against the
// reference implementation and against each other.

#include "core/board.h"
#include "host/debug_print.h"
#include "host/host_tables.h"
#include "host/uci_move.h"
#include "test_harness.h"
#include "test_util.h"

using namespace cb;

namespace {

// Masks computed straight from a Position's square array.
struct ExpectedMasks {
    u64 by_type[7] = {};
    u64 by_color[2] = {};
};

ExpectedMasks masks_of(const Position& p) {
    ExpectedMasks m;
    for (int i = 0; i < 64; ++i) {
        const u8 code = p.squares[i];
        if (!code) continue;
        m.by_type[code_type(code)] |= bit(Square(i));
        m.by_color[code_color(code)] |= bit(Square(i));
    }
    return m;
}

template <class Board>
void check_board_matches(const Board& b, const Position& p, const std::string& context) {
    const ExpectedMasks m = masks_of(p);
    CHECK_MSG(b.template pieces<PAWN>() == m.by_type[PAWN], context);
    CHECK_MSG(b.template pieces<KNIGHT>() == m.by_type[KNIGHT], context);
    CHECK_MSG(b.template pieces<BISHOP>() == m.by_type[BISHOP], context);
    CHECK_MSG(b.template pieces<ROOK>() == m.by_type[ROOK], context);
    CHECK_MSG(b.template pieces<QUEEN>() == m.by_type[QUEEN], context);
    CHECK_MSG(b.template pieces<KING>() == m.by_type[KING], context);
    CHECK_MSG(b.color(WHITE) == m.by_color[WHITE], context);
    CHECK_MSG(b.color(BLACK) == m.by_color[BLACK], context);
    CHECK_MSG(b.occupied() == (m.by_color[WHITE] | m.by_color[BLACK]), context);
    CHECK_MSG(b.diagonal_sliders() == (m.by_type[BISHOP] | m.by_type[QUEEN]), context);
    CHECK_MSG(b.straight_sliders() == (m.by_type[ROOK] | m.by_type[QUEEN]), context);
    for (int i = 0; i < 64; ++i)
        CHECK_MSG(b.piece_type_on(Square(i)) == code_type(p.squares[i]), context + " square " + square_name(Square(i)));
    CHECK_MSG(b.state() == p.state, context);
    CHECK_MSG(b.internal_error() == nullptr, context + ": " + (b.internal_error() ? b.internal_error() : ""));
}

template <class Board>
void check_round_trip_and_masks() {
    for (const std::string& fen : testutil::test_fens()) {
        const Position p = testutil::parse_or_die(fen);
        const Board b = to_board<Board>(p);
        check_board_matches(b, p, std::string(Board::kName) + " " + fen);
        CHECK_MSG(to_position(b, p.fullmove) == p, fen);
    }
}

}  // namespace

TEST_CASE(board_std_masks_and_round_trip) { check_round_trip_and_masks<StdBoard>(); }
TEST_CASE(board_quad_masks_and_round_trip) { check_round_trip_and_masks<QuadBoard>(); }

TEST_CASE(board_toggle_adds_and_removes) {
    StdBoard s;
    QuadBoard q;
    for (int c = 0; c < 2; ++c) {
        for (int pt = PAWN; pt <= KING; ++pt) {
            const Square sq = Square(8 + pt * 4 + c);
            s.toggle(Color(c), PieceType(pt), bit(sq));
            q.toggle(Color(c), PieceType(pt), bit(sq));
            CHECK_EQ(s.piece_type_on(sq), PieceType(pt));
            CHECK_EQ(q.piece_type_on(sq), PieceType(pt));
            CHECK((s.color(Color(c)) & bit(sq)) != 0);
            CHECK((q.color(Color(c)) & bit(sq)) != 0);
            CHECK(s.internal_error() == nullptr);
            CHECK(q.internal_error() == nullptr);
        }
    }
    CHECK(to_position(s) == to_position(q));
    for (int c = 0; c < 2; ++c) {
        for (int pt = PAWN; pt <= KING; ++pt) {
            const Square sq = Square(8 + pt * 4 + c);
            s.toggle(Color(c), PieceType(pt), bit(sq));
            q.toggle(Color(c), PieceType(pt), bit(sq));
        }
    }
    CHECK(s == StdBoard());
    CHECK(q == QuadBoard());
}

TEST_CASE(board_king_square_and_attackers) {
    const Position p = testutil::parse_or_die("k3r3/8/8/8/8/5n2/8/4K3 w - - 0 1");
    const HqAttacks a(host_attack_tables());
    const StdBoard s = to_board<StdBoard>(p);
    const QuadBoard q = to_board<QuadBoard>(p);
    CHECK_EQ(king_square(s, WHITE), E1);
    CHECK_EQ(king_square(q, BLACK), A8);
    // e1 is attacked by the rook on e8 and the knight on f3.
    CHECK_EQ(attackers_to(s, a, E1, s.occupied()), bit(E8) | bit(F3));
    CHECK_EQ(attackers_to(q, a, E1, q.occupied()), bit(E8) | bit(F3));
}

namespace {

// For every node of each test position's tree: play every legal move on both
// boards and on the reference, compare the results, then take the move back
// and check the board is restored exactly.
template <class Board>
void check_make_unmake_against_reference() {
    long long moves_checked = 0;
    for (const std::string& fen : testutil::test_fens()) {
        long long budget = testutil::kTreeBudget;
        auto visit = [&](const Position& p, const std::vector<Move>& ref_moves) {
            Board b = to_board<Board>(p);
            const Board original = b;
            const ref::Board r = ref::from_position(p);
            for (const Move m : ref_moves) {
                const Undo undo = make_move(b, m);
                const Position expected = ref::to_position(ref::play(r, m));
                const Position got = to_position(b);
                if (!(got == expected)) {
                    CHECK_MSG(false, std::string(Board::kName) + " make " + move_to_uci(m) + " in " + format_fen(p) +
                                         "\n got      " + format_fen(got) + "\n expected " + format_fen(expected));
                }
                CHECK_MSG(b.internal_error() == nullptr, format_fen(p) + " " + move_to_uci(m));
                unmake_move(b, m, undo);
                CHECK_MSG(b == original, std::string(Board::kName) + " unmake " + move_to_uci(m) + " in " + format_fen(p));
                ++moves_checked;
            }
        };
        testutil::walk_tree(testutil::parse_or_die(fen), 3, budget, visit);
    }
    CHECK(moves_checked > 1000);
}

}  // namespace

TEST_CASE(make_unmake_std_matches_reference) { check_make_unmake_against_reference<StdBoard>(); }
TEST_CASE(make_unmake_quad_matches_reference) { check_make_unmake_against_reference<QuadBoard>(); }

TEST_CASE(make_move_state_details) {
    // Double push sets the en passant file; any other move clears it.
    Position p = testutil::parse_or_die(kStartFen);
    StdBoard b = to_board<StdBoard>(p);
    make_move(b, Move(E2, E4, DOUBLE_PUSH));
    CHECK(state::has_ep(b.state()));
    CHECK_EQ(state::ep_square(b.state()), E3);
    CHECK_EQ(state::side(b.state()), BLACK);
    make_move(b, Move(G8, F6, QUIET));
    CHECK(!state::has_ep(b.state()));
    CHECK_EQ(state::halfmove(b.state()), 1);

    // King and rook moves remove the matching castling rights.
    p = testutil::parse_or_die("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1");
    b = to_board<StdBoard>(p);
    make_move(b, Move(H1, H5, QUIET));
    CHECK_EQ(state::castling(b.state()), u32(WHITE_OOO | BLACK_OO | BLACK_OOO));
    make_move(b, Move(E8, D8, QUIET));
    CHECK_EQ(state::castling(b.state()), u32(WHITE_OOO));
    // Capturing a rook on its square removes that side's right.
    p = testutil::parse_or_die("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1");
    b = to_board<StdBoard>(p);
    make_move(b, Move(A1, A8, CAPTURE));
    CHECK_EQ(state::castling(b.state()), u32(WHITE_OO | BLACK_OO));
    CHECK_EQ(state::halfmove(b.state()), 0);

    // The halfmove clock saturates at 127.
    p = testutil::parse_or_die("4k3/8/8/8/8/8/8/4K2R w K - 126 80");
    b = to_board<StdBoard>(p);
    make_move(b, Move(H1, H2, QUIET));
    CHECK_EQ(state::halfmove(b.state()), 127);
    make_move(b, Move(E8, D8, QUIET));
    CHECK_EQ(state::halfmove(b.state()), 127);
}

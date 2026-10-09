#include "host/position.h"

#include "core/board.h"
#include "host/host_tables.h"

namespace cb {

namespace {

bool fail(std::string& error, const char* message) {
    error = message;
    return false;
}

}  // namespace

bool validate_position(const Position& p, std::string& error) {
    int kings[2] = {0, 0};
    for (int i = 0; i < 64; ++i) {
        const u8 code = p.squares[i];
        if (code == 0) continue;
        const PieceType pt = code_type(code);
        if (code > piece_code(BLACK, KING) || pt < PAWN || pt > KING) return fail(error, "invalid piece code");
        if (pt == KING) ++kings[code_color(code)];
        const int rank = rank_of(Square(i));
        if (pt == PAWN && (rank == 0 || rank == 7)) return fail(error, "pawn on the first or last rank");
    }
    if (kings[WHITE] != 1 || kings[BLACK] != 1) return fail(error, "each side needs exactly one king");
    if (p.state >> 16) return fail(error, "state word uses bits above 15");

    const Color us = state::side(p.state);
    const u32 rights = state::castling(p.state);
    const auto has = [&](Square s, Color c, PieceType pt) { return p.squares[s] == piece_code(c, pt); };
    if ((rights & WHITE_OO) && !(has(E1, WHITE, KING) && has(H1, WHITE, ROOK)))
        return fail(error, "castling right K needs the white king on e1 and a white rook on h1");
    if ((rights & WHITE_OOO) && !(has(E1, WHITE, KING) && has(A1, WHITE, ROOK)))
        return fail(error, "castling right Q needs the white king on e1 and a white rook on a1");
    if ((rights & BLACK_OO) && !(has(E8, BLACK, KING) && has(H8, BLACK, ROOK)))
        return fail(error, "castling right k needs the black king on e8 and a black rook on h8");
    if ((rights & BLACK_OOO) && !(has(E8, BLACK, KING) && has(A8, BLACK, ROOK)))
        return fail(error, "castling right q needs the black king on e8 and a black rook on a8");

    if (state::has_ep(p.state)) {
        // The opponent just pushed a pawn two squares: from `origin`, over
        // `to` (the en passant square), onto `pushed`.
        const Square to = state::ep_square(p.state);
        const Square pushed = Square(to ^ 8);
        const Square origin = Square(2 * int(to) - int(pushed));
        if (p.squares[to] != 0 || p.squares[origin] != 0 || !has(pushed, ~us, PAWN))
            return fail(error, "en passant square does not follow a double pawn push");
    }

    // The side that just moved can't have left its own king in check.
    const StdBoard b = to_board<StdBoard>(p);
    const HqAttacks a(host_attack_tables());
    if (attackers_to(b, a, king_square(b, ~us), b.occupied()) & b.color(us))
        return fail(error, "the side not to move is in check");
    return true;
}

}  // namespace cb

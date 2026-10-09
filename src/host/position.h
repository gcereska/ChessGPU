#pragma once

#include <array>
#include <string>

#include "core/state.h"

namespace cb {

// Piece codes in a Position: 0 = empty, otherwise color * 8 + piece type
// (white 1-6, black 9-14).
constexpr u8 piece_code(Color c, PieceType pt) { return u8(int(c) * 8 + int(pt)); }
constexpr Color code_color(u8 code) { return Color(code >> 3); }
constexpr PieceType code_type(u8 code) { return PieceType(code & 7); }

// Representation-independent position: used for parsing, printing,
// validation, and comparing boards of different representations.
struct Position {
    std::array<u8, 64> squares{};  // piece code per square
    u32 state = 0;                 // state word (see core/state.h)
    int fullmove = 1;              // not tracked by boards; kept for FEN round trips

    bool operator==(const Position&) const = default;
};

template <class Board>
Board to_board(const Position& p) {
    Board b;
    for (int i = 0; i < 64; ++i) {
        const u8 code = p.squares[i];
        if (code) b.toggle(code_color(code), code_type(code), bit(Square(i)));
    }
    b.set_state(p.state);
    return b;
}

template <class Board>
Position to_position(const Board& b, int fullmove = 1) {
    Position p;
    const u64 black = b.color(BLACK);
    for (int i = 0; i < 64; ++i) {
        const PieceType pt = b.piece_type_on(Square(i));
        if (pt != NO_PIECE_TYPE) p.squares[i] = piece_code(((black >> i) & 1) ? BLACK : WHITE, pt);
    }
    p.state = b.state();
    p.fullmove = fullmove;
    return p;
}

// Checks that p is a legal chess position (not that it is reachable): valid
// piece codes, one king per side, no pawns on the first or last rank, the
// side not to move is not in check, and the castling rights and en passant
// square match the pieces. On failure returns false and sets `error`.
bool validate_position(const Position& p, std::string& error);

}  // namespace cb

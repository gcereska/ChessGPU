#pragma once

// Generic board operations, written once against the board interface (see
// board_std.h) and shared by every board representation.

#include <type_traits>

#include "core/attacks.h"
#include "core/board_quad.h"
#include "core/board_std.h"

namespace cb {

// The same position in another board representation.
template <class To, class From>
CB_HD To convert_board(const From& b) {
    if constexpr (std::is_same_v<To, From>) {
        return b;
    } else {
        To t;
        const u64 black = b.color(BLACK);
        const u64 white = b.color(WHITE);
        const u64 masks[6] = {b.template pieces<PAWN>(), b.template pieces<KNIGHT>(), b.template pieces<BISHOP>(),
                              b.template pieces<ROOK>(),  b.template pieces<QUEEN>(),  b.template pieces<KING>()};
        CB_UNROLL
        for (int k = 0; k < 6; ++k) {
            t.toggle(WHITE, PieceType(k + 1), masks[k] & white);
            t.toggle(BLACK, PieceType(k + 1), masks[k] & black);
        }
        t.set_state(b.state());
        return t;
    }
}

template <class Board>
CB_HD Square king_square(const Board& b, Color c) {
    return Square(lsb(b.template pieces<KING>() & b.color(c)));
}

// Pieces of both colors attacking square s, given the occupied squares.
template <class Board, class Attacks>
CB_HD u64 attackers_to(const Board& b, const Attacks& a, Square s, u64 occupied) {
    const u64 pawns = b.template pieces<PAWN>();
    // A pawn of color c on s attacks exactly the squares from which an
    // opposing pawn attacks s.
    return (a.pawn(WHITE, s) & pawns & b.color(BLACK)) | (a.pawn(BLACK, s) & pawns & b.color(WHITE)) |
           (a.knight(s) & b.template pieces<KNIGHT>()) | (a.king(s) & b.template pieces<KING>()) |
           (a.bishop(s, occupied) & b.diagonal_sliders()) | (a.rook(s, occupied) & b.straight_sliders());
}

// Is the side to move in check? (After a move: did the move give check?)
template <class Board, class Attacks>
CB_HD bool side_to_move_in_check(const Board& b, const Attacks& a) {
    const Color us = state::side(b.state());
    return (attackers_to(b, a, king_square(b, us), b.occupied()) & b.color(~us)) != 0;
}

// Does legal move m give check? (Plays it on a copy.)
template <class Board, class Attacks>
CB_HD bool gives_check(const Board& b, Move m, const Attacks& a) {
    Board after = b;
    make_move(after, m);
    return side_to_move_in_check(after, a);
}

// Does one of our sliders (side `us`) see their king, on `king`, once our
// piece on `from` has left it, with `occupied` the squares occupied then?
// What moving that piece can uncover.
template <class Board, class Attacks>
CB_HD bool slider_sees_king(const Board& b, const Attacks& a, Color us, Square king, Square from, u64 occupied) {
    const u64 ours = b.color(us) & ~bit(from);
    return ((a.bishop(king, occupied) & b.diagonal_sliders() & ours) |
            (a.rook(king, occupied) & b.straight_sliders() & ours)) != 0;
}

// The same for a quiet move that isn't castling (flags QUIET or
// DOUBLE_PUSH), without playing it: the piece checks from its new square, or
// one of our sliders sees the king through the square it left. Nothing else
// can: pawn, knight and king attacks don't depend on other pieces, and the
// opponent's king isn't in check on our move.
template <class Board, class Attacks>
CB_HD bool quiet_move_gives_check(const Board& b, Move m, const Attacks& a) {
    const Color us = state::side(b.state());
    const Square king = king_square(b, ~us);
    const Square from = m.from();
    const Square to = m.to();
    const u64 occupied = (b.occupied() ^ bit(from)) | bit(to);
    const PieceType pt = b.piece_type_on(from);
    u64 attacks = 0;  // a king checks only by discovery
    if (pt == PAWN) attacks = a.pawn(us, to);
    else if (pt == KNIGHT) attacks = a.knight(to);
    else if (pt == BISHOP) attacks = a.bishop(to, occupied);
    else if (pt == ROOK) attacks = a.rook(to, occupied);
    else if (pt == QUEEN) attacks = a.bishop(to, occupied) | a.rook(to, occupied);
    if (attacks & bit(king)) return true;
    return slider_sees_king(b, a, us, king, from, occupied);
}

namespace detail {

// Castling rights lost when a move starts or ends on square s (a king or rook
// leaving its square, or a rook being captured on it).
CB_HD constexpr u32 castling_rights_lost(Square s) {
    return (s == E1 ? u32(WHITE_OO | WHITE_OOO) : 0u) | (s == H1 ? u32(WHITE_OO) : 0u) |
           (s == A1 ? u32(WHITE_OOO) : 0u) | (s == E8 ? u32(BLACK_OO | BLACK_OOO) : 0u) |
           (s == H8 ? u32(BLACK_OO) : 0u) | (s == A8 ? u32(BLACK_OOO) : 0u);
}

// The rook's start and end squares for a castling move by color c.
CB_HD constexpr Square castling_rook_from(Color c, bool king_side) {
    return Square((c == WHITE ? 0 : 56) + (king_side ? 7 : 0));
}
CB_HD constexpr Square castling_rook_to(Color c, bool king_side) {
    return Square((c == WHITE ? 0 : 56) + (king_side ? 5 : 3));
}

// For en passant: the square of the captured pawn, one rank behind the
// landing square (rank 6 -> 5 for white, rank 3 -> 4 for black).
CB_HD constexpr Square en_passant_victim(Square to) { return Square(to ^ 8); }

}  // namespace detail

// Receives every piece change make_move and unmake_move perform, so state
// kept next to the board (hash key, evaluation) can follow incrementally.
// Promotions report the pawn's removal and the new piece's arrival.
struct NoObserver {
    CB_HD void add(Color, PieceType, Square) {}
    CB_HD void remove(Color, PieceType, Square) {}
};

namespace detail {

// make_move with the moving and captured piece types already known: 16-bit
// moves look them up (make_move below), 32-bit moves carry them
// (core/move32.h). captured is a pawn for en passant, NO_PIECE_TYPE for
// non-captures.
template <class Board, class Observer>
CB_HD Undo make_move_known(Board& b, Move m, PieceType moving, PieceType captured, Observer& obs) {
    const u32 st = b.state();
    const Color us = state::side(st);
    const Color them = ~us;
    const Square from = m.from();
    const Square to = m.to();
    const u32 flags = m.flags();
    const Square capture_square = flags == EP_CAPTURE ? en_passant_victim(to) : to;
    const Undo undo(st, captured);

    if (captured != NO_PIECE_TYPE) {
        b.toggle(them, captured, bit(capture_square));
        obs.remove(them, captured, capture_square);
    }
    b.toggle(us, moving, bit(from) | bit(to));
    obs.remove(us, moving, from);
    if (m.is_promotion()) {
        b.toggle(us, PAWN, bit(to));
        b.toggle(us, m.promotion_type(), bit(to));
        obs.add(us, m.promotion_type(), to);
    } else {
        obs.add(us, moving, to);
        if (flags == KING_CASTLE || flags == QUEEN_CASTLE) {
            const bool king_side = flags == KING_CASTLE;
            const Square rook_from = castling_rook_from(us, king_side);
            const Square rook_to = castling_rook_to(us, king_side);
            b.toggle(us, ROOK, bit(rook_from) | bit(rook_to));
            obs.remove(us, ROOK, rook_from);
            obs.add(us, ROOK, rook_to);
        }
    }

    const u32 rights = state::castling(st) & ~(castling_rights_lost(from) | castling_rights_lost(to));
    const bool reset_clock = moving == PAWN || captured != NO_PIECE_TYPE;
    const int halfmove = reset_clock ? 0 : state::halfmove(st) + 1;
    b.set_state(state::pack(rights, flags == DOUBLE_PUSH, file_of(from), them, halfmove));
    return undo;
}

// unmake_move with the moved piece type known (a pawn for promotions).
template <class Board, class Observer>
CB_HD void unmake_move_known(Board& b, Move m, PieceType moved, Undo undo, Observer& obs) {
    const u32 st = undo.prev_state();
    const Color us = state::side(st);
    const Color them = ~us;
    const Square from = m.from();
    const Square to = m.to();
    const u32 flags = m.flags();
    b.set_state(st);

    if (m.is_promotion()) {
        b.toggle(us, m.promotion_type(), bit(to));
        b.toggle(us, PAWN, bit(to));
        obs.remove(us, m.promotion_type(), to);
    } else {
        obs.remove(us, moved, to);
        if (flags == KING_CASTLE || flags == QUEEN_CASTLE) {
            const bool king_side = flags == KING_CASTLE;
            const Square rook_from = castling_rook_from(us, king_side);
            const Square rook_to = castling_rook_to(us, king_side);
            b.toggle(us, ROOK, bit(rook_from) | bit(rook_to));
            obs.remove(us, ROOK, rook_to);
            obs.add(us, ROOK, rook_from);
        }
    }
    b.toggle(us, moved, bit(from) | bit(to));
    obs.add(us, moved, from);

    const PieceType captured = undo.captured();
    if (captured != NO_PIECE_TYPE) {
        const Square capture_square = flags == EP_CAPTURE ? en_passant_victim(to) : to;
        b.toggle(them, captured, bit(capture_square));
        obs.add(them, captured, capture_square);
    }
}

}  // namespace detail

// Plays a legal move. Returns what unmake_move needs to take it back.
template <class Board, class Observer>
CB_HD Undo make_move(Board& b, Move m, Observer& obs) {
    const PieceType captured =
        m.flags() == EP_CAPTURE ? PAWN : m.is_capture() ? b.piece_type_on(m.to()) : NO_PIECE_TYPE;
    return detail::make_move_known(b, m, b.piece_type_on(m.from()), captured, obs);
}

template <class Board>
CB_HD Undo make_move(Board& b, Move m) {
    NoObserver obs;
    return make_move(b, m, obs);
}

// Takes back m, which must be the last move played on b.
template <class Board, class Observer>
CB_HD void unmake_move(Board& b, Move m, Undo undo, Observer& obs) {
    detail::unmake_move_known(b, m, m.is_promotion() ? PAWN : b.piece_type_on(m.to()), undo, obs);
}

template <class Board>
CB_HD void unmake_move(Board& b, Move m, Undo undo) {
    NoObserver obs;
    unmake_move(b, m, undo, obs);
}

// Passes the move to the other side (null-move pruning): no piece moves, the
// en passant square goes away, and the halfmove clock restarts at 0, so
// repetition detection never looks back across the pass (it only compares
// positions within the clock's reach). Never call it in check.
template <class Board>
CB_HD Undo make_null_move(Board& b) {
    const u32 st = b.state();
    b.set_state(state::pack(state::castling(st), false, 0, ~state::side(st), 0));
    return Undo(st, NO_PIECE_TYPE);
}

template <class Board>
CB_HD void unmake_null_move(Board& b, Undo undo) {
    b.set_state(undo.prev_state());
}

}  // namespace cb

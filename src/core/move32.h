#pragma once

// 32-bit moves (docs/ARCHITECTURE.md, Move generation): the 16-bit move
// plus the moving and the captured piece types, filled in when the move is
// generated, so make_move and unmake_move skip their board lookups. Twice the
// memory per move. A perft variant (MovesKind: CLI --moves, CMake
// CHESSGPU_MOVES); the search keeps 16-bit moves.
//
// Layout: bits 0-15 the 16-bit move, 16-19 the moving piece type, 20-23 the
// captured piece type (NO_PIECE_TYPE for non-captures; a pawn for en passant).

#include "core/movegen.h"

namespace cb {

class Move32 {
  public:
    CB_HD constexpr Move32() : v_(0) {}
    CB_HD constexpr Move32(Move m, PieceType moving, PieceType captured)
        : v_(u32(m.raw()) | (u32(moving) << 16) | (u32(captured) << 20)) {}

    CB_HD constexpr Move move() const { return Move::from_raw(u16(v_)); }
    CB_HD constexpr PieceType moving() const { return PieceType((v_ >> 16) & 15); }
    CB_HD constexpr PieceType captured() const { return PieceType((v_ >> 20) & 15); }

    // The 16-bit move's interface.
    CB_HD constexpr Square from() const { return move().from(); }
    CB_HD constexpr Square to() const { return move().to(); }
    CB_HD constexpr u32 flags() const { return move().flags(); }
    CB_HD constexpr bool is_capture() const { return move().is_capture(); }
    CB_HD constexpr bool is_promotion() const { return move().is_promotion(); }
    CB_HD constexpr PieceType promotion_type() const { return move().promotion_type(); }
    CB_HD constexpr u32 raw() const { return v_; }
    CB_HD constexpr bool is_none() const { return u16(v_) == 0; }

    CB_HD constexpr bool operator==(Move32 o) const { return v_ == o.v_; }
    CB_HD constexpr bool operator!=(Move32 o) const { return v_ != o.v_; }

  private:
    u32 v_;
};

// Plays and takes back a 32-bit move: as for the 16-bit one, with the pieces
// the move carries instead of looking them up (core/board.h).
template <class Board, class Observer>
CB_HD Undo make_move(Board& b, Move32 m, Observer& obs) {
    return detail::make_move_known(b, m.move(), m.moving(), m.captured(), obs);
}

template <class Board>
CB_HD Undo make_move(Board& b, Move32 m) {
    NoObserver obs;
    return make_move(b, m, obs);
}

template <class Board, class Observer>
CB_HD void unmake_move(Board& b, Move32 m, Undo undo, Observer& obs) {
    detail::unmake_move_known(b, m.move(), m.moving(), undo, obs);
}

template <class Board>
CB_HD void unmake_move(Board& b, Move32 m, Undo undo) {
    NoObserver obs;
    unmake_move(b, m, undo, obs);
}

// Fixed-capacity buffer of 32-bit moves, a move generation sink like
// MoveBuffer. It reads the pieces off the board the moves are generated on
// (bind() before each generation): the moving piece once per piece (its
// targets share it), the captured piece once per capture.
template <class Board, int kCapacity>
struct MoveBuffer32 {
    static constexpr int capacity = kCapacity;
    using MoveType = Move32;

    Move32 moves[kCapacity];
    int size = 0;
    const Board* board = nullptr;

    // Empties the buffer for moves of b.
    CB_HD void bind(const Board& b) {
        board = &b;
        size = 0;
    }

    // Castling and en passant (the other moves come in sets below).
    CB_HD void add(Move m) {
        const PieceType captured =
            m.flags() == EP_CAPTURE ? PAWN : m.is_capture() ? board->piece_type_on(m.to()) : NO_PIECE_TYPE;
        moves[size++] = Move32(m, board->piece_type_on(m.from()), captured);
    }

    CB_HD void add_piece_moves(Square from, u64 targets, u64 enemies) {
        const PieceType moving = board->piece_type_on(from);
        while (targets) {
            const Square to = Square(pop_lsb(targets));
            const bool capture = (bit(to) & enemies) != 0;
            moves[size++] = Move32(Move(from, to, capture ? CAPTURE : QUIET), moving,
                                   capture ? board->piece_type_on(to) : NO_PIECE_TYPE);
        }
    }

    CB_HD void add_pawn_moves(u64 targets, int offset, u32 flags) {
        while (targets) {
            const Square to = Square(pop_lsb(targets));
            moves[size++] = Move32(Move(Square(to - offset), to, flags), PAWN,
                                   flags == CAPTURE ? board->piece_type_on(to) : NO_PIECE_TYPE);
        }
    }

    CB_HD void add_promotions(u64 targets, int offset, bool capture) {
        const u32 first = capture ? PROMO_CAPTURE_KNIGHT : PROMO_KNIGHT;
        while (targets) {
            const Square to = Square(pop_lsb(targets));
            const Square from = Square(to - offset);
            const PieceType captured = capture ? board->piece_type_on(to) : NO_PIECE_TYPE;
            for (u32 p = 0; p < 4; ++p) moves[size++] = Move32(Move(from, to, first + p), PAWN, captured);
        }
    }
};

// The two encodings as template parameters of perft (kernels and CPU).
struct Moves16 {
    static constexpr const char* kName = "16";
    using MoveType = Move;
    template <class Board, int kCapacity = MoveList::capacity>
    using Buffer = MoveBuffer<kCapacity>;
};
struct Moves32 {
    static constexpr const char* kName = "32";
    using MoveType = Move32;
    template <class Board, int kCapacity = MoveList::capacity>
    using Buffer = MoveBuffer32<Board, kCapacity>;
};

// The 16-bit move of either encoding.
CB_HD constexpr Move move16(Move m) { return m; }
CB_HD constexpr Move move16(Move32 m) { return m.move(); }

}  // namespace cb

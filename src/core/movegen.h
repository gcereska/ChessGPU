#pragma once

// Legal move generation, written once against the board interface and an
// attacks policy (e.g. HqAttacks).
//
// Moves are reported to a sink through four callbacks, so a counting sink can
// popcount whole target sets instead of expanding individual moves:
//   add(m)                                   one move
//   add_piece_moves(from, targets, enemies)  one piece's moves; targets in `enemies` are captures
//   add_pawn_moves(targets, offset, flags)   pawn moves with from = to - offset
//   add_promotions(targets, offset, capture) four promotions per target, from = to - offset

#include "core/board.h"

namespace cb {

// Fixed-capacity move buffer. No position has more than 218 legal moves, and
// no single piece more than 27.
template <int kCapacity>
struct MoveBuffer {
    static constexpr int capacity = kCapacity;
    using MoveType = Move;

    Move moves[kCapacity];
    int size = 0;

    // Empties the buffer for moves of a board (MoveBuffer32 reads pieces off it).
    template <class Board>
    CB_HD void bind(const Board&) {
        size = 0;
    }

    CB_HD void add(Move m) { moves[size++] = m; }

    CB_HD void add_piece_moves(Square from, u64 targets, u64 enemies) {
        while (targets) {
            const Square to = Square(pop_lsb(targets));
            add(Move(from, to, (bit(to) & enemies) ? CAPTURE : QUIET));
        }
    }

    CB_HD void add_pawn_moves(u64 targets, int offset, u32 flags) {
        while (targets) {
            const Square to = Square(pop_lsb(targets));
            add(Move(Square(to - offset), to, flags));
        }
    }

    CB_HD void add_promotions(u64 targets, int offset, bool capture) {
        const u32 first = capture ? PROMO_CAPTURE_KNIGHT : PROMO_KNIGHT;
        while (targets) {
            const Square to = Square(pop_lsb(targets));
            const Square from = Square(to - offset);
            for (u32 p = 0; p < 4; ++p) add(Move(from, to, first + p));
        }
    }
};

using MoveList = MoveBuffer<256>;

// Counts moves without storing them.
struct MoveCounter {
    u32 count = 0;

    CB_HD void add(Move) { ++count; }
    CB_HD void add_piece_moves(Square, u64 targets, u64) { count += u32(popcount(targets)); }
    CB_HD void add_pawn_moves(u64 targets, int, u32) { count += u32(popcount(targets)); }
    CB_HD void add_promotions(u64 targets, int, bool) { count += 4 * u32(popcount(targets)); }
};

// Facts about the position that legal move generation needs, computed once.
struct LegalContext {
    u64 occupied;
    u64 our;        // pieces of the side to move
    u64 their;      // pieces of the opponent
    u64 checkers;   // opponent pieces giving check
    u64 checkmask;  // squares non-king moves may land on: everything when not in check, the
                    // checker plus the squares between it and our king in single check,
                    // nothing in double check
    u64 pinned;     // our pieces pinned to our king
    u64 danger;     // squares the opponent attacks, computed with our king removed so the
                    // king can't step back along a checking line
    Square king;    // our king
    Color us;
};

template <class Board, class Attacks>
CB_HD LegalContext make_legal_context(const Board& b, const Attacks& a) {
    LegalContext c;
    c.us = state::side(b.state());
    const Color them = ~c.us;
    c.occupied = b.occupied();
    c.our = b.color(c.us);
    c.their = b.color(them);
    const u64 kings = b.template pieces<KING>();
    c.king = Square(lsb(kings & c.our));

    const u64 their_pawns = b.template pieces<PAWN>() & c.their;
    const u64 their_knights = b.template pieces<KNIGHT>() & c.their;
    const u64 their_diagonal = b.diagonal_sliders() & c.their;
    const u64 their_straight = b.straight_sliders() & c.their;

    c.checkers = (a.pawn(c.us, c.king) & their_pawns) | (a.knight(c.king) & their_knights) |
                 (a.bishop(c.king, c.occupied) & their_diagonal) | (a.rook(c.king, c.occupied) & their_straight);

    const u64 occupied_without_king = c.occupied ^ bit(c.king);
    u64 danger = pawn_attacks_set(them, their_pawns) | knight_attacks_set(their_knights) |
                 a.king(Square(lsb(kings & c.their)));
    for (u64 s = their_diagonal; s;) danger |= a.bishop(Square(pop_lsb(s)), occupied_without_king);
    for (u64 s = their_straight; s;) danger |= a.rook(Square(pop_lsb(s)), occupied_without_king);
    c.danger = danger;

    if (c.checkers == 0) {
        c.checkmask = ~0ull;
    } else if ((c.checkers & (c.checkers - 1)) == 0) {
        c.checkmask = a.between(c.king, Square(lsb(c.checkers))) | c.checkers;
    } else {
        c.checkmask = 0;
    }

    // A piece is pinned when it is the only piece between our king and an
    // enemy slider that would otherwise attack the king.
    u64 pinned = 0;
    u64 snipers =
        (a.rook_empty_board(c.king) & their_straight) | (a.bishop_empty_board(c.king) & their_diagonal);
    while (snipers) {
        const Square s = Square(pop_lsb(snipers));
        const u64 blockers = a.between(c.king, s) & c.occupied;
        if (blockers && (blockers & (blockers - 1)) == 0) pinned |= blockers & c.our;
    }
    c.pinned = pinned;
    return c;
}

// True when c leaves only king moves (double check).
CB_HD bool only_king_moves(const LegalContext& c) { return (c.checkers & (c.checkers - 1)) != 0; }

template <class Board, class Attacks, class Sink>
CB_HD void generate_king_moves(const Board& b, const Attacks& a, const LegalContext& c, Sink& sink) {
    sink.add_piece_moves(c.king, a.king(c.king) & ~c.our & ~c.danger, c.their);
    if (c.checkers) return;

    // Castling rights guarantee the king and rook are on their start squares.
    // The squares the king crosses and lands on must be safe; `danger` was
    // computed without our king, which is fine here: an attack that only
    // reaches them through the king's square would mean we are in check.
    const u32 rights = state::castling(b.state());
    const int base = c.us == WHITE ? 0 : 56;
    const u32 king_side = c.us == WHITE ? WHITE_OO : BLACK_OO;
    const u32 queen_side = c.us == WHITE ? WHITE_OOO : BLACK_OOO;
    const u64 king_side_path = (bit(F1) | bit(G1)) << base;
    const u64 queen_side_empty = (bit(B1) | bit(C1) | bit(D1)) << base;
    const u64 queen_side_safe = (bit(C1) | bit(D1)) << base;
    if ((rights & king_side) && !(c.occupied & king_side_path) && !(c.danger & king_side_path))
        sink.add(Move(c.king, Square(G1 + base), KING_CASTLE));
    if ((rights & queen_side) && !(c.occupied & queen_side_empty) && !(c.danger & queen_side_safe))
        sink.add(Move(c.king, Square(C1 + base), QUEEN_CASTLE));
}

namespace detail {

// Splits pawn targets into ordinary moves and promotions.
template <class Sink>
CB_HD void emit_pawn_targets(Sink& sink, u64 targets, int offset, bool capture, u64 last_rank) {
    sink.add_pawn_moves(targets & ~last_rank, offset, capture ? CAPTURE : QUIET);
    sink.add_promotions(targets & last_rank, offset, capture);
}

}  // namespace detail

// Pushes and captures (not en passant) of the pawns in `pawns`, all at once,
// landing only on `allowed`.
template <class Sink>
CB_HD void generate_pawn_pushes_and_captures(const LegalContext& c, u64 pawns, u64 allowed, Sink& sink) {
    const Color us = c.us;
    const u64 empty = ~c.occupied;
    const u64 last_rank = us == WHITE ? kRank8 : kRank1;
    const u64 double_push_rank = us == WHITE ? rank_mask(3) : rank_mask(4);
    const int up = forward_offset(us);

    const u64 single = shift_forward(pawns, us) & empty;
    const u64 double_push = shift_forward(single, us) & empty & double_push_rank;
    const u64 west = shift_forward_west(pawns, us) & c.their;
    const u64 east = shift_forward_east(pawns, us) & c.their;

    detail::emit_pawn_targets(sink, single & allowed, up, false, last_rank);
    sink.add_pawn_moves(double_push & allowed, 2 * up, DOUBLE_PUSH);
    detail::emit_pawn_targets(sink, west & allowed, forward_west_offset(us), true, last_rank);
    detail::emit_pawn_targets(sink, east & allowed, forward_east_offset(us), true, last_rank);
}

// En passant is checked by playing it out: it removes two pieces from one
// rank at once, which pin masks don't capture.
template <class Board, class Attacks>
CB_HD bool en_passant_is_legal(const Board& b, const Attacks& a, const LegalContext& c, Square from, Square to) {
    const Square victim = detail::en_passant_victim(to);
    const u64 occupied = (c.occupied ^ bit(from) ^ bit(victim)) | bit(to);
    const u64 their = c.their ^ bit(victim);
    const u64 attackers = (a.pawn(c.us, c.king) & b.template pieces<PAWN>() & their) |
                          (a.knight(c.king) & b.template pieces<KNIGHT>() & their) |
                          (a.bishop(c.king, occupied) & b.diagonal_sliders() & their) |
                          (a.rook(c.king, occupied) & b.straight_sliders() & their);
    return attackers == 0;
}

// En passant captures by the pawns in `pawns`.
template <class Board, class Attacks, class Sink>
CB_HD void generate_en_passant(const Board& b, const Attacks& a, const LegalContext& c, u64 pawns, Sink& sink) {
    const u32 st = b.state();
    if (!state::has_ep(st)) return;
    const Square to = state::ep_square(st);
    // Our pawns that attack `to` stand where an opposing pawn on `to` would attack.
    for (u64 from_set = a.pawn(~c.us, to) & pawns; from_set;) {
        const Square from = Square(pop_lsb(from_set));
        if (en_passant_is_legal(b, a, c, from, to)) sink.add(Move(from, to, EP_CAPTURE));
    }
}

template <class Board, class Attacks, class Sink>
CB_HD void generate_pawn_moves(const Board& b, const Attacks& a, const LegalContext& c, Sink& sink) {
    const u64 pawns = b.template pieces<PAWN>() & c.our;
    // Pawns that aren't pinned, all at once.
    generate_pawn_pushes_and_captures(c, pawns & ~c.pinned, c.checkmask, sink);
    // Pinned pawns one at a time: they may only move along their pin line.
    for (u64 p = pawns & c.pinned; p;) {
        const Square from = Square(pop_lsb(p));
        generate_pawn_pushes_and_captures(c, bit(from), c.checkmask & a.line(c.king, from), sink);
    }
    generate_en_passant(b, a, c, pawns, sink);
}

template <class Board, class Attacks, class Sink>
CB_HD void generate_knight_moves(const Board& b, const Attacks& a, const LegalContext& c, Sink& sink) {
    const u64 allowed = ~c.our & c.checkmask;
    // A pinned knight can never move: no knight jump stays on a line.
    for (u64 n = b.template pieces<KNIGHT>() & c.our & ~c.pinned; n;) {
        const Square from = Square(pop_lsb(n));
        sink.add_piece_moves(from, a.knight(from) & allowed, c.their);
    }
}

// Queens are included in both loops; their diagonal and straight moves never overlap.
template <class Board, class Attacks, class Sink>
CB_HD void generate_slider_moves(const Board& b, const Attacks& a, const LegalContext& c, Sink& sink) {
    const u64 allowed = ~c.our & c.checkmask;
    for (u64 s = b.diagonal_sliders() & c.our; s;) {
        const Square from = Square(pop_lsb(s));
        const u64 pin_line = (bit(from) & c.pinned) ? a.line(c.king, from) : ~0ull;
        sink.add_piece_moves(from, a.bishop(from, c.occupied) & allowed & pin_line, c.their);
    }
    for (u64 s = b.straight_sliders() & c.our; s;) {
        const Square from = Square(pop_lsb(s));
        const u64 pin_line = (bit(from) & c.pinned) ? a.line(c.king, from) : ~0ull;
        sink.add_piece_moves(from, a.rook(from, c.occupied) & allowed & pin_line, c.their);
    }
}

// All legal moves of the side to move.
template <class Board, class Attacks, class Sink>
CB_HD void generate_legal(const Board& b, const Attacks& a, Sink& sink) {
    const LegalContext c = make_legal_context(b, a);
    generate_king_moves(b, a, c, sink);
    if (only_king_moves(c)) return;
    generate_pawn_moves(b, a, c, sink);
    generate_knight_moves(b, a, c, sink);
    generate_slider_moves(b, a, c, sink);
}

// Legal moves of the single piece of ours on `from` (the king's include castling).
template <class Board, class Attacks, class Sink>
CB_HD void generate_moves_from(const Board& b, const Attacks& a, const LegalContext& c, Square from, Sink& sink) {
    const PieceType pt = b.piece_type_on(from);
    if (pt == KING) {
        generate_king_moves(b, a, c, sink);
        return;
    }
    if (only_king_moves(c)) return;
    const u64 pin_line = (bit(from) & c.pinned) ? a.line(c.king, from) : ~0ull;
    const u64 allowed = c.checkmask & pin_line;
    switch (pt) {
        case PAWN:
            generate_pawn_pushes_and_captures(c, bit(from), allowed, sink);
            generate_en_passant(b, a, c, bit(from), sink);
            break;
        case KNIGHT:
            sink.add_piece_moves(from, a.knight(from) & ~c.our & allowed, c.their);
            break;
        case BISHOP:
            sink.add_piece_moves(from, a.bishop(from, c.occupied) & ~c.our & allowed, c.their);
            break;
        case ROOK:
            sink.add_piece_moves(from, a.rook(from, c.occupied) & ~c.our & allowed, c.their);
            break;
        case QUEEN:
            sink.add_piece_moves(from, a.queen(from, c.occupied) & ~c.our & allowed, c.their);
            break;
        default:
            break;
    }
}

// The same moves as generate_legal, produced one piece at a time (how the
// warp model splits generation across lanes).
template <class Board, class Attacks, class Sink>
CB_HD void generate_legal_per_piece(const Board& b, const Attacks& a, Sink& sink) {
    const LegalContext c = make_legal_context(b, a);
    for (u64 pieces = only_king_moves(c) ? bit(c.king) : c.our; pieces;)
        generate_moves_from(b, a, c, Square(pop_lsb(pieces)), sink);
}

template <class Board, class Attacks>
CB_HD u32 count_legal(const Board& b, const Attacks& a) {
    MoveCounter counter;
    generate_legal(b, a, counter);
    return counter.count;
}

}  // namespace cb

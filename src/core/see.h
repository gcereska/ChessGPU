#pragma once

// Static exchange evaluation (SEE), and the quiescence pruning built on it
// and on piece values (feature sets with kQPrune; docs/ARCHITECTURE.md,
// Search):
// - delta pruning: a capture whose victim (plus a promotion's gain) and a
//   margin can't lift the stand pat to alpha is skipped;
// - SEE pruning: a capture that loses material when both sides keep
//   capturing on its square is skipped.

#include "core/board.h"
#include "core/features.h"
#include "core/score.h"

namespace cb {

// Piece values for exchanges: the material evaluation's, with the king worth
// more than everything else (it only ever captures last).
CB_HD constexpr int see_value(int pt) {
    return pt == PAWN     ? 100
           : pt == KNIGHT ? 320
           : pt == BISHOP ? 330
           : pt == ROOK   ? 500
           : pt == QUEEN  ? 900
           : pt == KING   ? 20000
                          : 0;
}

// What capture or promotion m takes off the board for the side to move:
// the victim's value plus a promotion's gain (en passant takes a pawn).
template <class Board>
CB_HD int material_gain(const Board& b, Move m) {
    int gain = m.flags() == EP_CAPTURE ? see_value(PAWN) : (m.is_capture() ? see_value(b.piece_type_on(m.to())) : 0);
    if (m.is_promotion()) gain += see_value(m.promotion_type()) - see_value(PAWN);
    return gain;
}

// The material the side to move wins with capture or promotion m (negative:
// loses) when both sides then keep capturing on m's target square, each with
// its least valuable piece (the lowest square among equals) and each free to
// stop: the swap-list algorithm. Pins are ignored; a king captures only onto
// a square the other side no longer attacks; a pawn recapturing onto the last
// rank stays a pawn.
template <class Board, class Attacks>
CB_HD int see(const Board& b, const Attacks& a, Move m) {
    const Square to = m.to();
    const Square from = m.from();
    const Color us = state::side(b.state());
    u64 occupied = b.occupied() ^ bit(from);
    if (m.flags() == EP_CAPTURE) occupied ^= bit(Square(int(to) - forward_offset(us)));
    // gain[d]: the balance, from the point of view of the side making the
    // d-th capture, if the exchange stops right after it.
    int gain[32];
    gain[0] = material_gain(b, m);
    int on_square = m.is_promotion() ? see_value(m.promotion_type()) : see_value(b.piece_type_on(from));
    u64 attackers = attackers_to(b, a, to, occupied) & occupied;
    Color side = ~us;
    int d = 0;
    while (d < 31) {
        const u64 mine = attackers & b.color(side);
        if (!mine) break;
        // The least valuable piece of `side` attacking the square.
        u64 pick = mine & b.template pieces<PAWN>();
        int pt = PAWN;
        if (!pick) {
            pick = mine & b.template pieces<KNIGHT>();
            pt = KNIGHT;
        }
        if (!pick) {
            pick = mine & b.template pieces<BISHOP>();
            pt = BISHOP;
        }
        if (!pick) {
            pick = mine & b.template pieces<ROOK>();
            pt = ROOK;
        }
        if (!pick) {
            pick = mine & b.template pieces<QUEEN>();
            pt = QUEEN;
        }
        if (!pick) {
            pick = mine;
            pt = KING;
        }
        pick &= 0 - pick;  // the lowest square
        occupied ^= pick;
        if (pt == KING && (attackers_to(b, a, to, occupied) & occupied & b.color(~side))) break;
        ++d;
        gain[d] = on_square - gain[d - 1];
        on_square = see_value(pt);
        // Sliders behind the piece that just captured now reach the square.
        attackers = (attackers | (a.bishop(to, occupied) & b.diagonal_sliders()) |
                     (a.rook(to, occupied) & b.straight_sliders())) &
                    occupied;
        side = ~side;
    }
    // Back up: each side either stops or makes its capture, whichever is better for it.
    for (; d > 0; --d) gain[d - 1] = -(-gain[d - 1] > gain[d] ? -gain[d - 1] : gain[d]);
    return gain[0];
}

// Quiescence pruning (feature sets with kQPrune), for a capture or queen
// promotion m at a quiescence node not in check, whose stand pat is `stand`:
// the bound m's score can't exceed by delta pruning (kScoreInfinite if m may
// be searched), to be compared with alpha. SEE pruning returns -kScoreInfinite:
// skip m without a bound.
template <class Board, class Attacks>
CB_HD int quiescence_prune_bound(const Board& b, const Attacks& a, const PruneParams& prune, Move m, int stand,
                                 int alpha) {
    if (prune.delta_margin > 0) {
        const int bound = stand + material_gain(b, m) + prune.delta_margin;
        if (bound <= alpha) return bound;
    }
    if (prune.see_prune && see(b, a, m) < 0) return -kScoreInfinite;
    return kScoreInfinite;
}

}  // namespace cb

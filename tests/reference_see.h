#pragma once

// Static exchange evaluation written again on the reference board
// (tests/reference_chess.h), recursively and sharing nothing with
// core/see.h but the definition: both sides keep capturing on the target
// square with their least valuable attacker (the lowest square among
// equals), each free to stop; pins are ignored; a king captures only onto a
// square the other side doesn't attack once it has left its own; a pawn
// recapturing onto the last rank stays a pawn.

#include <algorithm>
#include <climits>
#include <vector>

#include "reference_chess.h"

namespace refsee {

using cb::Color;
using cb::Move;
using cb::PieceType;

inline int value(PieceType pt) {
    switch (pt) {
        case cb::PAWN: return 100;
        case cb::KNIGHT: return 320;
        case cb::BISHOP: return 330;
        case cb::ROOK: return 500;
        case cb::QUEEN: return 900;
        case cb::KING: return 20000;
        default: return 0;
    }
}

// Squares of c's pieces attacking square t on b (removed pieces are empty
// squares, so sliders behind them attack through).
inline std::vector<int> attackers(const ref::Board& b, int t, Color c) {
    std::vector<int> out;
    const int tf = t % 8;
    const int tr = t / 8;
    const int pawn_rank = tr - (c == cb::WHITE ? 1 : -1);
    for (int df : {-1, 1})
        if (ref::on_board(tf + df, pawn_rank) && b.sq[pawn_rank * 8 + tf + df] == ref::code(c, cb::PAWN))
            out.push_back(pawn_rank * 8 + tf + df);
    for (const auto& s : ref::kKnightSteps)
        if (ref::on_board(tf + s[0], tr + s[1]) && b.sq[(tr + s[1]) * 8 + tf + s[0]] == ref::code(c, cb::KNIGHT))
            out.push_back((tr + s[1]) * 8 + tf + s[0]);
    for (const auto& s : ref::kKingSteps)
        if (ref::on_board(tf + s[0], tr + s[1]) && b.sq[(tr + s[1]) * 8 + tf + s[0]] == ref::code(c, cb::KING))
            out.push_back((tr + s[1]) * 8 + tf + s[0]);
    const auto slide = [&](const int (&dirs)[4][2], PieceType slider) {
        for (const auto& d : dirs) {
            for (int f = tf + d[0], r = tr + d[1]; ref::on_board(f, r); f += d[0], r += d[1]) {
                const int code = b.sq[r * 8 + f];
                if (code == 0) continue;
                if (code == ref::code(c, slider) || code == ref::code(c, cb::QUEEN)) out.push_back(r * 8 + f);
                break;
            }
        }
    };
    slide(ref::kRookDirs, cb::ROOK);
    slide(ref::kBishopDirs, cb::BISHOP);
    return out;
}

// What `side` gains by capturing on t, where a piece worth `on_square`
// stands (0 if it can't or shouldn't capture).
inline int exchange(ref::Board& b, int t, Color side, int on_square) {
    const std::vector<int> att = attackers(b, t, side);
    if (att.empty()) return 0;
    int from = -1;
    int v = INT_MAX;
    for (const int s : att) {
        const int sv = value(ref::type_of(b.sq[s]));
        if (sv < v || (sv == v && s < from)) {
            v = sv;
            from = s;
        }
    }
    const int code = b.sq[from];
    b.sq[from] = 0;
    int result = 0;
    if (ref::type_of(code) != cb::KING || attackers(b, t, ~side).empty())
        result = std::max(0, on_square - exchange(b, t, ~side, v));
    b.sq[from] = code;
    return result;
}

// The exchange value of capture or promotion m for the side to move.
inline int see(const cb::Position& p, Move m) {
    ref::Board b = ref::from_position(p);
    const int t = m.to();
    const int f = m.from();
    int gain = m.flags() == cb::EP_CAPTURE ? 100 : (b.sq[t] != 0 ? value(ref::type_of(b.sq[t])) : 0);
    int mover = value(ref::type_of(b.sq[f]));
    if (m.is_promotion()) {
        gain += value(m.promotion_type()) - 100;
        mover = value(m.promotion_type());
    }
    if (m.flags() == cb::EP_CAPTURE) b.sq[t - (b.side == cb::WHITE ? 8 : -8)] = 0;
    b.sq[f] = 0;
    return gain - exchange(b, t, ~b.side, mover);
}

}  // namespace refsee

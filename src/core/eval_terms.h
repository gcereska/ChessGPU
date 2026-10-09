#pragma once

// Hand-written evaluation terms on top of PeSTO (docs/ARCHITECTURE.md,
// Evaluation). Each term is counted from the board when a position
// is evaluated (white's count minus black's), weighted by a middlegame and an
// endgame value, and tapered by the game phase together with PeSTO's tables.
// The counts are linear features, so the weights can be fitted to game
// results (Texel tuning: `chessgpu tune`), which prints new weights for
// term_weight() below.
//
// Mobility takes its attack sets from an `Attacks` object: the search's own
// lookups (hyperbola quintessence or magic bitboards) inside a search, or
// TableFreeAttacks (Kogge-Stone fills, no tables) elsewhere; the counts are
// the same either way. Everything else is bitboard arithmetic, so the terms
// run unchanged on the GPU.

#include "core/attacks_kogge.h"
#include "core/eval.h"

namespace cb {

enum Term : int {
    kTermBishopPair,       // a side with two bishops or more
    kTermDoubledPawn,      // pawns beyond the first on their file
    kTermIsolatedPawn,     // pawns with no friendly pawn on an adjacent file
    kTermPassedPawn2,      // passed pawns (no enemy pawn ahead on their own or an adjacent file),
    kTermPassedPawn3,      // by rank counted from their own side
    kTermPassedPawn4,
    kTermPassedPawn5,
    kTermPassedPawn6,
    kTermPassedPawn7,
    kTermKnightMobility,   // squares each piece attacks that hold no piece of its own side and no
    kTermBishopMobility,   // enemy pawn attacks
    kTermRookMobility,
    kTermQueenMobility,
    kTermRookOpenFile,     // rooks on a file without pawns
    kTermRookSemiOpenFile, // ... without pawns of their own side
    kTermKingShield1,      // own pawns on the three squares in front of the king
    kTermKingShield2,      // ... and on the three beyond them
    kTermCount
};

// Knight and slider attack sets without tables: count_terms' default, for
// code that has no attack tables at hand (the tuner, tests).
struct TableFreeAttacks {
    CB_HD u64 knight(Square s) const { return knight_attacks_set(bit(s)); }
    CB_HD u64 bishop(Square s, u64 occupied) const { return kogge::bishop_attacks(bit(s), occupied); }
    CB_HD u64 rook(Square s, u64 occupied) const { return kogge::rook_attacks(bit(s), occupied); }
};

// Term counts of a position, white's minus black's.
struct TermCounts {
    int v[kTermCount];
};

// Weights in centipawns: middlegame (mg) and endgame.
CB_HD constexpr int term_weight(int term, bool endgame) {
    switch (term) {
        case kTermBishopPair: return endgame ? 50 : 30;
        case kTermDoubledPawn: return endgame ? -20 : -10;
        case kTermIsolatedPawn: return endgame ? -15 : -10;
        case kTermPassedPawn2: return endgame ? 5 : 0;
        case kTermPassedPawn3: return endgame ? 10 : 0;
        case kTermPassedPawn4: return endgame ? 15 : 5;
        case kTermPassedPawn5: return endgame ? 25 : 10;
        case kTermPassedPawn6: return endgame ? 40 : 20;
        case kTermPassedPawn7: return endgame ? 60 : 30;
        case kTermKnightMobility: return endgame ? 4 : 4;
        case kTermBishopMobility: return endgame ? 5 : 4;
        case kTermRookMobility: return endgame ? 4 : 2;
        case kTermQueenMobility: return endgame ? 2 : 1;
        case kTermRookOpenFile: return endgame ? 10 : 25;
        case kTermRookSemiOpenFile: return endgame ? 5 : 12;
        case kTermKingShield1: return endgame ? 0 : 10;
        case kTermKingShield2: return endgame ? 0 : 5;
        default: return 0;
    }
}

namespace terms {

CB_HD constexpr u64 fill_north(u64 b) {
    b |= b << 8;
    b |= b << 16;
    b |= b << 32;
    return b;
}
CB_HD constexpr u64 fill_south(u64 b) {
    b |= b >> 8;
    b |= b >> 16;
    b |= b >> 32;
    return b;
}
// Every square of every file that holds a square of b.
CB_HD constexpr u64 file_fill(u64 b) { return fill_north(b) | fill_south(b); }
CB_HD constexpr u64 widen(u64 b) { return b | shift_east(b) | shift_west(b); }

// One side's counts (the side whose pieces are `own`).
template <class Board, class Attacks>
CB_HD void count_side(const Board& b, const Attacks& a, Color c, int sign, TermCounts& t) {
    const u64 occupied = b.occupied();
    const u64 own = b.color(c);
    const u64 their = b.color(~c);
    const u64 pawns = b.template pieces<PAWN>();
    const u64 own_pawns = pawns & own;
    const u64 their_pawns = pawns & their;

    if (popcount(b.template pieces<BISHOP>() & own) >= 2) t.v[kTermBishopPair] += sign;

    // Pawn structure. Files are taken from the first rank of a file fill.
    const u64 own_files = file_fill(own_pawns);
    t.v[kTermDoubledPawn] += sign * (popcount(own_pawns) - popcount(own_files & kRank1));
    t.v[kTermIsolatedPawn] += sign * popcount(own_pawns & ~(shift_east(own_files) | shift_west(own_files)));
    // Passed: no enemy pawn ahead on the same or an adjacent file.
    const u64 their_front = c == WHITE ? widen(fill_south(their_pawns >> 8)) : widen(fill_north(their_pawns << 8));
    const u64 passed = own_pawns & ~their_front;
    for (int r = 1; r <= 6; ++r) {
        const int rank = c == WHITE ? r : 7 - r;
        t.v[kTermPassedPawn2 + r - 1] += sign * popcount(passed & rank_mask(rank));
    }

    // Mobility.
    const u64 safe = ~own & ~pawn_attacks_set(~c, their_pawns);
    for (u64 p = b.template pieces<KNIGHT>() & own; p;) {
        t.v[kTermKnightMobility] += sign * popcount(a.knight(Square(pop_lsb(p))) & safe);
    }
    for (u64 p = b.template pieces<BISHOP>() & own; p;) {
        t.v[kTermBishopMobility] += sign * popcount(a.bishop(Square(pop_lsb(p)), occupied) & safe);
    }
    for (u64 p = b.template pieces<ROOK>() & own; p;) {
        t.v[kTermRookMobility] += sign * popcount(a.rook(Square(pop_lsb(p)), occupied) & safe);
    }
    for (u64 p = b.template pieces<QUEEN>() & own; p;) {
        const Square s = Square(pop_lsb(p));
        t.v[kTermQueenMobility] += sign * popcount((a.rook(s, occupied) | a.bishop(s, occupied)) & safe);
    }

    // Rooks on open and semi-open files.
    const u64 rooks = b.template pieces<ROOK>() & own;
    const u64 pawn_files = file_fill(pawns);
    t.v[kTermRookOpenFile] += sign * popcount(rooks & ~pawn_files);
    t.v[kTermRookSemiOpenFile] += sign * popcount(rooks & pawn_files & ~own_files);

    // King shield.
    const u64 king = b.template pieces<KING>() & own;
    const u64 shield1 = widen(shift_forward(king, c));
    const u64 shield2 = shift_forward(shield1, c);
    t.v[kTermKingShield1] += sign * popcount(own_pawns & shield1);
    t.v[kTermKingShield2] += sign * popcount(own_pawns & shield2);
}

}  // namespace terms

template <class Board, class Attacks = TableFreeAttacks>
CB_HD TermCounts count_terms(const Board& b, const Attacks& a = Attacks{}) {
    TermCounts t{};
    terms::count_side(b, a, WHITE, 1, t);
    terms::count_side(b, a, BLACK, -1, t);
    return t;
}

// PeSTO plus the terms above.
struct PestoTermsEval : PestoEval {
    static constexpr const char* kName = "terms";

    template <class Board>
    CB_HD static int evaluate(const Board& b, const Accumulator& acc, Color side_to_move) {
        return evaluate(b, acc, side_to_move, TableFreeAttacks{});
    }

    template <class Board, class Attacks>
    CB_HD static int evaluate(const Board& b, const Accumulator& acc, Color side_to_move, const Attacks& a) {
        const TermCounts t = count_terms(b, a);
        int mg = unpack_mg(acc.packed);
        int eg = unpack_eg(acc.packed);
        CB_UNROLL
        for (int i = 0; i < kTermCount; ++i) {
            mg += term_weight(i, false) * t.v[i];
            eg += term_weight(i, true) * t.v[i];
        }
        const int mg_phase = acc.phase < pesto::kPhaseMax ? acc.phase : pesto::kPhaseMax;
        const int eg_phase = pesto::kPhaseMax - mg_phase;
        const int white = (mg * mg_phase + eg * eg_phase) / pesto::kPhaseMax;
        // Truncation toward zero keeps the color-flipped position's score exactly negated.
        return side_to_move == WHITE ? white : -white;
    }
};

}  // namespace cb

#pragma once

#include "core/types.h"

namespace cb {

constexpr u64 kFileA = 0x0101010101010101ull;
constexpr u64 kFileB = kFileA << 1;
constexpr u64 kFileG = kFileA << 6;
constexpr u64 kFileH = kFileA << 7;
constexpr u64 kRank1 = 0xFFull;
constexpr u64 kRank8 = kRank1 << 56;

CB_HD constexpr u64 file_mask(int file) { return kFileA << file; }
CB_HD constexpr u64 rank_mask(int rank) { return kRank1 << (8 * rank); }

// One-step shifts. Bits that would wrap around the board edge are dropped.
CB_HD constexpr u64 shift_north(u64 b) { return b << 8; }
CB_HD constexpr u64 shift_south(u64 b) { return b >> 8; }
CB_HD constexpr u64 shift_east(u64 b) { return (b << 1) & ~kFileA; }
CB_HD constexpr u64 shift_west(u64 b) { return (b >> 1) & ~kFileH; }
CB_HD constexpr u64 shift_north_east(u64 b) { return (b << 9) & ~kFileA; }
CB_HD constexpr u64 shift_north_west(u64 b) { return (b << 7) & ~kFileH; }
CB_HD constexpr u64 shift_south_east(u64 b) { return (b >> 7) & ~kFileA; }
CB_HD constexpr u64 shift_south_west(u64 b) { return (b >> 9) & ~kFileH; }

// "Forward" as seen by color c. c is a runtime value: both directions are
// computed and one is selected, so there is no branch.
CB_HD constexpr u64 shift_forward(u64 b, Color c) { return c == WHITE ? shift_north(b) : shift_south(b); }
CB_HD constexpr u64 shift_forward_west(u64 b, Color c) {
    return c == WHITE ? shift_north_west(b) : shift_south_west(b);
}
CB_HD constexpr u64 shift_forward_east(u64 b, Color c) {
    return c == WHITE ? shift_north_east(b) : shift_south_east(b);
}

// Square offset (to - from) of each forward shift above.
CB_HD constexpr int forward_offset(Color c) { return c == WHITE ? 8 : -8; }
CB_HD constexpr int forward_west_offset(Color c) { return c == WHITE ? 7 : -9; }
CB_HD constexpr int forward_east_offset(Color c) { return c == WHITE ? 9 : -7; }

// Rank (0-7) as seen from color c's side of the board.
CB_HD constexpr int relative_rank(Color c, int rank) { return c == WHITE ? rank : 7 - rank; }

// Attack sets of whole groups of pieces at once.
CB_HD constexpr u64 pawn_attacks_set(Color c, u64 pawns) {
    return shift_forward_west(pawns, c) | shift_forward_east(pawns, c);
}

CB_HD constexpr u64 knight_attacks_set(u64 knights) {
    const u64 l1 = (knights >> 1) & ~kFileH;
    const u64 l2 = (knights >> 2) & ~(kFileG | kFileH);
    const u64 r1 = (knights << 1) & ~kFileA;
    const u64 r2 = (knights << 2) & ~(kFileA | kFileB);
    const u64 h1 = l1 | r1;
    const u64 h2 = l2 | r2;
    return (h1 << 16) | (h1 >> 16) | (h2 << 8) | (h2 >> 8);
}

CB_HD constexpr u64 king_attacks_set(u64 kings) {
    const u64 sideways = shift_east(kings) | shift_west(kings);
    const u64 row = kings | sideways;
    return sideways | shift_north(row) | shift_south(row);
}

}  // namespace cb

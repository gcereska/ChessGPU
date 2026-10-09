#pragma once

// Kogge-Stone fills for sliding pieces (docs/ARCHITECTURE.md, Move
// generation): in each direction, the piece's bit spreads over empty squares
// by 1, 2 and 4 steps (each step also doubling the reach of the "empty"
// mask), then one more step adds the first blocker. No tables, only shifts
// and masks; the same code would fill every piece of a set at once.

#include "core/attacks.h"

namespace cb {

namespace kogge {

constexpr u64 kNotFileA = ~kFileA;
constexpr u64 kNotFileH = ~kFileH;

// Shifts toward higher squares for positive kShift, lower for negative.
template <int kShift>
CB_HD constexpr u64 shift(u64 b) {
    if constexpr (kShift > 0) {
        return b << kShift;
    } else {
        return b >> -kShift;
    }
}

// Squares attacked from the set `gen` in one direction (one square further
// each step of kShift); kWrap (0 for vertical steps) masks out squares a step
// would reach by wrapping around the board's edge.
template <int kShift, u64 kWrap>
CB_HD constexpr u64 fill(u64 gen, u64 empty) {
    u64 pro = kWrap ? (empty & kWrap) : empty;
    gen |= pro & shift<kShift>(gen);
    pro &= shift<kShift>(pro);
    gen |= pro & shift<2 * kShift>(gen);
    pro &= shift<2 * kShift>(pro);
    gen |= pro & shift<4 * kShift>(gen);
    const u64 attacks = shift<kShift>(gen);
    return kWrap ? (attacks & kWrap) : attacks;
}

// Squares the rooks (or bishops) in `sliders` attack together, given the
// occupied squares. Table-free, so code without attack tables (the
// evaluation's mobility terms) can use them too.
CB_HD constexpr u64 rook_attacks(u64 sliders, u64 occupied) {
    const u64 empty = ~occupied;
    return fill<8, 0>(sliders, empty) | fill<-8, 0>(sliders, empty) | fill<1, kNotFileA>(sliders, empty) |
           fill<-1, kNotFileH>(sliders, empty);
}
CB_HD constexpr u64 bishop_attacks(u64 sliders, u64 occupied) {
    const u64 empty = ~occupied;
    return fill<9, kNotFileA>(sliders, empty) | fill<7, kNotFileH>(sliders, empty) |
           fill<-7, kNotFileA>(sliders, empty) | fill<-9, kNotFileH>(sliders, empty);
}

}  // namespace kogge

class KoggeStoneAttacks : public TableAttacks {
  public:
    static constexpr const char* kName = "kogge";

    CB_HD explicit KoggeStoneAttacks(const AttackTables& tables) : TableAttacks(tables) {}

    CB_HD u64 rook(Square s, u64 occupied) const { return kogge::rook_attacks(bit(s), occupied); }
    CB_HD u64 bishop(Square s, u64 occupied) const { return kogge::bishop_attacks(bit(s), occupied); }
    CB_HD u64 queen(Square s, u64 occupied) const { return bishop(s, occupied) | rook(s, occupied); }
};

}  // namespace cb

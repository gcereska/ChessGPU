#pragma once

#include "core/state.h"

namespace cb {

// One bitboard per piece type plus one per color: 8 x 64 bits = 16 registers.
// Piece-type masks are stored directly.
//
// Board interface (shared with QuadBoard, used by all generic code):
//   occupied(), color(c), pieces<PT>(), diagonal_sliders(), straight_sliders()
//   piece_type_on(s), toggle(c, pt, mask), state(), set_state(s)
// No field is ever selected by a runtime index, so the board can stay in
// registers: runtime choices use selects, and toggle() masks every field.
class StdBoard {
  public:
    static constexpr const char* kName = "std";

    CB_HD StdBoard() : by_type_{0, 0, 0, 0, 0, 0}, white_(0), black_(0), state_(0) {}

    CB_HD u64 occupied() const { return white_ | black_; }
    CB_HD u64 color(Color c) const { return c == WHITE ? white_ : black_; }

    // Pieces of type PT, both colors.
    template <PieceType PT>
    CB_HD u64 pieces() const {
        static_assert(PT >= PAWN && PT <= KING, "not a piece type");
        return by_type_[PT - 1];
    }
    CB_HD u64 diagonal_sliders() const { return by_type_[BISHOP - 1] | by_type_[QUEEN - 1]; }
    CB_HD u64 straight_sliders() const { return by_type_[ROOK - 1] | by_type_[QUEEN - 1]; }

    CB_HD PieceType piece_type_on(Square s) const {
        int type = 0;
        CB_UNROLL
        for (int k = 0; k < 6; ++k) type += int((by_type_[k] >> s) & 1) * (k + 1);
        return PieceType(type);
    }

    // XORs mask into the bitboards for color c and type pt: adds pieces on
    // empty squares and removes them where they stand.
    CB_HD void toggle(Color c, PieceType pt, u64 mask) {
        CB_UNROLL
        for (int k = 0; k < 6; ++k) by_type_[k] ^= mask & mask_if(pt == k + 1);
        white_ ^= mask & mask_if(c == WHITE);
        black_ ^= mask & mask_if(c == BLACK);
    }

    CB_HD u32 state() const { return state_; }
    CB_HD void set_state(u32 s) { state_ = s; }

    CB_HD bool operator==(const StdBoard& o) const {
        bool equal = white_ == o.white_ && black_ == o.black_ && state_ == o.state_;
        CB_UNROLL
        for (int k = 0; k < 6; ++k) equal = equal && by_type_[k] == o.by_type_[k];
        return equal;
    }

    // Returns a description of the first broken representation invariant, or
    // nullptr if all hold. Host only; for tests and debugging.
    const char* internal_error() const {
        u64 seen = 0;
        for (int k = 0; k < 6; ++k) {
            if (by_type_[k] & seen) return "two piece types on one square";
            seen |= by_type_[k];
        }
        if (white_ & black_) return "square marked both white and black";
        if (seen != (white_ | black_)) return "piece-type and color bitboards disagree";
        if (state_ >> 16) return "state word uses bits above 15";
        return nullptr;
    }

  private:
    u64 by_type_[6];  // index = PieceType - 1
    u64 white_;
    u64 black_;
    u32 state_;
};

}  // namespace cb

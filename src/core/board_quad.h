#pragma once

#include "core/state.h"

namespace cb {

// Bit-sliced board: every square has a 4-bit code (a color bit plus its
// 3-bit PieceType), and bit s of each of the 4 bitboards holds one bit of
// square s's code. 4 x 64 bits = 8 registers. Piece-type masks are built
// with 1-3 bit operations.
//
// Implements the same interface as StdBoard (see board_std.h).
class QuadBoard {
  public:
    static constexpr const char* kName = "quad";

    CB_HD QuadBoard() : black_(0), t0_(0), t1_(0), t2_(0), state_(0) {}

    CB_HD u64 occupied() const { return t0_ | t1_ | t2_; }
    CB_HD u64 color(Color c) const { return c == WHITE ? (occupied() & ~black_) : black_; }

    // Pieces of type PT, both colors: squares whose 3 type bits equal PT.
    template <PieceType PT>
    CB_HD u64 pieces() const {
        static_assert(PT >= PAWN && PT <= KING, "not a piece type");
        constexpr int code = PT;
        return ((code & 1) ? t0_ : ~t0_) & ((code & 2) ? t1_ : ~t1_) & ((code & 4) ? t2_ : ~t2_);
    }
    // Codes are chosen so these are cheap: bishop 011 and queen 101 are the
    // codes with t0 set and exactly one of t1/t2; rook 100 and queen 101 are
    // the codes with t2 set and t1 clear.
    CB_HD u64 diagonal_sliders() const { return t0_ & (t1_ ^ t2_); }
    CB_HD u64 straight_sliders() const { return t2_ & ~t1_; }

    CB_HD PieceType piece_type_on(Square s) const {
        return PieceType(int((t0_ >> s) & 1) | (int((t1_ >> s) & 1) << 1) | (int((t2_ >> s) & 1) << 2));
    }

    // XORs mask into the bits of piece code (c, pt): adds pieces on empty
    // squares and removes them where they stand.
    CB_HD void toggle(Color c, PieceType pt, u64 mask) {
        t0_ ^= mask & mask_if((pt & 1) != 0);
        t1_ ^= mask & mask_if((pt & 2) != 0);
        t2_ ^= mask & mask_if((pt & 4) != 0);
        black_ ^= mask & mask_if(c == BLACK);
    }

    CB_HD u32 state() const { return state_; }
    CB_HD void set_state(u32 s) { state_ = s; }

    CB_HD bool operator==(const QuadBoard& o) const {
        return black_ == o.black_ && t0_ == o.t0_ && t1_ == o.t1_ && t2_ == o.t2_ && state_ == o.state_;
    }

    // Returns a description of the first broken representation invariant, or
    // nullptr if all hold. Host only; for tests and debugging.
    const char* internal_error() const {
        if (t0_ & t1_ & t2_) return "square with type code 7";
        if (black_ & ~occupied()) return "color bit set on an empty square";
        if (state_ >> 16) return "state word uses bits above 15";
        return nullptr;
    }

  private:
    u64 black_;  // color bit: 1 = black piece, 0 = white piece or empty square
    u64 t0_;     // type bit 0
    u64 t1_;     // type bit 1
    u64 t2_;     // type bit 2
    u32 state_;
};

}  // namespace cb

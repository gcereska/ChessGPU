#pragma once

#include <cstdint>

#include "core/bits.h"

namespace cb {

using u8 = std::uint8_t;
using u16 = std::uint16_t;
using u32 = std::uint32_t;
using i16 = std::int16_t;
using i32 = std::int32_t;
using i64 = std::int64_t;

enum Color : int { WHITE = 0, BLACK = 1 };

CB_HD constexpr Color operator~(Color c) { return Color(c ^ 1); }

// Piece types start at 1 so that 0 means "no piece". QuadBoard stores these
// values directly as its 3-bit type code.
enum PieceType : int {
    NO_PIECE_TYPE = 0,
    PAWN = 1,
    KNIGHT = 2,
    BISHOP = 3,
    ROOK = 4,
    QUEEN = 5,
    KING = 6,
};

// Squares are numbered rank * 8 + file: a1 = 0, h1 = 7, a8 = 56, h8 = 63.
// clang-format off
enum Square : int {
    A1, B1, C1, D1, E1, F1, G1, H1,
    A2, B2, C2, D2, E2, F2, G2, H2,
    A3, B3, C3, D3, E3, F3, G3, H3,
    A4, B4, C4, D4, E4, F4, G4, H4,
    A5, B5, C5, D5, E5, F5, G5, H5,
    A6, B6, C6, D6, E6, F6, G6, H6,
    A7, B7, C7, D7, E7, F7, G7, H7,
    A8, B8, C8, D8, E8, F8, G8, H8,
    NO_SQUARE = 64,
};
// clang-format on

CB_HD constexpr Square make_square(int file, int rank) { return Square(rank * 8 + file); }
CB_HD constexpr int file_of(Square s) { return s & 7; }
CB_HD constexpr int rank_of(Square s) { return s >> 3; }
CB_HD constexpr u64 bit(Square s) { return 1ull << s; }

// Castling rights, as stored in bits 0-3 of the state word.
enum CastlingRight : u32 {
    WHITE_OO = 1,
    WHITE_OOO = 2,
    BLACK_OO = 4,
    BLACK_OOO = 8,
};

// Move flags (bits 12-15 of a Move). Flag bit 2 (value 4) marks captures, flag
// bit 3 (value 8) marks promotions, and the low 2 bits of a promotion pick the
// piece, so each property is a single AND.
enum MoveFlag : u32 {
    QUIET = 0,
    DOUBLE_PUSH = 1,
    KING_CASTLE = 2,
    QUEEN_CASTLE = 3,
    CAPTURE = 4,
    EP_CAPTURE = 5,
    PROMO_KNIGHT = 8,
    PROMO_BISHOP = 9,
    PROMO_ROOK = 10,
    PROMO_QUEEN = 11,
    PROMO_CAPTURE_KNIGHT = 12,
    PROMO_CAPTURE_BISHOP = 13,
    PROMO_CAPTURE_ROOK = 14,
    PROMO_CAPTURE_QUEEN = 15,
};

constexpr u32 kCaptureFlagBit = 4;
constexpr u32 kPromotionFlagBit = 8;

// 16-bit move: bits 0-5 from square, bits 6-11 to square, bits 12-15 flags.
// The all-zero value (a1 to a1, quiet) is never a legal move and means "no move".
class Move {
  public:
    CB_HD constexpr Move() : v_(0) {}
    CB_HD constexpr Move(Square from, Square to, u32 flags)
        : v_(static_cast<u16>(u32(from) | (u32(to) << 6) | (flags << 12))) {}

    CB_HD static constexpr Move from_raw(u16 raw) {
        Move m;
        m.v_ = raw;
        return m;
    }

    CB_HD constexpr Square from() const { return Square(v_ & 63); }
    CB_HD constexpr Square to() const { return Square((v_ >> 6) & 63); }
    CB_HD constexpr u32 flags() const { return u32(v_) >> 12; }
    CB_HD constexpr bool is_capture() const { return (flags() & kCaptureFlagBit) != 0; }
    CB_HD constexpr bool is_promotion() const { return (flags() & kPromotionFlagBit) != 0; }
    // Only meaningful when is_promotion().
    CB_HD constexpr PieceType promotion_type() const { return PieceType(KNIGHT + (flags() & 3)); }
    CB_HD constexpr u16 raw() const { return v_; }
    CB_HD constexpr bool is_none() const { return v_ == 0; }

    CB_HD constexpr bool operator==(Move o) const { return v_ == o.v_; }
    CB_HD constexpr bool operator!=(Move o) const { return v_ != o.v_; }

  private:
    u16 v_;
};

}  // namespace cb

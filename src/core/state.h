#pragma once

#include "core/types.h"

namespace cb {

// The state word packs everything about a position except piece placement:
//   bits 0-3   castling rights (CastlingRight flags)
//   bits 4-6   en passant file
//   bit  7     en passant square is set
//   bit  8     side to move (0 = white, 1 = black)
//   bits 9-15  halfmove clock for the 50-move rule, saturating at 127
// Bits 16-31 are always zero.
//
// The en passant square follows FEN: it is set after every double pawn push,
// whether or not an en passant capture is actually possible.
namespace state {

constexpr u32 kCastlingMask = 0xF;
constexpr int kEpFileShift = 4;
constexpr u32 kEpSetBit = 1u << 7;
constexpr int kSideShift = 8;
constexpr int kHalfmoveShift = 9;
constexpr int kHalfmoveMax = 127;

CB_HD constexpr u32 castling(u32 s) { return s & kCastlingMask; }
CB_HD constexpr bool has_ep(u32 s) { return (s & kEpSetBit) != 0; }
CB_HD constexpr int ep_file(u32 s) { return int((s >> kEpFileShift) & 7); }
CB_HD constexpr Color side(u32 s) { return Color((s >> kSideShift) & 1); }
CB_HD constexpr int halfmove(u32 s) { return int((s >> kHalfmoveShift) & 127); }

// Square a capturing pawn lands on. Only meaningful when has_ep(s): white to
// move captures onto rank 6, black onto rank 3.
CB_HD constexpr Square ep_square(u32 s) { return make_square(ep_file(s), side(s) == WHITE ? 5 : 2); }

// ep_file is ignored unless has_ep. halfmove is clamped to [0, kHalfmoveMax].
CB_HD constexpr u32 pack(u32 castling_rights, bool has_ep, int ep_file, Color side, int halfmove) {
    const int hm = halfmove < 0 ? 0 : (halfmove > kHalfmoveMax ? kHalfmoveMax : halfmove);
    return (castling_rights & kCastlingMask) |
           (has_ep ? (kEpSetBit | (u32(ep_file & 7) << kEpFileShift)) : 0u) |
           (u32(side) << kSideShift) | (u32(hm) << kHalfmoveShift);
}

}  // namespace state

// What make_move overwrites and unmake_move can't recompute: the previous
// state word (bits 0-15) and the captured piece type (bits 16-18).
class Undo {
  public:
    CB_HD constexpr Undo() : v_(0) {}
    CB_HD constexpr Undo(u32 prev_state, PieceType captured)
        : v_((prev_state & 0xFFFFu) | (u32(captured) << 16)) {}

    CB_HD constexpr u32 prev_state() const { return v_ & 0xFFFFu; }
    CB_HD constexpr PieceType captured() const { return PieceType((v_ >> 16) & 7); }

  private:
    u32 v_;
};

}  // namespace cb

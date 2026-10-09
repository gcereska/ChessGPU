#pragma once

#include <string>

#include "host/position.h"

namespace cb {

// 8x8 grid, rank 8 at the top: 'x' for set bits, '.' for clear ones.
std::string bitboard_to_string(u64 b);

// Board diagram (white pieces uppercase) followed by the FEN.
std::string position_to_string(const Position& p);

}  // namespace cb

#pragma once

#include <string>
#include <string_view>

#include "host/position.h"

namespace cb {

inline constexpr const char* kStartFen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

// Parses a FEN string into `out` and validates the position (see
// validate_position). The halfmove clock and fullmove number are optional
// (defaults 0 and 1). A halfmove clock above 127 is stored as 127. On failure
// returns false, leaves `out` unchanged, and describes the problem in `error`.
bool parse_fen(std::string_view fen, Position& out, std::string& error);

std::string format_fen(const Position& p);

}  // namespace cb

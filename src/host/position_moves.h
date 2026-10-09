#pragma once

#include <string>
#include <string_view>

#include "host/position.h"

namespace cb {

// Plays space-separated UCI moves ("e2e4 e7e5") on p, advancing the fullmove
// number after each black move. Every move must be legal in the position it
// is played in. On failure returns false, leaves p unchanged and sets `error`.
bool apply_uci_moves(Position& p, std::string_view moves, std::string& error);

}  // namespace cb

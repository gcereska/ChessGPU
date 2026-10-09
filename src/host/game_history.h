#pragma once

#include <string>
#include <string_view>
#include <vector>

#include "core/search.h"
#include "host/position.h"

namespace cb {

// The game's positions before the search root, for repetition detection.
struct GameHistory {
    std::vector<HistoryEntry> entries;  // oldest first; the last one is the root's parent
    bool root_repeated = false;         // the root position already occurred earlier in the game

    // Entry of the root's parent, or -1 when the game starts at the root.
    int root_parent() const { return int(entries.size()) - 1; }
};

// Plays space-separated UCI `moves` from `start`. The position after the
// last move becomes `root`, and the positions before it go into `history`.
// On failure returns false, sets `error` and leaves the outputs unchanged.
bool build_game(const Position& start, std::string_view moves, Position& root, GameHistory& history,
                std::string& error);

}  // namespace cb

#pragma once

// MultiPV (UCI MultiPV): the best few root moves, each with an exact score
// and its line, built on any backend's fixed-depth search without changing
// the searches. Every root move is searched as a root of its own, one ply
// shallower: the first ones with a full window, the rest with a zero window
// around the worst line kept so far, and again (for the exact score) if they
// beat it. A search never scores its own root as a draw, so positions one ply
// after the root get the rules the search applies there (repetitions, the
// 50-move rule, insufficient material) before any search.

#include <functional>
#include <vector>

#include "host/search_backend.h"

namespace cb {

// A legal root move and the search root it leads to.
struct RootChild {
    Move move;
    Position position;
    GameHistory history;  // the positions before it: the game's and the root
    bool draw = false;    // a draw where it stands (one ply after the root): scores 0 unsearched
};

std::vector<RootChild> root_children(const Position& root, const GameHistory& history);

// A child search's score seen from the root (one ply further from any mate),
// and the child score that gives a root score.
int score_from_child(int child_score);
int score_to_child(int root_score);

// Iterative deepening reporting `request.multipv` lines per iteration (as
// IterationReport::lines, best first). A root without legal moves gets the
// normal search.
SearchOutcome multipv_deepening(SearchBackend& backend, const SearchRequest& request,
                                const std::function<void(const IterationReport&)>& report);

}  // namespace cb

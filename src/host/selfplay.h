#pragma once

// Self-play games with the CPU search, for Texel tuning and NNUE training
// data (scripts/make_tuning_set.py writes the first three fields from match
// PGNs): every position after the opening, with the game's result and
// number, and the search's score of the position from white's side.
// Games start from a random opening of the suite plus a few random moves,
// then both sides search to a fixed depth. A game ends by the rules
// (mate, stalemate, threefold repetition, the 50-move rule, insufficient
// material), by adjudication (both sides agreeing on a score of at least
// `adjudicate_cp` for 4 plies), or as a draw after `max_plies`.

#include <functional>
#include <string>
#include <vector>

#include "host/search_cpu.h"

namespace cb {

struct SelfPlayOptions {
    int games = 1000;
    int depth = 5;
    int threads = 4;
    int random_plies = 4;    // random legal moves after the opening
    int max_plies = 400;
    int adjudicate_cp = 1000;
    unsigned long long seed = 1;
    std::vector<std::string> openings;  // FENs; the start position if empty
    std::string out_path;               // "<FEN> | <result> | <game> | <white's score>" lines (appended)
    CpuSearchOptions search{BoardKind::Quad, EvalKind::Pesto, OrderingKind::Staged};
    size_t hash_mb = 16;  // per thread, for feature sets with a table
    int first_game = 1;   // number of the first game (continuing an earlier file)
};

struct SelfPlayStats {
    int games = 0;
    int white_wins = 0;
    int black_wins = 0;
    int draws = 0;
    unsigned long long positions = 0;
};

// Plays the games; `progress` (may be empty) is called after each game,
// from the playing threads (serialized).
SelfPlayStats run_selfplay(const SelfPlayOptions& options, const std::function<void(const SelfPlayStats&)>& progress);

}  // namespace cb

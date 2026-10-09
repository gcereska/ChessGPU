#pragma once

// CPU perft: the reference the GPU results are checked against.

#include <algorithm>
#include <utility>
#include <vector>

#include "core/move32.h"
#include "host/uci_move.h"

namespace cb {

// Number of leaf positions `depth` plies below b, counting the last ply
// without playing it. b is restored before returning. Moves: Moves16 or
// Moves32 (core/move32.h).
template <class Moves = Moves16, class Board, class Attacks>
u64 perft_recursive(Board& b, const Attacks& a, int depth) {
    if (depth <= 0) return 1;
    if (depth == 1) return count_legal(b, a);
    typename Moves::template Buffer<Board> list;
    list.bind(b);
    generate_legal(b, a, list);
    u64 nodes = 0;
    for (int i = 0; i < list.size; ++i) {
        const Undo undo = make_move(b, list.moves[i]);
        nodes += perft_recursive<Moves>(b, a, depth - 1);
        unmake_move(b, list.moves[i], undo);
    }
    return nodes;
}

// Perft split by root move, sorted by UCI notation (the order Stockfish's
// "go perft" prints). depth must be >= 1.
template <class Moves = Moves16, class Board, class Attacks>
std::vector<std::pair<Move, u64>> perft_divide(Board& b, const Attacks& a, int depth) {
    typename Moves::template Buffer<Board> list;
    list.bind(b);
    generate_legal(b, a, list);
    std::vector<std::pair<Move, u64>> result;
    for (int i = 0; i < list.size; ++i) {
        const Undo undo = make_move(b, list.moves[i]);
        result.emplace_back(move16(list.moves[i]), perft_recursive<Moves>(b, a, depth - 1));
        unmake_move(b, list.moves[i], undo);
    }
    std::sort(result.begin(), result.end(),
              [](const auto& x, const auto& y) { return move_to_uci(x.first) < move_to_uci(y.first); });
    return result;
}

}  // namespace cb

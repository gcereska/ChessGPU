#pragma once

#include <ostream>
#include <string>
#include <string_view>

#include "core/movegen.h"

namespace cb {

std::string square_name(Square s);  // "e4"

// Parses a square name like "e4". Returns false if it isn't one.
bool parse_square(std::string_view text, Square& out);

// UCI notation: "e2e4", promotions "e7e8q". Castling is written as the king's
// move ("e1g1"). The empty move is "0000".
std::string move_to_uci(Move m);

// The legal move of the side to move whose UCI notation is `text`, or the
// empty Move if there is none.
template <class Board, class Attacks>
Move parse_uci_move(const Board& b, const Attacks& a, std::string_view text) {
    MoveList list;
    generate_legal(b, a, list);
    for (int i = 0; i < list.size; ++i)
        if (move_to_uci(list.moves[i]) == text) return list.moves[i];
    return Move();
}

std::ostream& operator<<(std::ostream& os, Move m);

}  // namespace cb

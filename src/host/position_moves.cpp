#include "host/position_moves.h"

#include "core/movegen.h"
#include "host/host_tables.h"
#include "host/uci_move.h"

namespace cb {

bool apply_uci_moves(Position& p, std::string_view moves, std::string& error) {
    const HqAttacks a(host_attack_tables());
    QuadBoard b = to_board<QuadBoard>(p);
    int fullmove = p.fullmove;
    size_t i = 0;
    while (i < moves.size()) {
        while (i < moves.size() && moves[i] == ' ') ++i;
        const size_t start = i;
        while (i < moves.size() && moves[i] != ' ') ++i;
        if (i == start) break;
        const std::string_view text = moves.substr(start, i - start);
        const Move m = parse_uci_move(b, a, text);
        if (m.is_none()) {
            error = "illegal or malformed move: " + std::string(text);
            return false;
        }
        if (state::side(b.state()) == BLACK) ++fullmove;
        make_move(b, m);
    }
    p = to_position(b, fullmove);
    return true;
}

}  // namespace cb

#include "host/game_history.h"

#include "host/host_tables.h"
#include "host/uci_move.h"

namespace cb {

namespace {

// Does position i occur earlier within its own reversible stretch (its
// halfmove clock), counting back in steps of two plies from 4?
bool repeats_earlier(const std::vector<u64>& keys, const std::vector<int>& halfmoves, size_t i) {
    for (size_t d = 4; d <= size_t(halfmoves[i]) && d <= i; d += 2)
        if (keys[i - d] == keys[i]) return true;
    return false;
}

}  // namespace

bool build_game(const Position& start, std::string_view moves, Position& root, GameHistory& history,
                std::string& error) {
    const HqAttacks a(host_attack_tables());
    const ZobristKeys& z = host_search_tables().zobrist;
    QuadBoard b = to_board<QuadBoard>(start);
    int fullmove = start.fullmove;
    std::vector<u64> keys = {compute_key(b, a, z)};
    std::vector<int> halfmoves = {state::halfmove(b.state())};

    size_t i = 0;
    while (i < moves.size()) {
        while (i < moves.size() && moves[i] == ' ') ++i;
        const size_t begin = i;
        while (i < moves.size() && moves[i] != ' ') ++i;
        if (i == begin) break;
        const std::string_view text = moves.substr(begin, i - begin);
        const Move m = parse_uci_move(b, a, text);
        if (m.is_none()) {
            error = "illegal or malformed move: " + std::string(text);
            return false;
        }
        if (state::side(b.state()) == BLACK) ++fullmove;
        make_move(b, m);
        keys.push_back(compute_key(b, a, z));
        halfmoves.push_back(state::halfmove(b.state()));
    }

    const size_t n = keys.size() - 1;  // the root's index
    GameHistory h;
    h.entries.reserve(n);
    for (size_t j = 0; j < n; ++j) {
        HistoryEntry e;
        e.key = keys[j];
        e.prev = int(j) - 1;
        e.ply = short(int(j) - int(n));
        e.repeated = repeats_earlier(keys, halfmoves, j) ? 1 : 0;
        h.entries.push_back(e);
    }
    h.root_repeated = repeats_earlier(keys, halfmoves, n);
    root = to_position(b, fullmove);
    history = std::move(h);
    return true;
}

}  // namespace cb

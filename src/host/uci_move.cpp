#include "host/uci_move.h"

namespace cb {

std::string square_name(Square s) {
    return std::string{char('a' + file_of(s)), char('1' + rank_of(s))};
}

bool parse_square(std::string_view text, Square& out) {
    if (text.size() != 2 || text[0] < 'a' || text[0] > 'h' || text[1] < '1' || text[1] > '8') return false;
    out = make_square(text[0] - 'a', text[1] - '1');
    return true;
}

std::string move_to_uci(Move m) {
    if (m.is_none()) return "0000";
    std::string s = square_name(m.from()) + square_name(m.to());
    if (m.is_promotion()) s += " nbrq"[m.promotion_type() - KNIGHT + 1];
    return s;
}

std::ostream& operator<<(std::ostream& os, Move m) { return os << move_to_uci(m); }

}  // namespace cb

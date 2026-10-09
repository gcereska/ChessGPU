#include "host/debug_print.h"

#include "host/fen.h"

namespace cb {

std::string bitboard_to_string(u64 b) {
    std::string s;
    for (int rank = 7; rank >= 0; --rank) {
        s += char('1' + rank);
        s += ' ';
        for (int file = 0; file < 8; ++file) {
            s += (b & bit(make_square(file, rank))) ? 'x' : '.';
            s += file < 7 ? ' ' : '\n';
        }
    }
    s += "  a b c d e f g h\n";
    return s;
}

std::string position_to_string(const Position& p) {
    static constexpr const char* kLetters = ".PNBRQK";
    std::string s;
    for (int rank = 7; rank >= 0; --rank) {
        s += char('1' + rank);
        s += ' ';
        for (int file = 0; file < 8; ++file) {
            const u8 code = p.squares[make_square(file, rank)];
            char c = kLetters[code_type(code)];
            if (code && code_color(code) == BLACK) c = char(c - 'A' + 'a');
            s += c;
            s += file < 7 ? ' ' : '\n';
        }
    }
    s += "  a b c d e f g h\n";
    s += format_fen(p) + '\n';
    return s;
}

}  // namespace cb

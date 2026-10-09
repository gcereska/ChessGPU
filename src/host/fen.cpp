#include "host/fen.h"

#include <charconv>
#include <vector>

#include "host/uci_move.h"

namespace cb {

namespace {

constexpr const char* kPieceLetters = " PNBRQK";

std::vector<std::string_view> split_whitespace(std::string_view text) {
    std::vector<std::string_view> fields;
    size_t i = 0;
    while (i < text.size()) {
        while (i < text.size() && (text[i] == ' ' || text[i] == '\t')) ++i;
        const size_t start = i;
        while (i < text.size() && text[i] != ' ' && text[i] != '\t') ++i;
        if (i > start) fields.push_back(text.substr(start, i - start));
    }
    return fields;
}

// 0 if c is not a piece letter.
u8 piece_code_from_letter(char c) {
    for (int pt = PAWN; pt <= KING; ++pt) {
        if (c == kPieceLetters[pt]) return piece_code(WHITE, PieceType(pt));
        if (c == kPieceLetters[pt] - 'A' + 'a') return piece_code(BLACK, PieceType(pt));
    }
    return 0;
}

char piece_letter(u8 code) {
    const char upper = kPieceLetters[code_type(code)];
    return code_color(code) == WHITE ? upper : char(upper - 'A' + 'a');
}

bool parse_non_negative(std::string_view text, int& out) {
    int value = 0;
    const auto [end, ec] = std::from_chars(text.data(), text.data() + text.size(), value);
    if (ec != std::errc() || end != text.data() + text.size() || value < 0) return false;
    out = value;
    return true;
}

bool fail(std::string& error, std::string message) {
    error = std::move(message);
    return false;
}

}  // namespace

bool parse_fen(std::string_view fen, Position& out, std::string& error) {
    const std::vector<std::string_view> fields = split_whitespace(fen);
    if (fields.size() < 4 || fields.size() > 6) return fail(error, "expected 4 to 6 fields");

    Position p;
    int rank = 7;
    int file = 0;
    for (const char c : fields[0]) {
        if (c == '/') {
            if (file != 8) return fail(error, "rank " + std::to_string(rank + 1) + " does not have 8 squares");
            if (--rank < 0) return fail(error, "more than 8 ranks");
            file = 0;
        } else if (c >= '1' && c <= '8') {
            file += c - '0';
            if (file > 8) return fail(error, "rank " + std::to_string(rank + 1) + " has more than 8 squares");
        } else {
            const u8 code = piece_code_from_letter(c);
            if (!code) return fail(error, std::string("unknown piece letter '") + c + "'");
            if (file >= 8) return fail(error, "rank " + std::to_string(rank + 1) + " has more than 8 squares");
            p.squares[make_square(file, rank)] = code;
            ++file;
        }
    }
    if (rank != 0 || file != 8) return fail(error, "piece placement must describe 8 ranks of 8 squares");

    Color side;
    if (fields[1] == "w") {
        side = WHITE;
    } else if (fields[1] == "b") {
        side = BLACK;
    } else {
        return fail(error, "side to move must be 'w' or 'b'");
    }

    u32 rights = 0;
    if (fields[2] != "-") {
        for (const char c : fields[2]) {
            const u32 right = c == 'K' ? WHITE_OO : c == 'Q' ? WHITE_OOO : c == 'k' ? BLACK_OO : c == 'q' ? BLACK_OOO : 0;
            if (!right) return fail(error, std::string("unknown castling letter '") + c + "'");
            if (rights & right) return fail(error, std::string("castling letter '") + c + "' repeated");
            rights |= right;
        }
    }

    bool has_ep = false;
    int ep_file = 0;
    if (fields[3] != "-") {
        Square s;
        if (!parse_square(fields[3], s)) return fail(error, "invalid en passant square");
        if (rank_of(s) != (side == WHITE ? 5 : 2))
            return fail(error, "en passant square must be on rank 6 (white to move) or rank 3 (black to move)");
        has_ep = true;
        ep_file = file_of(s);
    }

    int halfmove = 0;
    int fullmove = 1;
    if (fields.size() >= 5 && !parse_non_negative(fields[4], halfmove)) return fail(error, "invalid halfmove clock");
    if (fields.size() == 6 && !parse_non_negative(fields[5], fullmove)) return fail(error, "invalid fullmove number");

    p.state = state::pack(rights, has_ep, ep_file, side, halfmove);
    p.fullmove = fullmove;
    if (!validate_position(p, error)) return false;
    out = p;
    return true;
}

std::string format_fen(const Position& p) {
    std::string fen;
    for (int rank = 7; rank >= 0; --rank) {
        int empty = 0;
        for (int file = 0; file < 8; ++file) {
            const u8 code = p.squares[make_square(file, rank)];
            if (!code) {
                ++empty;
                continue;
            }
            if (empty) fen += char('0' + empty);
            empty = 0;
            fen += piece_letter(code);
        }
        if (empty) fen += char('0' + empty);
        if (rank > 0) fen += '/';
    }

    fen += state::side(p.state) == WHITE ? " w " : " b ";
    const u32 rights = state::castling(p.state);
    if (rights & WHITE_OO) fen += 'K';
    if (rights & WHITE_OOO) fen += 'Q';
    if (rights & BLACK_OO) fen += 'k';
    if (rights & BLACK_OOO) fen += 'q';
    if (!rights) fen += '-';
    fen += ' ';
    fen += state::has_ep(p.state) ? square_name(state::ep_square(p.state)) : "-";
    fen += ' ' + std::to_string(state::halfmove(p.state)) + ' ' + std::to_string(p.fullmove);
    return fen;
}

}  // namespace cb

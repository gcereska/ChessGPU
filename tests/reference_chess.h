#pragma once

// Deliberately simple chess rules on an 8x8 array, sharing nothing with the
// bitboard code except the Move/Position types. Slow but easy to check by
// eye; used as an oracle for move generation and make_move.

#include <vector>

#include "host/position.h"

namespace ref {

using cb::Color;
using cb::Move;
using cb::PieceType;

struct Board {
    int sq[64];  // piece codes as in cb::Position (0 = empty)
    Color side;
    unsigned castling;  // cb::CastlingRight flags
    int ep;             // en passant target square, or -1
    int halfmove;       // saturates at 127 like the state word
};

inline bool on_board(int file, int rank) { return file >= 0 && file < 8 && rank >= 0 && rank < 8; }
inline int code(Color c, PieceType pt) { return cb::piece_code(c, pt); }
inline bool is_color(int code_value, Color c) { return code_value != 0 && cb::code_color(cb::u8(code_value)) == c; }
inline PieceType type_of(int code_value) { return cb::code_type(cb::u8(code_value)); }

inline Board from_position(const cb::Position& p) {
    Board b{};
    for (int i = 0; i < 64; ++i) b.sq[i] = p.squares[i];
    b.side = cb::state::side(p.state);
    b.castling = cb::state::castling(p.state);
    b.ep = cb::state::has_ep(p.state) ? int(cb::state::ep_square(p.state)) : -1;
    b.halfmove = cb::state::halfmove(p.state);
    return b;
}

inline cb::Position to_position(const Board& b) {
    cb::Position p;
    for (int i = 0; i < 64; ++i) p.squares[i] = cb::u8(b.sq[i]);
    p.state = cb::state::pack(b.castling, b.ep >= 0, b.ep >= 0 ? b.ep % 8 : 0, b.side, b.halfmove);
    return p;
}

constexpr int kKnightSteps[8][2] = {{1, 2}, {2, 1}, {2, -1}, {1, -2}, {-1, -2}, {-2, -1}, {-2, 1}, {-1, 2}};
constexpr int kKingSteps[8][2] = {{1, 0}, {1, 1}, {0, 1}, {-1, 1}, {-1, 0}, {-1, -1}, {0, -1}, {1, -1}};
constexpr int kRookDirs[4][2] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};
constexpr int kBishopDirs[4][2] = {{1, 1}, {1, -1}, {-1, 1}, {-1, -1}};

// Is `target` attacked by any piece of color `by`?
inline bool attacked(const Board& b, int target, Color by) {
    const int tf = target % 8;
    const int tr = target / 8;
    // A pawn of color `by` attacks diagonally forward, so it stands one rank
    // behind the target from its own point of view.
    const int pawn_rank = tr - (by == cb::WHITE ? 1 : -1);
    for (int df : {-1, 1})
        if (on_board(tf + df, pawn_rank) && b.sq[pawn_rank * 8 + tf + df] == code(by, cb::PAWN)) return true;
    for (const auto& s : kKnightSteps)
        if (on_board(tf + s[0], tr + s[1]) && b.sq[(tr + s[1]) * 8 + tf + s[0]] == code(by, cb::KNIGHT)) return true;
    for (const auto& s : kKingSteps)
        if (on_board(tf + s[0], tr + s[1]) && b.sq[(tr + s[1]) * 8 + tf + s[0]] == code(by, cb::KING)) return true;
    for (int straight = 0; straight < 2; ++straight) {
        const auto& dirs = straight ? kRookDirs : kBishopDirs;
        const PieceType slider = straight ? cb::ROOK : cb::BISHOP;
        for (const auto& d : dirs) {
            int f = tf + d[0];
            int r = tr + d[1];
            while (on_board(f, r)) {
                const int c = b.sq[r * 8 + f];
                if (c) {
                    if (c == code(by, slider) || c == code(by, cb::QUEEN)) return true;
                    break;
                }
                f += d[0];
                r += d[1];
            }
        }
    }
    return false;
}

inline int find_king(const Board& b, Color c) {
    for (int i = 0; i < 64; ++i)
        if (b.sq[i] == code(c, cb::KING)) return i;
    return -1;
}

// Plays m (assumed pseudo-legal) and returns the new position.
inline Board play(const Board& b, Move m) {
    Board n = b;
    const int from = m.from();
    const int to = m.to();
    const unsigned flags = m.flags();
    const int piece = n.sq[from];
    const bool is_capture = n.sq[to] != 0 || flags == cb::EP_CAPTURE;

    if (flags == cb::EP_CAPTURE) n.sq[to + (b.side == cb::WHITE ? -8 : 8)] = 0;
    n.sq[to] = piece;
    n.sq[from] = 0;
    if (m.is_promotion()) n.sq[to] = code(b.side, m.promotion_type());
    if (flags == cb::KING_CASTLE) {
        n.sq[to - 1] = n.sq[to + 1];  // rook from the h-file to the f-file
        n.sq[to + 1] = 0;
    } else if (flags == cb::QUEEN_CASTLE) {
        n.sq[to + 1] = n.sq[to - 2];  // rook from the a-file to the d-file
        n.sq[to - 2] = 0;
    }

    if (piece == code(cb::WHITE, cb::KING)) n.castling &= ~unsigned(cb::WHITE_OO | cb::WHITE_OOO);
    if (piece == code(cb::BLACK, cb::KING)) n.castling &= ~unsigned(cb::BLACK_OO | cb::BLACK_OOO);
    for (int s : {from, to}) {
        if (s == cb::H1) n.castling &= ~unsigned(cb::WHITE_OO);
        if (s == cb::A1) n.castling &= ~unsigned(cb::WHITE_OOO);
        if (s == cb::H8) n.castling &= ~unsigned(cb::BLACK_OO);
        if (s == cb::A8) n.castling &= ~unsigned(cb::BLACK_OOO);
    }

    n.ep = flags == cb::DOUBLE_PUSH ? (from + to) / 2 : -1;
    n.halfmove = (type_of(piece) == cb::PAWN || is_capture) ? 0 : (b.halfmove < 127 ? b.halfmove + 1 : 127);
    n.side = ~b.side;
    return n;
}

inline std::vector<Move> pseudo_legal_moves(const Board& b) {
    std::vector<Move> moves;
    const Color us = b.side;
    const Color them = ~us;
    const auto add = [&](int from, int to, unsigned flags) { moves.push_back(Move(cb::Square(from), cb::Square(to), flags)); };

    for (int from = 0; from < 64; ++from) {
        if (!is_color(b.sq[from], us)) continue;
        const int ff = from % 8;
        const int fr = from / 8;
        switch (type_of(b.sq[from])) {
            case cb::PAWN: {
                const int dir = us == cb::WHITE ? 1 : -1;
                const int start_rank = us == cb::WHITE ? 1 : 6;
                const int last_rank = us == cb::WHITE ? 7 : 0;
                const int r1 = fr + dir;
                if (on_board(ff, r1) && b.sq[r1 * 8 + ff] == 0) {
                    if (r1 == last_rank) {
                        for (unsigned p = cb::PROMO_KNIGHT; p <= cb::PROMO_QUEEN; ++p) add(from, r1 * 8 + ff, p);
                    } else {
                        add(from, r1 * 8 + ff, cb::QUIET);
                    }
                    const int r2 = fr + 2 * dir;
                    if (fr == start_rank && b.sq[r2 * 8 + ff] == 0) add(from, r2 * 8 + ff, cb::DOUBLE_PUSH);
                }
                for (int df : {-1, 1}) {
                    const int f = ff + df;
                    if (!on_board(f, r1)) continue;
                    const int to = r1 * 8 + f;
                    if (is_color(b.sq[to], them)) {
                        if (r1 == last_rank) {
                            for (unsigned p = cb::PROMO_CAPTURE_KNIGHT; p <= cb::PROMO_CAPTURE_QUEEN; ++p) add(from, to, p);
                        } else {
                            add(from, to, cb::CAPTURE);
                        }
                    } else if (to == b.ep) {
                        add(from, to, cb::EP_CAPTURE);
                    }
                }
                break;
            }
            case cb::KNIGHT:
            case cb::KING: {
                const bool knight = type_of(b.sq[from]) == cb::KNIGHT;
                for (int i = 0; i < 8; ++i) {
                    const int f = ff + (knight ? kKnightSteps[i][0] : kKingSteps[i][0]);
                    const int r = fr + (knight ? kKnightSteps[i][1] : kKingSteps[i][1]);
                    if (!on_board(f, r)) continue;
                    const int to = r * 8 + f;
                    if (b.sq[to] == 0) add(from, to, cb::QUIET);
                    else if (is_color(b.sq[to], them)) add(from, to, cb::CAPTURE);
                }
                break;
            }
            default: {
                const PieceType pt = type_of(b.sq[from]);
                for (int straight = 0; straight < 2; ++straight) {
                    if (straight && pt == cb::BISHOP) continue;
                    if (!straight && pt == cb::ROOK) continue;
                    const auto& dirs = straight ? kRookDirs : kBishopDirs;
                    for (const auto& d : dirs) {
                        int f = ff + d[0];
                        int r = fr + d[1];
                        while (on_board(f, r)) {
                            const int to = r * 8 + f;
                            if (b.sq[to] == 0) {
                                add(from, to, cb::QUIET);
                            } else {
                                if (is_color(b.sq[to], them)) add(from, to, cb::CAPTURE);
                                break;
                            }
                            f += d[0];
                            r += d[1];
                        }
                    }
                }
                break;
            }
        }
    }

    // Castling: rights, king and rook in place, empty squares between them,
    // and the king never on, through, or into an attacked square.
    const int base = us == cb::WHITE ? 0 : 56;
    const int king = code(us, cb::KING);
    const int rook = code(us, cb::ROOK);
    const unsigned oo = us == cb::WHITE ? cb::WHITE_OO : cb::BLACK_OO;
    const unsigned ooo = us == cb::WHITE ? cb::WHITE_OOO : cb::BLACK_OOO;
    if ((b.castling & oo) && b.sq[base + 4] == king && b.sq[base + 7] == rook && b.sq[base + 5] == 0 &&
        b.sq[base + 6] == 0 && !attacked(b, base + 4, them) && !attacked(b, base + 5, them) &&
        !attacked(b, base + 6, them))
        add(base + 4, base + 6, cb::KING_CASTLE);
    if ((b.castling & ooo) && b.sq[base + 4] == king && b.sq[base + 0] == rook && b.sq[base + 1] == 0 &&
        b.sq[base + 2] == 0 && b.sq[base + 3] == 0 && !attacked(b, base + 4, them) &&
        !attacked(b, base + 3, them) && !attacked(b, base + 2, them))
        add(base + 4, base + 2, cb::QUEEN_CASTLE);
    return moves;
}

// Pseudo-legal moves that don't leave our own king attacked.
inline std::vector<Move> legal_moves(const Board& b) {
    std::vector<Move> legal;
    for (const Move m : pseudo_legal_moves(b)) {
        const Board n = play(b, m);
        if (!attacked(n, find_king(n, b.side), ~b.side)) legal.push_back(m);
    }
    return legal;
}

}  // namespace ref

#pragma once

// Zobrist hash keys: a position's key is the XOR of one random 64-bit number
// per (color, piece type, square) occupied, plus numbers for the castling
// rights, the en passant file and the side to move. Making a move XORs the
// changed parts in and out. Used for repetition detection now and the
// transposition table later.

#include "core/movegen.h"

namespace cb {

struct ZobristKeys {
    u64 piece[2][8][64];  // [color][piece type][square]; type 0 (no piece) stays zero
    u64 castling[16];     // [castling rights]: XOR of one key per right held
    u64 ep_file[8];       // only included when an en passant capture is legal (see below)
    u64 side;             // included when black is to move
};

namespace detail {

// SplitMix64 (Steele, Lea, Flood): a fixed seed gives the same keys in every
// build, so keys can be compared across runs and processes.
inline u64 splitmix64(u64& state) {
    u64 z = (state += 0x9E3779B97F4A7C15ull);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}

}  // namespace detail

// Fills z (host only).
inline void init_zobrist_keys(ZobristKeys& z) {
    u64 seed = 0x63686573736F7421ull;
    for (int c = 0; c < 2; ++c)
        for (int pt = 0; pt < 8; ++pt)
            for (int s = 0; s < 64; ++s)
                z.piece[c][pt][s] = (pt >= PAWN && pt <= KING) ? detail::splitmix64(seed) : 0;
    u64 right_keys[4];
    for (u64& k : right_keys) k = detail::splitmix64(seed);
    for (u32 rights = 0; rights < 16; ++rights) {
        z.castling[rights] = 0;
        for (int i = 0; i < 4; ++i)
            if (rights & (1u << i)) z.castling[rights] ^= right_keys[i];
    }
    for (u64& k : z.ep_file) k = detail::splitmix64(seed);
    z.side = detail::splitmix64(seed);
}

// True if the side to move has a legal en passant capture. The state word
// keeps the en passant square after every double push (like FEN), but the
// key includes it only when the capture is actually possible, so positions
// that differ only by an unusable en passant square get the same key and
// count as repetitions, as the rules of chess say they should.
template <class Board, class Attacks>
CB_HD bool en_passant_capture_possible(const Board& b, const Attacks& a) {
    const u32 st = b.state();
    if (!state::has_ep(st)) return false;
    const Color us = state::side(st);
    const Square to = state::ep_square(st);
    u64 from_set = a.pawn(~us, to) & b.template pieces<PAWN>() & b.color(us);
    if (!from_set) return false;
    // en_passant_is_legal only reads these fields of the context.
    LegalContext c;
    c.us = us;
    c.occupied = b.occupied();
    c.their = b.color(~us);
    c.king = king_square(b, us);
    while (from_set) {
        if (en_passant_is_legal(b, a, c, Square(pop_lsb(from_set)), to)) return true;
    }
    return false;
}

// Key of the state word's castling rights, side to move and en passant
// file. `ep_possible` must be en_passant_capture_possible() of the position.
CB_HD u64 state_key(const ZobristKeys& z, u32 st, bool ep_possible) {
    u64 k = load_readonly(&z.castling[state::castling(st)]);
    if (state::side(st) == BLACK) k ^= load_readonly(&z.side);
    if (ep_possible) k ^= load_readonly(&z.ep_file[state::ep_file(st)]);
    return k;
}

CB_HD u64 piece_key(const ZobristKeys& z, Color c, PieceType pt, Square s) {
    return load_readonly(&z.piece[c][pt][s]);
}

// The key of b computed from scratch.
template <class Board, class Attacks>
CB_HD u64 compute_key(const Board& b, const Attacks& a, const ZobristKeys& z) {
    u64 k = state_key(z, b.state(), en_passant_capture_possible(b, a));
    const u64 black = b.color(BLACK);
    for (u64 occ = b.occupied(); occ;) {
        const Square s = Square(pop_lsb(occ));
        k ^= piece_key(z, (black & bit(s)) ? BLACK : WHITE, b.piece_type_on(s), s);
    }
    return k;
}

}  // namespace cb

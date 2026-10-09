#pragma once

#include <cstring>

#include "core/bitboard.h"

namespace cb {

// The four lines through a square. Diagonals run a1-h8, anti-diagonals a8-h1.
enum LineDirection : int {
    LINE_FILE = 0,
    LINE_RANK = 1,
    LINE_DIAGONAL = 2,
    LINE_ANTI_DIAGONAL = 3,
};

// Lookup tables for move generation. Built once on the host by
// init_attack_tables(); kernels use a copy in device memory.
struct AttackTables {
    u64 knight[64];
    u64 king[64];
    u64 pawn[2][64];       // [color][square]: squares a pawn of that color attacks from there
    u64 line_mask[4][64];  // [LineDirection][square]: the line through the square, excluding it
    u64 between[64][64];   // squares strictly between two squares on a common line, else 0
    u64 line[64][64];      // the whole line through two squares on a common line, both included, else 0
};

namespace detail {

inline bool on_board(int file, int rank) { return file >= 0 && file < 8 && rank >= 0 && rank < 8; }

// Squares reached from s by repeatedly stepping (df, dr), up to and including
// the first square in occupied.
inline u64 walk_ray(Square s, int df, int dr, u64 occupied) {
    u64 result = 0;
    int f = file_of(s) + df;
    int r = rank_of(s) + dr;
    while (on_board(f, r)) {
        const Square t = make_square(f, r);
        result |= bit(t);
        if (occupied & bit(t)) break;
        f += df;
        r += dr;
    }
    return result;
}

inline u64 leaper_targets(Square s, const int (*offsets)[2], int count) {
    u64 result = 0;
    for (int i = 0; i < count; ++i) {
        const int f = file_of(s) + offsets[i][0];
        const int r = rank_of(s) + offsets[i][1];
        if (on_board(f, r)) result |= bit(make_square(f, r));
    }
    return result;
}

}  // namespace detail

// Fills t (host only). Every entry is derived directly from board geometry.
inline void init_attack_tables(AttackTables& t) {
    static constexpr int kKnightOffsets[8][2] = {{1, 2}, {2, 1}, {2, -1}, {1, -2},
                                                 {-1, -2}, {-2, -1}, {-2, 1}, {-1, 2}};
    static constexpr int kKingOffsets[8][2] = {{1, 0}, {1, 1}, {0, 1}, {-1, 1},
                                               {-1, 0}, {-1, -1}, {0, -1}, {1, -1}};
    static constexpr int kWhitePawnOffsets[2][2] = {{-1, 1}, {1, 1}};
    static constexpr int kBlackPawnOffsets[2][2] = {{-1, -1}, {1, -1}};
    // Each direction and the line it lies on, so between/line can share one walk.
    static constexpr int kDirections[8][3] = {
        {0, 1, LINE_FILE},     {0, -1, LINE_FILE},     {1, 0, LINE_RANK},
        {-1, 0, LINE_RANK},    {1, 1, LINE_DIAGONAL},  {-1, -1, LINE_DIAGONAL},
        {1, -1, LINE_ANTI_DIAGONAL}, {-1, 1, LINE_ANTI_DIAGONAL},
    };

    std::memset(&t, 0, sizeof t);
    for (int i = 0; i < 64; ++i) {
        const Square s = Square(i);
        t.knight[s] = detail::leaper_targets(s, kKnightOffsets, 8);
        t.king[s] = detail::leaper_targets(s, kKingOffsets, 8);
        t.pawn[WHITE][s] = detail::leaper_targets(s, kWhitePawnOffsets, 2);
        t.pawn[BLACK][s] = detail::leaper_targets(s, kBlackPawnOffsets, 2);
        for (const auto& d : kDirections) t.line_mask[d[2]][s] |= detail::walk_ray(s, d[0], d[1], 0);
    }
    for (int i = 0; i < 64; ++i) {
        const Square a = Square(i);
        for (const auto& d : kDirections) {
            const u64 full_line = t.line_mask[d[2]][a] | bit(a);
            u64 passed = 0;  // squares strictly between a and the current square
            int f = file_of(a) + d[0];
            int r = rank_of(a) + d[1];
            while (detail::on_board(f, r)) {
                const Square b = make_square(f, r);
                t.between[a][b] = passed;
                t.line[a][b] = full_line;
                passed |= bit(b);
                f += d[0];
                r += d[1];
            }
        }
    }
}

// The lookups every attack variant shares: leapers, lines between squares,
// and sliders on an empty board, all from AttackTables. Cheap to copy (one
// pointer); the tables can live anywhere the code can read. The variants
// (docs/ARCHITECTURE.md, Move generation) add bishop, rook and queen attacks
// given the occupied squares; code takes the variant as a template
// parameter (`Attacks`).
class TableAttacks {
  public:
    CB_HD explicit TableAttacks(const AttackTables& tables) : t_(&tables) {}

    CB_HD u64 knight(Square s) const { return load_readonly(&t_->knight[s]); }
    CB_HD u64 king(Square s) const { return load_readonly(&t_->king[s]); }
    CB_HD u64 pawn(Color c, Square s) const { return load_readonly(&t_->pawn[c][s]); }

    // Attacks on an empty board.
    CB_HD u64 bishop_empty_board(Square s) const {
        return load_readonly(&t_->line_mask[LINE_DIAGONAL][s]) |
               load_readonly(&t_->line_mask[LINE_ANTI_DIAGONAL][s]);
    }
    CB_HD u64 rook_empty_board(Square s) const {
        return load_readonly(&t_->line_mask[LINE_FILE][s]) | load_readonly(&t_->line_mask[LINE_RANK][s]);
    }

    CB_HD u64 between(Square a, Square b) const { return load_readonly(&t_->between[a][b]); }
    CB_HD u64 line(Square a, Square b) const { return load_readonly(&t_->line[a][b]); }

  protected:
    const AttackTables* t_;
};

// Hyperbola quintessence for sliding pieces: on each line through the
// square, subtracting twice the piece's bit from the blockers ripples up to
// the first blocker above it, and the same trick on the bit-reversed board
// finds the first blocker below it.
class HqAttacks : public TableAttacks {
  public:
    static constexpr const char* kName = "hq";

    CB_HD explicit HqAttacks(const AttackTables& tables) : TableAttacks(tables) {}

    CB_HD u64 bishop(Square s, u64 occupied) const {
        return line_attacks(s, occupied, load_readonly(&t_->line_mask[LINE_DIAGONAL][s])) |
               line_attacks(s, occupied, load_readonly(&t_->line_mask[LINE_ANTI_DIAGONAL][s]));
    }
    CB_HD u64 rook(Square s, u64 occupied) const {
        return line_attacks(s, occupied, load_readonly(&t_->line_mask[LINE_FILE][s])) |
               line_attacks(s, occupied, load_readonly(&t_->line_mask[LINE_RANK][s]));
    }
    CB_HD u64 queen(Square s, u64 occupied) const { return bishop(s, occupied) | rook(s, occupied); }

  private:
    // Attacks along one line (mask excludes s) given the occupied squares.
    CB_HD static u64 line_attacks(Square s, u64 occupied, u64 mask) {
        const u64 o = occupied & mask;
        const u64 forward = o - (bit(s) << 1);
        const u64 reverse = reverse_bits(reverse_bits(o) - (bit(Square(63 - s)) << 1));
        return (forward ^ reverse) & mask;
    }
};

}  // namespace cb

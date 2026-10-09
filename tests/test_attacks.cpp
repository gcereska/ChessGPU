// Attack tables and sliding attacks, checked against independent geometry.

#include <random>

#include "core/attacks.h"
#include "host/host_tables.h"
#include "reference_chess.h"
#include "test_harness.h"

using namespace cb;

namespace {

const HqAttacks& attacks() {
    static const HqAttacks a(host_attack_tables());
    return a;
}

// Squares attacked from `from` by a single piece on an otherwise empty board,
// according to the reference implementation.
u64 reference_single_piece_attacks(Color c, PieceType pt, Square from) {
    ref::Board b{};
    b.sq[from] = ref::code(c, pt);
    b.ep = -1;
    u64 result = 0;
    for (int t = 0; t < 64; ++t)
        if (t != from && ref::attacked(b, t, c)) result |= bit(Square(t));
    return result;
}

// Slider attacks by walking rays, independent of the attacks.h helpers.
u64 walk(Square s, u64 occupied, const int (*dirs)[2]) {
    u64 result = 0;
    for (int d = 0; d < 4; ++d) {
        int f = file_of(s) + dirs[d][0];
        int r = rank_of(s) + dirs[d][1];
        while (f >= 0 && f < 8 && r >= 0 && r < 8) {
            result |= bit(make_square(f, r));
            if (occupied & bit(make_square(f, r))) break;
            f += dirs[d][0];
            r += dirs[d][1];
        }
    }
    return result;
}

u64 naive_rook(Square s, u64 occupied) { return walk(s, occupied, ref::kRookDirs); }
u64 naive_bishop(Square s, u64 occupied) { return walk(s, occupied, ref::kBishopDirs); }

// Calls f(subset) for every subset of mask (including 0 and mask itself).
template <class F>
void for_each_subset(u64 mask, F&& f) {
    u64 subset = 0;
    do {
        f(subset);
        subset = (subset - mask) & mask;
    } while (subset != 0);
}

}  // namespace

TEST_CASE(attacks_leapers_match_reference) {
    const AttackTables& t = host_attack_tables();
    for (int i = 0; i < 64; ++i) {
        const Square s = Square(i);
        CHECK_EQ(t.knight[s], reference_single_piece_attacks(WHITE, KNIGHT, s));
        CHECK_EQ(t.king[s], reference_single_piece_attacks(WHITE, KING, s));
        CHECK_EQ(t.pawn[WHITE][s], reference_single_piece_attacks(WHITE, PAWN, s));
        CHECK_EQ(t.pawn[BLACK][s], reference_single_piece_attacks(BLACK, PAWN, s));
        CHECK_EQ(attacks().knight(s), t.knight[s]);
        CHECK_EQ(attacks().king(s), t.king[s]);
        CHECK_EQ(attacks().pawn(BLACK, s), t.pawn[BLACK][s]);
    }
    // Spot checks against hand-computed values.
    CHECK_EQ(t.knight[A1], bit(B3) | bit(C2));
    CHECK_EQ(t.king[H8], bit(G8) | bit(G7) | bit(H7));
    CHECK_EQ(t.pawn[WHITE][E4], bit(D5) | bit(F5));
    CHECK_EQ(t.pawn[BLACK][A5], bit(B4));
}

TEST_CASE(attacks_set_wise_match_tables) {
    const AttackTables& t = host_attack_tables();
    for (int i = 0; i < 64; ++i) {
        const Square s = Square(i);
        CHECK_EQ(knight_attacks_set(bit(s)), t.knight[s]);
        CHECK_EQ(king_attacks_set(bit(s)), t.king[s]);
        CHECK_EQ(pawn_attacks_set(WHITE, bit(s)), t.pawn[WHITE][s]);
        CHECK_EQ(pawn_attacks_set(BLACK, bit(s)), t.pawn[BLACK][s]);
    }
    // Sets are the union of their members.
    std::mt19937_64 rng(7);
    for (int trial = 0; trial < 2000; ++trial) {
        const u64 set = rng() & rng();
        u64 knights = 0, kings = 0, white = 0, black = 0;
        for (u64 b = set; b;) {
            const Square s = Square(pop_lsb(b));
            knights |= t.knight[s];
            kings |= t.king[s];
            white |= t.pawn[WHITE][s];
            black |= t.pawn[BLACK][s];
        }
        CHECK_EQ(knight_attacks_set(set), knights);
        CHECK_EQ(king_attacks_set(set), kings);
        CHECK_EQ(pawn_attacks_set(WHITE, set), white);
        CHECK_EQ(pawn_attacks_set(BLACK, set), black);
    }
}

TEST_CASE(attacks_sliders_exhaustive) {
    // Every subset of blockers on the piece's own lines, edges included.
    long long checked = 0;
    for (int i = 0; i < 64; ++i) {
        const Square s = Square(i);
        const u64 rook_lines = naive_rook(s, 0);
        const u64 bishop_lines = naive_bishop(s, 0);
        CHECK_EQ(attacks().rook_empty_board(s), rook_lines);
        CHECK_EQ(attacks().bishop_empty_board(s), bishop_lines);
        for_each_subset(rook_lines, [&](u64 occ) {
            CHECK_EQ(attacks().rook(s, occ), naive_rook(s, occ));
            ++checked;
        });
        for_each_subset(bishop_lines, [&](u64 occ) {
            CHECK_EQ(attacks().bishop(s, occ), naive_bishop(s, occ));
            ++checked;
        });
    }
    CHECK(checked > 500000);
}

TEST_CASE(attacks_sliders_random_occupancy) {
    // Random full boards: pieces off the lines and on the slider's own square
    // must not matter.
    std::mt19937_64 rng(99);
    for (int trial = 0; trial < 20000; ++trial) {
        const Square s = Square(rng() % 64);
        u64 occ = rng() & rng();
        if (trial & 1) occ |= bit(s);
        CHECK_EQ(attacks().rook(s, occ), naive_rook(s, occ));
        CHECK_EQ(attacks().bishop(s, occ), naive_bishop(s, occ));
        CHECK_EQ(attacks().queen(s, occ), naive_rook(s, occ) | naive_bishop(s, occ));
    }
}

TEST_CASE(attacks_line_masks) {
    const AttackTables& t = host_attack_tables();
    for (int i = 0; i < 64; ++i) {
        const Square s = Square(i);
        CHECK_EQ(t.line_mask[LINE_FILE][s], file_mask(file_of(s)) & ~bit(s));
        CHECK_EQ(t.line_mask[LINE_RANK][s], rank_mask(rank_of(s)) & ~bit(s));
        u64 diag = 0, anti = 0;
        for (int j = 0; j < 64; ++j) {
            const Square o = Square(j);
            if (o == s) continue;
            if (file_of(o) - file_of(s) == rank_of(o) - rank_of(s)) diag |= bit(o);
            if (file_of(o) - file_of(s) == rank_of(s) - rank_of(o)) anti |= bit(o);
        }
        CHECK_EQ(t.line_mask[LINE_DIAGONAL][s], diag);
        CHECK_EQ(t.line_mask[LINE_ANTI_DIAGONAL][s], anti);
    }
}

TEST_CASE(attacks_between_and_line) {
    for (int i = 0; i < 64; ++i) {
        for (int j = 0; j < 64; ++j) {
            const Square a = Square(i);
            const Square b = Square(j);
            const int df = file_of(b) - file_of(a);
            const int dr = rank_of(b) - rank_of(a);
            const bool aligned = a != b && (df == 0 || dr == 0 || df == dr || df == -dr);
            u64 between = 0;
            u64 line = 0;
            if (aligned) {
                const int sf = (df > 0) - (df < 0);
                const int sr = (dr > 0) - (dr < 0);
                for (int k = 1; file_of(a) + k * sf != file_of(b) || rank_of(a) + k * sr != rank_of(b); ++k)
                    between |= bit(make_square(file_of(a) + k * sf, rank_of(a) + k * sr));
                // The whole line: walk both ways from a.
                line = bit(a);
                for (int sign : {1, -1}) {
                    int f = file_of(a) + sign * sf;
                    int r = rank_of(a) + sign * sr;
                    while (f >= 0 && f < 8 && r >= 0 && r < 8) {
                        line |= bit(make_square(f, r));
                        f += sign * sf;
                        r += sign * sr;
                    }
                }
            }
            CHECK_EQ(attacks().between(a, b), between);
            CHECK_EQ(attacks().line(a, b), line);
        }
    }
    CHECK_EQ(attacks().between(A1, H8), bit(B2) | bit(C3) | bit(D4) | bit(E5) | bit(F6) | bit(G7));
    CHECK_EQ(attacks().between(E1, E2), 0ull);
    CHECK_EQ(attacks().between(B1, C3), 0ull);  // a knight's jump is not a line
    CHECK_EQ(attacks().line(B1, C3), 0ull);
}

// Bit utilities, square/move/state encodings, and bitboard shifts.

#include <random>

#include "core/bitboard.h"
#include "core/state.h"
#include "test_harness.h"

using namespace cb;

namespace {

int naive_popcount(u64 b) {
    int n = 0;
    for (int i = 0; i < 64; ++i) n += int((b >> i) & 1);
    return n;
}

u64 naive_reverse(u64 b) {
    u64 r = 0;
    for (int i = 0; i < 64; ++i)
        if ((b >> i) & 1) r |= 1ull << (63 - i);
    return r;
}

std::vector<u64> sample_values() {
    std::vector<u64> values = {1, 2, 3, 0x8000000000000000ull, 0xFFFFFFFFFFFFFFFFull, 0x0123456789ABCDEFull,
                               0xF0F0F0F0F0F0F0F0ull, 0x00000000FFFFFFFFull, 0xFFFFFFFF00000000ull};
    for (int i = 0; i < 64; ++i) values.push_back(1ull << i);
    std::mt19937_64 rng(12345);
    for (int i = 0; i < 2000; ++i) values.push_back(rng());
    // Sparse values, like real bitboards.
    for (int i = 0; i < 2000; ++i) values.push_back(rng() & rng() & rng());
    return values;
}

}  // namespace

TEST_CASE(bits_popcount_lsb_reverse) {
    for (const u64 v : sample_values()) {
        CHECK_EQ(popcount(v), naive_popcount(v));
        CHECK_EQ(reverse_bits(v), naive_reverse(v));
        if (v == 0) continue;  // lsb is undefined for 0
        int expected_lsb = 0;
        while (!((v >> expected_lsb) & 1)) ++expected_lsb;
        CHECK_EQ(lsb(v), expected_lsb);
        u64 copy = v;
        CHECK_EQ(pop_lsb(copy), expected_lsb);
        CHECK_EQ(copy, v & (v - 1));
    }
    CHECK_EQ(popcount(0), 0);
    CHECK_EQ(reverse_bits(0), u64(0));
}

TEST_CASE(bits_mask_if) {
    CHECK_EQ(mask_if(true), ~0ull);
    CHECK_EQ(mask_if(false), 0ull);
}

TEST_CASE(squares) {
    for (int r = 0; r < 8; ++r) {
        for (int f = 0; f < 8; ++f) {
            const Square s = make_square(f, r);
            CHECK_EQ(int(s), r * 8 + f);
            CHECK_EQ(file_of(s), f);
            CHECK_EQ(rank_of(s), r);
            CHECK_EQ(bit(s), 1ull << (r * 8 + f));
        }
    }
    CHECK_EQ(int(E4), 28);
    CHECK_EQ(int(H8), 63);
    CHECK_EQ(~WHITE, BLACK);
    CHECK_EQ(~BLACK, WHITE);
}

TEST_CASE(move_encoding_round_trips_every_value) {
    for (u32 raw = 0; raw < 65536; ++raw) {
        const Move m = Move::from_raw(u16(raw));
        const Move rebuilt(m.from(), m.to(), m.flags());
        CHECK_EQ(rebuilt.raw(), u16(raw));
        CHECK_EQ(int(m.from()), int(raw & 63));
        CHECK_EQ(int(m.to()), int((raw >> 6) & 63));
        CHECK_EQ(m.flags(), raw >> 12);
    }
    CHECK(Move().is_none());
    CHECK(!Move(E2, E4, DOUBLE_PUSH).is_none());
}

TEST_CASE(move_flag_properties) {
    struct Expect {
        u32 flags;
        bool capture;
        bool promotion;
        PieceType promo;
    };
    const Expect table[] = {
        {QUIET, false, false, NO_PIECE_TYPE},
        {DOUBLE_PUSH, false, false, NO_PIECE_TYPE},
        {KING_CASTLE, false, false, NO_PIECE_TYPE},
        {QUEEN_CASTLE, false, false, NO_PIECE_TYPE},
        {CAPTURE, true, false, NO_PIECE_TYPE},
        {EP_CAPTURE, true, false, NO_PIECE_TYPE},
        {PROMO_KNIGHT, false, true, KNIGHT},
        {PROMO_BISHOP, false, true, BISHOP},
        {PROMO_ROOK, false, true, ROOK},
        {PROMO_QUEEN, false, true, QUEEN},
        {PROMO_CAPTURE_KNIGHT, true, true, KNIGHT},
        {PROMO_CAPTURE_BISHOP, true, true, BISHOP},
        {PROMO_CAPTURE_ROOK, true, true, ROOK},
        {PROMO_CAPTURE_QUEEN, true, true, QUEEN},
    };
    for (const Expect& e : table) {
        const Move m(B7, B8, e.flags);
        CHECK_EQ(m.is_capture(), e.capture);
        CHECK_EQ(m.is_promotion(), e.promotion);
        if (e.promotion) CHECK_EQ(m.promotion_type(), e.promo);
    }
}

TEST_CASE(state_word_round_trips) {
    for (u32 rights = 0; rights < 16; ++rights) {
        for (int ep = -1; ep < 8; ++ep) {
            for (int side = 0; side < 2; ++side) {
                for (int hm = 0; hm <= 127; ++hm) {
                    const u32 s = state::pack(rights, ep >= 0, ep >= 0 ? ep : 0, Color(side), hm);
                    CHECK_EQ(state::castling(s), rights);
                    CHECK_EQ(state::has_ep(s), ep >= 0);
                    if (ep >= 0) CHECK_EQ(state::ep_file(s), ep);
                    CHECK_EQ(state::side(s), Color(side));
                    CHECK_EQ(state::halfmove(s), hm);
                    CHECK_EQ(s >> 16, 0u);
                    if (ep < 0) CHECK_EQ((s >> state::kEpFileShift) & 15u, 0u);
                }
            }
        }
    }
    // Clamping.
    CHECK_EQ(state::halfmove(state::pack(0, false, 0, WHITE, 500)), 127);
    CHECK_EQ(state::halfmove(state::pack(0, false, 0, WHITE, -3)), 0);
    // En passant squares: white captures onto rank 6, black onto rank 3.
    CHECK_EQ(state::ep_square(state::pack(0, true, 3, WHITE, 0)), D6);
    CHECK_EQ(state::ep_square(state::pack(0, true, 4, BLACK, 0)), E3);
}

TEST_CASE(undo_round_trips) {
    for (u32 s = 0; s < 0x10000; s += 7) {
        for (int pt = NO_PIECE_TYPE; pt <= KING; ++pt) {
            const Undo u(s, PieceType(pt));
            CHECK_EQ(u.prev_state(), s);
            CHECK_EQ(u.captured(), PieceType(pt));
        }
    }
}

TEST_CASE(shifts_never_wrap) {
    for (int i = 0; i < 64; ++i) {
        const Square s = Square(i);
        const int f = file_of(s);
        const int r = rank_of(s);
        const auto expect = [&](int df, int dr) -> u64 {
            const int nf = f + df;
            const int nr = r + dr;
            return (nf >= 0 && nf < 8 && nr >= 0 && nr < 8) ? bit(make_square(nf, nr)) : 0;
        };
        CHECK_EQ(shift_north(bit(s)), expect(0, 1));
        CHECK_EQ(shift_south(bit(s)), expect(0, -1));
        CHECK_EQ(shift_east(bit(s)), expect(1, 0));
        CHECK_EQ(shift_west(bit(s)), expect(-1, 0));
        CHECK_EQ(shift_north_east(bit(s)), expect(1, 1));
        CHECK_EQ(shift_north_west(bit(s)), expect(-1, 1));
        CHECK_EQ(shift_south_east(bit(s)), expect(1, -1));
        CHECK_EQ(shift_south_west(bit(s)), expect(-1, -1));
        for (int c = 0; c < 2; ++c) {
            const Color color = Color(c);
            const int dr = color == WHITE ? 1 : -1;
            CHECK_EQ(shift_forward(bit(s), color), expect(0, dr));
            CHECK_EQ(shift_forward_west(bit(s), color), expect(-1, dr));
            CHECK_EQ(shift_forward_east(bit(s), color), expect(1, dr));
            // Offsets match the shifts wherever the target exists.
            if (expect(0, dr)) CHECK_EQ(lsb(expect(0, dr)) - i, forward_offset(color));
            if (expect(-1, dr)) CHECK_EQ(lsb(expect(-1, dr)) - i, forward_west_offset(color));
            if (expect(1, dr)) CHECK_EQ(lsb(expect(1, dr)) - i, forward_east_offset(color));
        }
    }
    for (int k = 0; k < 8; ++k) {
        CHECK_EQ(popcount(file_mask(k)), 8);
        CHECK_EQ(popcount(rank_mask(k)), 8);
        CHECK_EQ(file_mask(k) & rank_mask(k), bit(make_square(k, k)));
        CHECK_EQ(relative_rank(WHITE, k), k);
        CHECK_EQ(relative_rank(BLACK, k), 7 - k);
    }
}

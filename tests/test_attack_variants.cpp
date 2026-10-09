// Sliding-attack variants (docs/ARCHITECTURE.md, Move generation): magic
// bitboards (core/attacks_magic.h) and Kogge-Stone fills
// (core/attacks_kogge.h) must give exactly the attacks of rays walked square
// by square, as hyperbola quintessence does:
// - for every subset of blockers on every square's lines (edges included);
// - on random full boards (pieces off the lines and on the slider's square);
// - the magic tables have the fancy layout (each square's table sized by
//   its relevant squares, 107,648 entries in all) and their masks leave out
//   the edges;
// - the perft suites on the CPU give the expected counts with each variant,
//   on both boards.

#include <random>
#include <string>
#include <vector>

#include "core/attacks_kogge.h"
#include "core/attacks_magic.h"
#include "core/movegen.h"
#include "host/epd.h"
#include "host/fen.h"
#include "host/host_tables.h"
#include "host/perft_cpu.h"
#include "reference_chess.h"
#include "test_harness.h"
#include "test_util.h"

using namespace cb;

namespace {

const HqAttacks& hq() {
    static const HqAttacks a(host_attack_tables());
    return a;
}
const MagicAttacks& magic() {
    static const MagicAttacks a(host_attack_tables(), host_magic_tables());
    return a;
}
const KoggeStoneAttacks& kogge_stone() {
    static const KoggeStoneAttacks a(host_attack_tables());
    return a;
}

// Slider attacks by walking rays, independent of every variant.
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

template <class F>
void for_each_subset(u64 mask, F&& f) {
    u64 subset = 0;
    do {
        f(subset);
        subset = (subset - mask) & mask;
    } while (subset != 0);
}

template <class Attacks>
void check_exhaustive(const Attacks& a) {
    long long checked = 0, wrong = 0;
    for (int i = 0; i < 64; ++i) {
        const Square s = Square(i);
        for_each_subset(naive_rook(s, 0), [&](u64 occ) {
            if (a.rook(s, occ) != naive_rook(s, occ)) ++wrong;
            ++checked;
        });
        for_each_subset(naive_bishop(s, 0), [&](u64 occ) {
            if (a.bishop(s, occ) != naive_bishop(s, occ)) ++wrong;
            ++checked;
        });
    }
    CHECK_MSG(wrong == 0, std::string(Attacks::kName) + ": " + std::to_string(wrong) + " of " +
                              std::to_string(checked) + " blocker sets wrong");
    CHECK(checked > 500000);
}

template <class Attacks>
void check_random(const Attacks& a) {
    std::mt19937_64 rng(1234);
    int wrong = 0;
    for (int trial = 0; trial < 50000; ++trial) {
        const Square s = Square(rng() % 64);
        u64 occ = trial % 3 == 0 ? rng() : (trial % 3 == 1 ? rng() & rng() : rng() | rng());
        if (trial & 1) occ |= bit(s);
        if (a.rook(s, occ) != naive_rook(s, occ) || a.bishop(s, occ) != naive_bishop(s, occ) ||
            a.queen(s, occ) != (naive_rook(s, occ) | naive_bishop(s, occ)))
            ++wrong;
    }
    CHECK_MSG(wrong == 0, std::string(Attacks::kName) + ": " + std::to_string(wrong) + " random boards wrong");
}

template <class Board, class Attacks>
void check_perft_suite(const Attacks& a, const char* file, u64 max_nodes) {
    std::vector<PerftCase> cases;
    std::string error;
    REQUIRE_MSG(load_perft_suite(std::string(CHESSGPU_SOURCE_DIR "/tests/perft/") + file, cases, error), error);
    int checked = 0;
    for (const PerftCase& pc : cases) {
        const Position p = testutil::parse_or_die(pc.fen);
        for (const auto& [depth, expected] : pc.expected) {
            if (expected > max_nodes) break;
            Board b = to_board<Board>(p);
            const u64 got = perft_recursive(b, a, depth);
            CHECK_MSG(got == expected, std::string(Attacks::kName) + " " + Board::kName + " perft " +
                                           std::to_string(depth) + " of " + pc.fen + ": got " + std::to_string(got) +
                                           ", expected " + std::to_string(expected));
            ++checked;
        }
    }
    CHECK(checked >= int(2 * cases.size()));
}

#ifdef NDEBUG
constexpr u64 kPerftNodeLimit = 5'000'000;
#else
constexpr u64 kPerftNodeLimit = 200'000;
#endif

}  // namespace

TEST_CASE(attack_variants_exhaustive) {
    check_exhaustive(hq());
    check_exhaustive(magic());
    check_exhaustive(kogge_stone());
}

TEST_CASE(attack_variants_random_occupancy) {
    check_random(hq());
    check_random(magic());
    check_random(kogge_stone());
}

TEST_CASE(attack_variants_shared_lookups) {
    // Leapers, lines and empty-board sets come from the same tables in every variant.
    for (int i = 0; i < 64; ++i) {
        const Square s = Square(i);
        CHECK(magic().knight(s) == hq().knight(s) && kogge_stone().king(s) == hq().king(s));
        CHECK(magic().pawn(BLACK, s) == hq().pawn(BLACK, s) && kogge_stone().pawn(WHITE, s) == hq().pawn(WHITE, s));
        CHECK(magic().rook_empty_board(s) == naive_rook(s, 0) && kogge_stone().bishop_empty_board(s) == naive_bishop(s, 0));
        for (int j = 0; j < 64; ++j) {
            CHECK(magic().between(s, Square(j)) == hq().between(s, Square(j)));
            CHECK(kogge_stone().line(s, Square(j)) == hq().line(s, Square(j)));
        }
    }
}

TEST_CASE(attack_variants_magic_layout) {
    const MagicTables& m = host_magic_tables();
    constexpr u64 kEdges = 0xFF000000000000FFull | 0x8181818181818181ull;
    int expected_offset = 0;
    for (int i = 0; i < 64; ++i) {
        const Square s = Square(i);
        const MagicEntry& e = m.rook[i];
        // A rook's relevant squares: its lines without the last square of each ray.
        const u64 lines = naive_rook(s, 0);
        CHECK_EQ(e.mask & ~lines, u64(0));
        CHECK_EQ(64 - e.shift, popcount(e.mask));
        CHECK_EQ(e.offset, expected_offset);
        CHECK(popcount(lines) - popcount(e.mask) >= 2 && popcount(lines) - popcount(e.mask) <= 4);
        expected_offset += 1 << (64 - e.shift);
    }
    CHECK_EQ(expected_offset, kRookMagicEntries);
    for (int i = 0; i < 64; ++i) {
        const MagicEntry& e = m.bishop[i];
        CHECK_EQ(e.mask & kEdges, u64(0));  // a bishop's rays all end on the edge
        CHECK_EQ(e.mask & ~naive_bishop(Square(i), 0), u64(0));
        CHECK_EQ(e.offset, expected_offset);
        expected_offset += 1 << (64 - e.shift);
    }
    CHECK_EQ(expected_offset, kRookMagicEntries + kBishopMagicEntries);
    // The corner rook needs 12 relevant squares, the central bishop 9.
    CHECK_EQ(popcount(m.rook[A1].mask), 12);
    CHECK_EQ(popcount(m.bishop[E4].mask), 9);
}

TEST_CASE(attack_variants_perft) {
    check_perft_suite<QuadBoard>(magic(), "standard.epd", kPerftNodeLimit);
    check_perft_suite<StdBoard>(magic(), "edge_cases.epd", kPerftNodeLimit);
    check_perft_suite<QuadBoard>(kogge_stone(), "edge_cases.epd", kPerftNodeLimit);
    check_perft_suite<StdBoard>(kogge_stone(), "standard.epd", kPerftNodeLimit);
}

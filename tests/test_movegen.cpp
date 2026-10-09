// Legal move generation and CPU perft, cross-checked against the reference
// implementation, between generation paths, and against published counts.

#include <algorithm>

#include "core/perft.h"
#include "host/epd.h"
#include "host/host_tables.h"
#include "host/perft_cpu.h"
#include "test_harness.h"
#include "test_util.h"

using namespace cb;

namespace {

const HqAttacks& attacks() {
    static const HqAttacks a(host_attack_tables());
    return a;
}

std::string describe(const std::vector<u16>& raw) {
    std::string s;
    for (const u16 r : raw) s += move_to_uci(Move::from_raw(r)) + "(" + std::to_string(r >> 12) + ") ";
    return s;
}

// At every node: the fast generator, the per-piece generator and the counter
// must all agree with the reference move list (moves and flags).
template <class Board>
void check_generation_against_reference() {
    long long nodes = 0;
    for (const std::string& fen : testutil::test_fens()) {
        long long budget = testutil::kTreeBudget;
        auto visit = [&](const Position& p, const std::vector<Move>& ref_moves) {
            ++nodes;
            const Board b = to_board<Board>(p);
            const std::vector<u16> expected = testutil::sorted_raw(ref_moves);

            MoveList list;
            generate_legal(b, attacks(), list);
            const std::vector<u16> got = testutil::sorted_raw(list.moves, list.size);
            if (got != expected) {
                CHECK_MSG(false, std::string(Board::kName) + " generate_legal in " + format_fen(p) + "\n got      " +
                                     describe(got) + "\n expected " + describe(expected));
            }

            MoveList per_piece;
            generate_legal_per_piece(b, attacks(), per_piece);
            CHECK_MSG(testutil::sorted_raw(per_piece.moves, per_piece.size) == expected,
                      std::string(Board::kName) + " generate_legal_per_piece in " + format_fen(p));

            CHECK_MSG(count_legal(b, attacks()) == u32(expected.size()), format_fen(p));
        };
        testutil::walk_tree(testutil::parse_or_die(fen), 4, budget, visit);
    }
    CHECK(nodes > 1000);
}

}  // namespace

TEST_CASE(movegen_std_matches_reference) { check_generation_against_reference<StdBoard>(); }
TEST_CASE(movegen_quad_matches_reference) { check_generation_against_reference<QuadBoard>(); }

TEST_CASE(movegen_special_positions) {
    struct Case {
        const char* fen;
        int legal_moves;
    };
    // Counts verified by hand (and by the reference cross-check above).
    const Case cases[] = {
        {cb::kStartFen, 20},
        {"8/8/8/K2pP2r/8/8/8/7k w - d6 0 1", 6},       // exd6 illegal: 5 king moves + e6
        {"8/8/8/2k5/3Pp3/8/8/4K3 b - d3 0 1", 9},      // in check: exd3 e.p. + all 8 king moves
        {"k3r3/8/8/8/8/5n2/8/4K3 w - - 0 1", 3},       // double check: Kd1, Kf1, Kf2
        {"7k/5Q2/6K1/8/8/8/8/8 b - - 0 1", 0},         // stalemate
        {"R5k1/5ppp/8/8/8/8/8/6K1 b - - 0 1", 0},      // checkmate
    };
    for (const Case& c : cases) {
        const Position p = testutil::parse_or_die(c.fen);
        CHECK_MSG(int(count_legal(to_board<StdBoard>(p), attacks())) == c.legal_moves, c.fen);
        CHECK_MSG(int(count_legal(to_board<QuadBoard>(p), attacks())) == c.legal_moves, c.fen);
        CHECK_EQ(int(ref::legal_moves(ref::from_position(p)).size()), c.legal_moves);
    }
}

namespace {

// A perft suite on the CPU, up to a node limit per case.
template <class Board>
void check_perft_suite(const char* file, size_t expected_cases, u64 max_nodes) {
    std::vector<PerftCase> cases;
    std::string error;
    REQUIRE_MSG(load_perft_suite(std::string(CHESSGPU_SOURCE_DIR "/tests/perft/") + file, cases, error), error);
    REQUIRE(cases.size() == expected_cases);
    int checked = 0;
    for (const PerftCase& pc : cases) {
        const Position p = testutil::parse_or_die(pc.fen);
        for (const auto& [depth, expected] : pc.expected) {
            if (expected > max_nodes) break;
            Board b = to_board<Board>(p);
            const u64 got = perft_recursive(b, attacks(), depth);
            CHECK_MSG(got == expected, std::string(Board::kName) + " perft " + std::to_string(depth) + " of " +
                                           pc.fen + ": got " + std::to_string(got) + ", expected " +
                                           std::to_string(expected));
            CHECK(b == to_board<Board>(p));  // perft restores the board
            ++checked;
        }
    }
    CHECK(checked >= int(2 * expected_cases));
}

#ifdef NDEBUG
constexpr u64 kPerftNodeLimit = 20'000'000;
#else
constexpr u64 kPerftNodeLimit = 500'000;
#endif

}  // namespace

TEST_CASE(perft_suite_standard_std) { check_perft_suite<StdBoard>("standard.epd", 7, kPerftNodeLimit); }
TEST_CASE(perft_suite_standard_quad) { check_perft_suite<QuadBoard>("standard.epd", 7, kPerftNodeLimit); }
TEST_CASE(perft_suite_edge_cases_std) { check_perft_suite<StdBoard>("edge_cases.epd", 14, kPerftNodeLimit); }
TEST_CASE(perft_suite_edge_cases_quad) { check_perft_suite<QuadBoard>("edge_cases.epd", 14, kPerftNodeLimit); }

TEST_CASE(perft_iterative_matches_recursive) {
    // The explicit-stack loop the GPU thread model runs, on the CPU.
    static PerftStack<8> stack;  // ~4 KB; static to keep it off the test's stack
    for (const std::string& fen : testutil::test_fens()) {
        const Position p = testutil::parse_or_die(fen);
        for (int depth = 0; depth <= 3; ++depth) {
            StdBoard s = to_board<StdBoard>(p);
            QuadBoard q = to_board<QuadBoard>(p);
            const u64 expected = perft_recursive(s, attacks(), depth);
            CHECK_MSG(perft_iterative(s, attacks(), depth, stack) == expected, fen + " depth " + std::to_string(depth));
            CHECK_MSG(perft_iterative(q, attacks(), depth, stack) == expected, fen + " depth " + std::to_string(depth));
        }
    }
}

TEST_CASE(perft_divide_sums_to_total) {
    const Position p = testutil::parse_or_die("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1");
    StdBoard b = to_board<StdBoard>(p);
    const auto divide = perft_divide(b, attacks(), 3);
    CHECK_EQ(divide.size(), size_t(48));
    u64 total = 0;
    for (const auto& [move, nodes] : divide) total += nodes;
    CHECK_EQ(total, u64(97862));
    CHECK(std::is_sorted(divide.begin(), divide.end(), [](const auto& x, const auto& y) {
        return move_to_uci(x.first) < move_to_uci(y.first);
    }));
}

TEST_CASE(parse_uci_moves) {
    const Position p = testutil::parse_or_die("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1");
    const StdBoard b = to_board<StdBoard>(p);
    CHECK_EQ(parse_uci_move(b, attacks(), "e1g1"), Move(E1, G1, KING_CASTLE));
    CHECK_EQ(parse_uci_move(b, attacks(), "a1a8"), Move(A1, A8, CAPTURE));
    CHECK(parse_uci_move(b, attacks(), "e1e3").is_none());
    CHECK(parse_uci_move(b, attacks(), "garbage").is_none());
}

// 32-bit moves (core/move32.h), on the CPU:
// - the encoding keeps the 16-bit move and both piece types;
// - MoveBuffer32 lists exactly MoveBuffer's moves, in the same order, each
//   with the piece that moves and the piece it captures (a pawn for en
//   passant, none for quiet moves), on both boards;
// - make_move and unmake_move with a 32-bit move change the board, the undo
//   record and the observer's calls exactly as with the 16-bit move;
// - perft with 32-bit moves gives the suites' counts (CPU recursive and the
//   explicit-stack loop the GPU thread model runs).

#include <string>
#include <vector>

#include "core/move32.h"
#include "core/perft.h"
#include "host/epd.h"
#include "host/fen.h"
#include "host/host_tables.h"
#include "host/perft_cpu.h"
#include "host/uci_move.h"
#include "test_harness.h"
#include "test_util.h"

using namespace cb;

namespace {

const HqAttacks& attacks() {
    static const HqAttacks a(host_attack_tables());
    return a;
}

// Every piece change an observer hears, in order.
struct RecordingObserver {
    std::vector<int> calls;
    void add(Color c, PieceType pt, Square s) { calls.push_back(1000000 + int(c) * 10000 + int(pt) * 100 + int(s)); }
    void remove(Color c, PieceType pt, Square s) { calls.push_back(2000000 + int(c) * 10000 + int(pt) * 100 + int(s)); }
};

PieceType expected_capture(const QuadBoard& b, Move m) {
    if (m.flags() == EP_CAPTURE) return PAWN;
    return m.is_capture() ? b.piece_type_on(m.to()) : NO_PIECE_TYPE;
}

template <class Board>
void check_buffers_and_make_unmake() {
    long long nodes = 0, moves = 0;
    for (const std::string& fen : testutil::test_fens()) {
        long long budget = testutil::kTreeBudget / 4;
        auto visit = [&](const Position& p, const std::vector<Move>&) {
            ++nodes;
            const Board b = to_board<Board>(p);
            const QuadBoard q = to_board<QuadBoard>(p);
            MoveList list16;
            generate_legal(b, attacks(), list16);
            MoveBuffer32<Board, 256> list32;
            list32.bind(b);
            generate_legal(b, attacks(), list32);
            REQUIRE_MSG(list32.size == list16.size, fen);
            for (int i = 0; i < list16.size; ++i) {
                const Move m = list16.moves[i];
                const Move32 m32 = list32.moves[i];
                const std::string where = std::string(Board::kName) + " " + format_fen(p) + " " + move_to_uci(m);
                CHECK_MSG(m32.move() == m && move16(m32) == m, where);
                CHECK_MSG(m32.moving() == q.piece_type_on(m.from()) && m32.captured() == expected_capture(q, m),
                          where + ": pieces " + std::to_string(m32.moving()) + " " + std::to_string(m32.captured()));
                // Same board, undo record and observer calls both ways.
                Board a16 = b, a32 = b;
                RecordingObserver o16, o32;
                const Undo u16 = make_move(a16, m, o16);
                const Undo u32 = make_move(a32, m32, o32);
                CHECK_MSG(a16 == a32 && u16.prev_state() == u32.prev_state() && u16.captured() == u32.captured() &&
                              o16.calls == o32.calls,
                          where + ": make");
                unmake_move(a16, m, u16, o16);
                unmake_move(a32, m32, u32, o32);
                CHECK_MSG(a16 == b && a32 == b && o16.calls == o32.calls, where + ": unmake");
                ++moves;
            }
        };
        testutil::walk_tree(testutil::parse_or_die(fen), 3, budget, visit);
    }
    CHECK(nodes > 100 && moves > 1000);
}

template <class Board>
void check_perft_suite32(const char* file, u64 max_nodes) {
    std::vector<PerftCase> cases;
    std::string error;
    REQUIRE_MSG(load_perft_suite(std::string(CHESSGPU_SOURCE_DIR "/tests/perft/") + file, cases, error), error);
    int checked = 0;
    for (const PerftCase& pc : cases) {
        const Position p = testutil::parse_or_die(pc.fen);
        for (const auto& [depth, expected] : pc.expected) {
            if (expected > max_nodes) break;
            Board b = to_board<Board>(p);
            const u64 got = perft_recursive<Moves32>(b, attacks(), depth);
            CHECK_MSG(got == expected, std::string(Board::kName) + " 32-bit perft " + std::to_string(depth) + " of " +
                                           pc.fen + ": got " + std::to_string(got) + ", expected " +
                                           std::to_string(expected));
            CHECK(b == to_board<Board>(p));
            ++checked;
        }
    }
    CHECK(checked >= int(cases.size()));
}

#ifdef NDEBUG
constexpr u64 kNodeLimit = 5'000'000;
#else
constexpr u64 kNodeLimit = 200'000;
#endif

}  // namespace

TEST_CASE(move32_encoding) {
    const Move m(E7, D8, PROMO_CAPTURE_QUEEN);
    const Move32 x(m, PAWN, ROOK);
    CHECK(x.move() == m && x.moving() == PAWN && x.captured() == ROOK);
    CHECK(x.from() == E7 && x.to() == D8 && x.flags() == PROMO_CAPTURE_QUEEN && x.is_capture() && x.is_promotion() &&
          x.promotion_type() == QUEEN);
    CHECK(Move32().is_none() && !x.is_none());
    CHECK(Move32(Move(G1, F3, QUIET), KNIGHT, NO_PIECE_TYPE).captured() == NO_PIECE_TYPE);
    CHECK(Move32(Move(E1, G1, KING_CASTLE), KING, NO_PIECE_TYPE).moving() == KING);
    CHECK(x != Move32(m, PAWN, BISHOP));  // the pieces are part of the value
}

TEST_CASE(move32_buffers_and_make_unmake_std) { check_buffers_and_make_unmake<StdBoard>(); }
TEST_CASE(move32_buffers_and_make_unmake_quad) { check_buffers_and_make_unmake<QuadBoard>(); }

TEST_CASE(move32_perft_suites) {
    check_perft_suite32<StdBoard>("standard.epd", kNodeLimit);
    check_perft_suite32<QuadBoard>("standard.epd", kNodeLimit);
    check_perft_suite32<StdBoard>("edge_cases.epd", kNodeLimit);
    check_perft_suite32<QuadBoard>("edge_cases.epd", kNodeLimit);
}

TEST_CASE(move32_perft_iterative_and_divide) {
    // The explicit-stack loop of the GPU thread model, and divide's root moves.
    static PerftStack<8, MoveBuffer32<QuadBoard, 256>> stack32;
    static PerftStack<8, MoveBuffer32<StdBoard, 256>> stack32_std;
    for (const std::string& fen : testutil::test_fens()) {
        const Position p = testutil::parse_or_die(fen);
        for (int depth = 0; depth <= 3; ++depth) {
            QuadBoard q = to_board<QuadBoard>(p);
            StdBoard s = to_board<StdBoard>(p);
            const u64 expected = perft_recursive(q, attacks(), depth);
            CHECK_MSG(perft_iterative(q, attacks(), depth, stack32) == expected, fen + " depth " + std::to_string(depth));
            CHECK_MSG(perft_iterative(s, attacks(), depth, stack32_std) == expected,
                      fen + " depth " + std::to_string(depth));
        }
    }
    const Position kiwipete =
        testutil::parse_or_die("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1");
    QuadBoard b = to_board<QuadBoard>(kiwipete);
    CHECK(perft_divide<Moves32>(b, attacks(), 3) == perft_divide(b, attacks(), 3));
}

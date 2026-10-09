// FEN parsing, formatting and position validation.

#include "host/fen.h"
#include "host/position_moves.h"
#include "host/uci_move.h"
#include "test_harness.h"
#include "test_util.h"

using namespace cb;

TEST_CASE(fen_round_trips) {
    for (const std::string& fen : testutil::test_fens()) {
        Position p;
        std::string error;
        REQUIRE_MSG(parse_fen(fen, p, error), fen + " -> " + error);
        CHECK_EQ(format_fen(p), fen);
    }
}

TEST_CASE(fen_start_position_contents) {
    const Position p = testutil::parse_or_die(kStartFen);
    CHECK_EQ(int(p.squares[E1]), int(piece_code(WHITE, KING)));
    CHECK_EQ(int(p.squares[D8]), int(piece_code(BLACK, QUEEN)));
    CHECK_EQ(int(p.squares[A2]), int(piece_code(WHITE, PAWN)));
    CHECK_EQ(int(p.squares[H7]), int(piece_code(BLACK, PAWN)));
    CHECK_EQ(int(p.squares[E4]), 0);
    CHECK_EQ(state::side(p.state), WHITE);
    CHECK_EQ(state::castling(p.state), u32(WHITE_OO | WHITE_OOO | BLACK_OO | BLACK_OOO));
    CHECK(!state::has_ep(p.state));
    CHECK_EQ(state::halfmove(p.state), 0);
    CHECK_EQ(p.fullmove, 1);
}

TEST_CASE(fen_optional_counters_and_whitespace) {
    const Position four = testutil::parse_or_die("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq -");
    CHECK_EQ(state::halfmove(four.state), 0);
    CHECK_EQ(four.fullmove, 1);
    const Position five = testutil::parse_or_die("4k3/8/8/8/8/8/8/4K3 b - - 17");
    CHECK_EQ(state::halfmove(five.state), 17);
    CHECK_EQ(five.fullmove, 1);
    const Position spaced = testutil::parse_or_die("  4k3/8/8/8/8/8/8/4K3   w  -  -  3   9  ");
    CHECK_EQ(format_fen(spaced), "4k3/8/8/8/8/8/8/4K3 w - - 3 9");
    // Castling letters in any order, printed canonically.
    CHECK_EQ(format_fen(testutil::parse_or_die("r3k2r/8/8/8/8/8/8/R3K2R w qkQK - 0 1")),
             "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1");
    // A halfmove clock above the state word's range is clamped.
    CHECK_EQ(state::halfmove(testutil::parse_or_die("4k3/8/8/8/8/8/8/4K3 w - - 300 200").state), 127);
}

TEST_CASE(fen_rejects_malformed_input) {
    const char* bad[] = {
        "",
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq",          // too few fields
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1 x",  // too many fields
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP w KQkq - 0 1",             // 7 ranks
        "rnbqkbnr/pppppppp/8/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",  // 9 ranks
        "rnbqkbnr/ppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",     // short rank
        "rnbqkbnr/ppppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",   // long rank
        "rnbqkbnr/pppppppp/9/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",    // digit 9
        "rnbqkbnr/pppppppp/44/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",   // 4+4 is fine...
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNX w KQkq - 0 1",    // bad letter
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR x KQkq - 0 1",    // bad side
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkx - 0 1",    // bad castling letter
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KKkq - 0 1",    // repeated castling letter
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq e9 0 1",   // bad square
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq e3 0 1",   // ep on the wrong rank
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq e6 0 1",   // no pawn pushed there
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - -1 1",   // negative clock
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 x",    // bad fullmove
        "rnbq1bnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQ - 0 1",      // black king missing
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKKNR w KQkq - 0 1",    // two white kings
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNP w Qkq - 0 1",     // pawn on rank 1
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBN1 w KQkq - 0 1",    // K right without h1 rook
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQ1BNR w KQkq - 0 1",    // white king missing
        "4k3/8/8/8/8/8/8/4K2R b - - 0 1",                              // fine, used below
        "4k2R/8/8/8/8/8/8/4K3 w - - 0 1",                              // side not to move in check
    };
    for (const char* fen : bad) {
        const std::string text = fen;
        // Two lines above are actually valid; they document what is accepted.
        if (text == "rnbqkbnr/pppppppp/44/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1" ||
            text == "4k3/8/8/8/8/8/8/4K2R b - - 0 1")
            continue;
        Position p;
        std::string error;
        CHECK_MSG(!parse_fen(text, p, error), "accepted: " + text);
        if (!parse_fen(text, p, error)) CHECK_MSG(!error.empty(), "no error message for: " + text);
    }
    // Consecutive digits are allowed as long as the rank sums to 8.
    Position p;
    std::string error;
    CHECK(parse_fen("rnbqkbnr/pppppppp/44/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", p, error));
    CHECK_EQ(format_fen(p), std::string(kStartFen));
}

TEST_CASE(fen_failed_parse_leaves_output_untouched) {
    Position p = testutil::parse_or_die(kStartFen);
    const Position before = p;
    std::string error;
    CHECK(!parse_fen("not a fen at all", p, error));
    CHECK(p == before);
}

TEST_CASE(apply_uci_moves_plays_and_validates) {
    std::string error;
    Position p = testutil::parse_or_die(kStartFen);
    REQUIRE_MSG(apply_uci_moves(p, "e2e4 e7e5 g1f3", error), error);
    CHECK_EQ(format_fen(p), "rnbqkbnr/pppp1ppp/8/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 1 2");

    Position q = testutil::parse_or_die(kStartFen);
    REQUIRE_MSG(apply_uci_moves(q, "e2e4", error), error);
    CHECK_EQ(format_fen(q), "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1");
    const Position before = q;
    CHECK(!apply_uci_moves(q, "e7e5 e2e4", error));  // e2 is empty by then
    CHECK(!error.empty());
    CHECK(q == before);  // unchanged on failure
    CHECK(apply_uci_moves(q, "", error));
    CHECK(q == before);  // no moves, no change

    Position castle = testutil::parse_or_die("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1");
    REQUIRE_MSG(apply_uci_moves(castle, "e1c1 e8g8", error), error);
    CHECK_EQ(format_fen(castle), "r4rk1/8/8/8/8/8/8/2KR3R w - - 2 2");

    Position promo = testutil::parse_or_die("4k3/1P6/8/8/8/8/8/4K3 w - - 0 1");
    REQUIRE_MSG(apply_uci_moves(promo, "b7b8n", error), error);
    CHECK_EQ(format_fen(promo), "1N2k3/8/8/8/8/8/8/4K3 b - - 0 1");
}

TEST_CASE(square_names) {
    for (int i = 0; i < 64; ++i) {
        Square s;
        REQUIRE(parse_square(square_name(Square(i)), s));
        CHECK_EQ(int(s), i);
    }
    Square s;
    CHECK(!parse_square("i1", s));
    CHECK(!parse_square("a0", s));
    CHECK(!parse_square("a9", s));
    CHECK(!parse_square("e", s));
    CHECK(!parse_square("e44", s));
    CHECK_EQ(square_name(E4), "e4");
    CHECK_EQ(move_to_uci(Move(E7, E8, PROMO_QUEEN)), "e7e8q");
    CHECK_EQ(move_to_uci(Move(B2, A1, PROMO_CAPTURE_KNIGHT)), "b2a1n");
    CHECK_EQ(move_to_uci(Move(E1, G1, KING_CASTLE)), "e1g1");
    CHECK_EQ(move_to_uci(Move()), "0000");
}

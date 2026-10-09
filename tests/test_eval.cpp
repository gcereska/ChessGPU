// Evaluation, Zobrist keys and their incremental updates in make/unmake.

#include <map>
#include <string>

#include "core/search_board.h"
#include "host/host_tables.h"
#include "host/position_moves.h"
#include "host/uci_move.h"
#include "test_harness.h"
#include "test_util.h"

using namespace cb;

namespace {

const HqAttacks& attacks() {
    static const HqAttacks a(host_attack_tables());
    return a;
}

const SearchTables& tables() { return host_search_tables(); }

template <class Eval, class Board>
int eval_of(const Position& p) {
    const Board b = to_board<Board>(p);
    return evaluate_position<Eval>(b, compute_accumulator<Eval>(b, tables().eval), state::side(p.state));
}

u64 key_of(const Position& p) { return compute_key(to_board<StdBoard>(p), attacks(), tables().zobrist); }
u64 key_of(const std::string& fen) { return key_of(testutil::parse_or_die(fen)); }

// What makes two positions the same for the repetition rule: placement, side
// to move, castling rights, and the en passant file only if a capture is legal.
std::string repetition_identity(const Position& p) {
    const StdBoard b = to_board<StdBoard>(p);
    std::string id(p.squares.begin(), p.squares.end());
    id += char(state::side(p.state));
    id += char(state::castling(p.state));
    id += en_passant_capture_possible(b, attacks()) ? char('a' + state::ep_file(p.state)) : '-';
    return id;
}

}  // namespace

TEST_CASE(eval_pack_mg_eg_round_trips) {
    const int values[] = {-32000, -20000, -12345, -900, -1, 0, 1, 99, 900, 12345, 20000, 32000};
    for (const int mg : values) {
        for (const int eg : values) {
            const int packed = pack_mg_eg(mg, eg);
            CHECK_EQ(unpack_mg(packed), mg);
            CHECK_EQ(unpack_eg(packed), eg);
        }
    }
    // Sums of packed values unpack to the sums, while each half fits in 16 bits.
    int sum = 0;
    int mg_sum = 0;
    int eg_sum = 0;
    for (int i = 0; i < 200; ++i) {
        const int mg = (i * 37) % 301 - 150;
        const int eg = (i * 53) % 257 - 128;
        sum += pack_mg_eg(mg, eg);
        mg_sum += mg;
        eg_sum += eg;
        CHECK_EQ(unpack_mg(sum), mg_sum);
        CHECK_EQ(unpack_eg(sum), eg_sum);
    }
    for (int i = 0; i < 200; ++i) {
        const int mg = (i * 37) % 301 - 150;
        const int eg = (i * 53) % 257 - 128;
        sum -= pack_mg_eg(mg, eg);
    }
    CHECK_EQ(sum, 0);
}

TEST_CASE(eval_table_spot_values) {
    const EvalTables& t = tables().eval;
    // White pawn on e2: PeSTO table entry 52 (white reads s ^ 56).
    CHECK_EQ(unpack_mg(t.pesto[WHITE][PAWN][E2]), 82 - 15);
    CHECK_EQ(unpack_eg(t.pesto[WHITE][PAWN][E2]), 94 + 13);
    // Black pawn on e7 is its mirror image: the same numbers, negated.
    CHECK_EQ(unpack_mg(t.pesto[BLACK][PAWN][E7]), -(82 - 15));
    CHECK_EQ(unpack_eg(t.pesto[BLACK][PAWN][E7]), -(94 + 13));
    // White knight on g1 and white king on g1: entry 62.
    CHECK_EQ(unpack_mg(t.pesto[WHITE][KNIGHT][G1]), 337 - 19);
    CHECK_EQ(unpack_eg(t.pesto[WHITE][KNIGHT][G1]), 281 - 50);
    CHECK_EQ(unpack_mg(t.pesto[WHITE][KING][G1]), 24);
    CHECK_EQ(unpack_eg(t.pesto[WHITE][KING][G1]), -24);
    // Black queen on d8: black reads entry s = 59, the queen's home square.
    CHECK_EQ(unpack_mg(t.pesto[BLACK][QUEEN][D8]), -(1025 + 10));
    CHECK_EQ(unpack_eg(t.pesto[BLACK][QUEEN][D8]), -(936 - 43));
    // White rook on a7 (entry 8) and black rook on a2 (entry 8).
    CHECK_EQ(unpack_mg(t.pesto[WHITE][ROOK][A7]), 477 + 27);
    CHECK_EQ(unpack_mg(t.pesto[BLACK][ROOK][A2]), -(477 + 27));
    for (int s = 0; s < 64; ++s) {
        CHECK_EQ(t.pesto[WHITE][NO_PIECE_TYPE][s], 0);
        CHECK_EQ(t.pesto[BLACK][NO_PIECE_TYPE][s], 0);
    }

    CHECK_EQ(t.material[WHITE][PAWN], 100);
    CHECK_EQ(t.material[WHITE][QUEEN], 900);
    CHECK_EQ(t.material[BLACK][ROOK], -500);
    CHECK_EQ(t.material[BLACK][KNIGHT], -320);
    CHECK_EQ(t.material[WHITE][KING], 0);
    CHECK_EQ(t.phase[PAWN], 0);
    CHECK_EQ(t.phase[KNIGHT], 1);
    CHECK_EQ(t.phase[BISHOP], 1);
    CHECK_EQ(t.phase[ROOK], 2);
    CHECK_EQ(t.phase[QUEEN], 4);
    CHECK_EQ(t.phase[KING], 0);
}

TEST_CASE(eval_known_positions) {
    const Position start = testutil::parse_or_die(kStartFen);
    CHECK_EQ((eval_of<MaterialEval, StdBoard>(start)), 0);
    CHECK_EQ((eval_of<PestoEval, QuadBoard>(start)), 0);
    CHECK_EQ(compute_accumulator<PestoEval>(to_board<QuadBoard>(start), tables().eval).phase, 24);

    // King and queen against king. PeSTO: phase 4 of 24. The kings cancel
    // (mirror squares); the queen on d1 scores mg 1035, eg 893, and
    // (1035 * 4 + 893 * 20) / 24 = 916.67, truncated to 916.
    const Position kqk_w = testutil::parse_or_die("4k3/8/8/8/8/8/8/3QK3 w - - 0 1");
    const Position kqk_b = testutil::parse_or_die("4k3/8/8/8/8/8/8/3QK3 b - - 0 1");
    CHECK_EQ((eval_of<MaterialEval, StdBoard>(kqk_w)), 900);
    CHECK_EQ((eval_of<MaterialEval, QuadBoard>(kqk_b)), -900);
    CHECK_EQ((eval_of<PestoEval, StdBoard>(kqk_w)), 916);
    CHECK_EQ((eval_of<PestoEval, QuadBoard>(kqk_b)), -916);

    // Phase is capped at 24 after a promotion: three queens each.
    const Position promoted = testutil::parse_or_die("qqq1k3/8/8/8/8/8/8/QQQ1K3 w - - 0 1");
    const auto acc = compute_accumulator<PestoEval>(to_board<StdBoard>(promoted), tables().eval);
    CHECK_EQ(acc.phase, 24);
    const Position more = testutil::parse_or_die("qqqqk3/8/8/8/8/8/8/QQQQK3 w - - 0 1");
    CHECK_EQ(compute_accumulator<PestoEval>(to_board<StdBoard>(more), tables().eval).phase, 32);
    // Seven queens (phase 28): capped, the score is the pure middlegame score.
    const Position lopsided = testutil::parse_or_die("qq2k3/8/8/8/8/8/8/QQQQK2Q w - - 0 1");
    const auto lopsided_acc = compute_accumulator<PestoEval>(to_board<StdBoard>(lopsided), tables().eval);
    CHECK_EQ(lopsided_acc.phase, 28);
    CHECK(unpack_mg(lopsided_acc.packed) != unpack_eg(lopsided_acc.packed));
    CHECK_EQ(PestoEval::evaluate(lopsided_acc, WHITE), unpack_mg(lopsided_acc.packed));
    CHECK_EQ(PestoEval::evaluate(lopsided_acc, BLACK), -unpack_mg(lopsided_acc.packed));
}

namespace {

template <class Eval, class Board>
void check_mirror_symmetry() {
    long long nodes = 0;
    for (const std::string& fen : testutil::test_fens()) {
        long long budget = testutil::kTreeBudget / 4;
        auto visit = [&](const Position& p, const std::vector<Move>&) {
            ++nodes;
            const Position m = testutil::mirror(p);
            std::string error;
            REQUIRE_MSG(validate_position(m, error), format_fen(m) + ": " + error);
            CHECK_MSG((eval_of<Eval, Board>(p)) == (eval_of<Eval, Board>(m)),
                      std::string(Eval::kName) + " " + format_fen(p) + " vs " + format_fen(m));
        };
        testutil::walk_tree(testutil::parse_or_die(fen), 3, budget, visit);
    }
    CHECK(nodes > 1000);
}

}  // namespace

TEST_CASE(eval_material_mirror_symmetry) { check_mirror_symmetry<MaterialEval, StdBoard>(); }
TEST_CASE(eval_pesto_mirror_symmetry) { check_mirror_symmetry<PestoEval, QuadBoard>(); }

namespace {

// At every node: for each legal move, the incremental key and accumulator
// after make_search_move equal the ones computed from scratch, and
// unmake_search_move restores the original exactly.
template <class Eval, class Board>
void check_incremental_updates() {
    long long moves_checked = 0;
    for (const std::string& fen : testutil::test_fens()) {
        long long budget = testutil::kTreeBudget / 2;
        auto visit = [&](const Position& p, const std::vector<Move>& ref_moves) {
            const Board b = to_board<Board>(p);
            const SearchBoard<Board, Eval> sb = make_search_board<Eval>(b, attacks(), tables());
            for (const Move m : ref_moves) {
                SearchBoard<Board, Eval> child = sb;
                const Undo undo = make_search_move(child, m, attacks(), tables());
                Board expected_board = b;
                make_move(expected_board, m);
                const SearchBoard<Board, Eval> expected = make_search_board<Eval>(expected_board, attacks(), tables());
                const std::string where = std::string(Board::kName) + "/" + Eval::kName + " " + move_to_uci(m) +
                                          " in " + format_fen(p);
                CHECK_MSG(child.board == expected.board, where);
                CHECK_MSG(child.key == expected.key, where + " (key)");
                CHECK_MSG(child.acc == expected.acc, where + " (accumulator)");
                unmake_search_move(child, m, undo, sb.key, tables());
                CHECK_MSG(child.board == sb.board && child.key == sb.key && child.acc == sb.acc, where + " (unmake)");
                ++moves_checked;
            }
        };
        testutil::walk_tree(testutil::parse_or_die(fen), 3, budget, visit);
    }
    CHECK(moves_checked > 10000);
}

}  // namespace

TEST_CASE(incremental_material_std) { check_incremental_updates<MaterialEval, StdBoard>(); }
TEST_CASE(incremental_material_quad) { check_incremental_updates<MaterialEval, QuadBoard>(); }
TEST_CASE(incremental_pesto_std) { check_incremental_updates<PestoEval, StdBoard>(); }
TEST_CASE(incremental_pesto_quad) { check_incremental_updates<PestoEval, QuadBoard>(); }

TEST_CASE(zobrist_key_components) {
    const ZobristKeys& z = tables().zobrist;
    CHECK_EQ(key_of("8/8/8/8/8/8/8/K6k w - - 0 1") ^ key_of("8/8/8/8/8/8/8/K6k b - - 0 1"), z.side);
    CHECK_EQ(key_of("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1") ^ key_of("r3k2r/8/8/8/8/8/8/R3K2R w Kk - 0 1"),
             z.castling[15] ^ z.castling[WHITE_OO | BLACK_OO]);
    // The halfmove clock and fullmove number are not part of the key.
    CHECK_EQ(key_of("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1"), key_of("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 37 60"));
    // Different placements, same everything else.
    CHECK(key_of("8/8/8/8/8/8/8/K6k w - - 0 1") != key_of("8/8/8/8/8/8/K7/7k w - - 0 1"));

    // En passant square after 1. e4: no black pawn can capture, so the key is
    // the same as without it.
    CHECK_EQ(key_of("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"),
             key_of("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"));
    // The pawn on d4 can capture en passant: the file is part of the key.
    CHECK_EQ(key_of("rnbqkbnr/ppp1pppp/8/8/3pP3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 3") ^
                 key_of("rnbqkbnr/ppp1pppp/8/8/3pP3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 3"),
             z.ep_file[4]);
    // exd3 would expose the black king on a4 to the queen on h4: illegal, so
    // the en passant square is ignored.
    CHECK_EQ(key_of("8/8/8/8/k2Pp2Q/8/8/3K4 b - d3 0 1"), key_of("8/8/8/8/k2Pp2Q/8/8/3K4 b - - 0 1"));
}

TEST_CASE(zobrist_repetition_after_double_push) {
    // 1. e4 Nf6 2. Nf3 Ng8 3. Ng1 repeats the position after 1. e4: its
    // en passant square was unusable, so the keys must match.
    Position p = testutil::parse_or_die(kStartFen);
    QuadBoard b = to_board<QuadBoard>(p);
    SearchBoard<QuadBoard, PestoEval> sb = make_search_board<PestoEval>(b, attacks(), tables());
    std::vector<u64> keys = {sb.key};
    for (const char* text : {"e2e4", "g8f6", "g1f3", "f6g8", "f3g1"}) {
        const Move m = parse_uci_move(sb.board, attacks(), text);
        REQUIRE(!m.is_none());
        make_search_move(sb, m, attacks(), tables());
        CHECK_EQ(sb.key, compute_key(sb.board, attacks(), tables().zobrist));
        keys.push_back(sb.key);
    }
    CHECK_EQ(keys[5], keys[1]);
    CHECK(keys[5] != keys[0]);
    CHECK(keys[3] != keys[1]);
}

TEST_CASE(zobrist_keys_identify_positions) {
    // Over many positions: equal keys exactly when the positions are the
    // same under the repetition rule.
    std::map<u64, std::string> by_key;
    std::map<std::string, u64> by_identity;
    for (const std::string& fen : testutil::test_fens()) {
        long long budget = testutil::kTreeBudget / 2;
        auto visit = [&](const Position& p, const std::vector<Move>&) {
            const u64 k = key_of(p);
            const std::string id = repetition_identity(p);
            const auto [kit, new_key] = by_key.emplace(k, id);
            CHECK_MSG(new_key || kit->second == id, "key collision at " + format_fen(p));
            const auto [iit, new_id] = by_identity.emplace(id, k);
            CHECK_MSG(new_id || iit->second == k, "same position, different keys: " + format_fen(p));
        };
        testutil::walk_tree(testutil::parse_or_die(fen), 3, budget, visit);
    }
    CHECK(by_key.size() > 10000);
}

TEST_CASE(insufficient_material_cases) {
    const auto dead = [](const char* fen) { return insufficient_material(to_board<QuadBoard>(testutil::parse_or_die(fen))); };
    CHECK(dead("8/8/8/4k3/8/8/8/4K3 w - - 0 1"));
    CHECK(dead("8/8/8/4k3/8/8/8/4KN2 w - - 0 1"));
    CHECK(dead("8/8/8/4k3/8/8/8/4K1b1 b - - 0 1"));
    CHECK(!dead("8/8/8/4k3/8/8/8/3NKN2 w - - 0 1"));  // two knights: mate is possible
    CHECK(!dead("8/8/8/4k1b1/8/8/8/4KB2 w - - 0 1"));
    CHECK(!dead("8/8/8/4k3/8/8/4P3/4K3 w - - 0 1"));
    CHECK(!dead("8/8/8/4k3/8/8/8/4K2R w - - 0 1"));
    CHECK(!dead("8/8/8/4k3/8/8/8/3qK3 w - - 0 1"));
}

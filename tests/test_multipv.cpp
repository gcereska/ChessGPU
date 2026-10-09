// MultiPV (host/multipv.h):
// - root children: one per legal move, with the root appended to the
//   history, and the draws the search would score one ply after the root
//   (a third occurrence, insufficient material, the 50-move rule);
// - without a table (exact, order-independent scores), every line's score
//   is its move's score from a separate full-window search of the position
//   after it, the lines are the best moves in order, and the first line's
//   score is the normal search's;
// - mates count one ply further from the root; a draw on arrival scores 0;
//   asking for more lines than moves lists every move; a stop right after
//   "go" still finishes depth 1; MultiPV 1 is the normal search.

#include <algorithm>
#include <string>
#include <vector>

#include "host/fen.h"
#include "host/game_history.h"
#include "host/host_tables.h"
#include "host/multipv.h"
#include "host/uci_move.h"
#include "test_harness.h"
#include "test_util.h"

using namespace cb;

namespace {

#ifdef NDEBUG
constexpr int kDepth = 4;
#else
constexpr int kDepth = 3;
#endif

Move parse(const Position& p, const std::string& uci) {
    return parse_uci_move(to_board<QuadBoard>(p), HqAttacks(host_attack_tables()), uci);
}

GameHistory game(const std::string& moves, Position& root) {
    GameHistory h;
    std::string error;
    if (!build_game(testutil::parse_or_die(kStartFen), moves, root, h, error)) std::abort();
    return h;
}

std::vector<RootLine> last_lines(SearchBackend& backend, const Position& root, const GameHistory& history,
                                 int depth, int multipv, SearchOutcome* outcome = nullptr) {
    SearchRequest request;
    request.root = root;
    request.history = history;
    request.max_depth = depth;
    request.multipv = multipv;
    std::vector<RootLine> lines;
    const SearchOutcome o = iterative_deepening(backend, request, [&](const IterationReport& r) { lines = r.lines; });
    if (outcome) *outcome = o;
    return lines;
}

// Every root move's score from its own full-window search (draws on arrival: 0).
std::vector<std::pair<Move, int>> child_scores(SearchBackend& backend, const Position& root,
                                               const GameHistory& history, int depth) {
    std::vector<std::pair<Move, int>> out;
    for (const RootChild& c : root_children(root, history)) {
        int score = kScoreDraw;
        if (!c.draw) score = score_from_child(backend.search_depth(c.position, c.history, depth - 1, {}).search.score);
        out.push_back({c.move, score});
    }
    return out;
}

bool line_is_legal(const Position& root, const std::vector<Move>& pv) {
    QuadBoard b = to_board<QuadBoard>(root);
    const HqAttacks a(host_attack_tables());
    for (const Move m : pv) {
        MoveList list;
        generate_legal(b, a, list);
        if (std::find(list.moves, list.moves + list.size, m) == list.moves + list.size) return false;
        make_move(b, m);
    }
    return true;
}

}  // namespace

TEST_CASE(multipv_root_children) {
    const Position start = testutil::parse_or_die(kStartFen);
    const std::vector<RootChild> children = root_children(start, GameHistory{});
    REQUIRE(children.size() == 20);
    for (const RootChild& c : children) {
        CHECK(!c.draw && c.history.root_repeated && c.history.entries.size() == 1);
        CHECK(c.history.entries[0].ply == -1 && c.history.entries[0].prev == -1 && c.history.entries[0].repeated == 0);
    }
    const auto e4 = std::find_if(children.begin(), children.end(),
                                 [&](const RootChild& c) { return c.move == parse(start, "e2e4"); });
    REQUIRE(e4 != children.end());
    CHECK(format_fen(e4->position) == "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1");
    // After Black's move the move number goes up; earlier game positions keep their order, one ply further back.
    Position root;
    const GameHistory h = game("e2e4 e7e5 g1f3", root);
    const std::vector<RootChild> after = root_children(root, h);
    const auto nc6 = std::find_if(after.begin(), after.end(),
                                  [&](const RootChild& c) { return c.move == parse(root, "b8c6"); });
    REQUIRE(nc6 != after.end());
    CHECK(format_fen(nc6->position) == "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3");
    REQUIRE(nc6->history.entries.size() == 4);
    for (int i = 0; i < 4; ++i) CHECK(nc6->history.entries[size_t(i)].ply == i - 4);
    CHECK(nc6->history.entries[3].prev == 2 && nc6->history.entries[3].repeated == 0);
    // A root that occurred before is marked so in its children's histories.
    const GameHistory again = game("g1f3 g8f6 f3g1 f6g8", root);
    REQUIRE(again.root_repeated);
    for (const RootChild& c : root_children(root, again)) CHECK(c.history.entries.back().repeated == 1);
}

TEST_CASE(multipv_draws_on_arrival) {
    // A third occurrence: 4... Ng8 brings back the start position a third time.
    Position root;
    const GameHistory h = game("g1f3 g8f6 f3g1 f6g8 g1f3 g8f6 f3g1", root);
    for (const RootChild& c : root_children(root, h))
        CHECK_MSG(c.draw == (c.move == parse(root, "f6g8")), move_to_uci(c.move));
    // Only twice so far: not yet a draw (the search allows one repetition of a game position).
    const GameHistory once = game("g1f3 g8f6 f3g1", root);
    for (const RootChild& c : root_children(root, once)) CHECK_MSG(!c.draw, move_to_uci(c.move));
    // Insufficient material: Kxd3 leaves the kings alone.
    const Position bare = testutil::parse_or_die("8/8/8/4k3/8/3pK3/8/8 w - - 0 1");
    for (const RootChild& c : root_children(bare, GameHistory{}))
        CHECK_MSG(c.draw == (c.move == parse(bare, "e3d3")), move_to_uci(c.move));
    // The 50-move rule: every quiet move makes the clock 100, except a pawn move.
    const Position fifty = testutil::parse_or_die("4k3/8/8/8/8/8/P7/4K2R w K - 99 80");
    for (const RootChild& c : root_children(fifty, GameHistory{}))
        CHECK_MSG(c.draw == (c.move != parse(fifty, "a2a3") && c.move != parse(fifty, "a2a4")), move_to_uci(c.move));
    // ... but a checkmate on the hundredth half-move stands.
    const Position mate = testutil::parse_or_die("6k1/5ppp/8/8/8/8/8/R5K1 w - - 99 80");
    for (const RootChild& c : root_children(mate, GameHistory{}))
        CHECK_MSG(c.draw == (c.move != parse(mate, "a1a8")), move_to_uci(c.move));
}

TEST_CASE(multipv_scores) {
    CHECK(score_from_child(35) == -35 && score_to_child(-35) == 35);
    // Mated at the child's root: mate in one ply from the root, and back.
    CHECK(score_from_child(-kScoreMate) == kScoreMate - 1 && score_to_child(kScoreMate - 1) == -kScoreMate);
    CHECK(score_from_child(kScoreMate - 2) == -kScoreMate + 3 && score_to_child(-kScoreMate + 3) == kScoreMate - 2);
}

TEST_CASE(multipv_matches_separate_searches_without_a_table) {
    CpuBackend backend(CpuSearchOptions{BoardKind::Quad, EvalKind::Pesto, OrderingKind::Staged});
    int fens = 0;
    for (const std::string& fen : testutil::test_fens()) {
        const Position p = testutil::parse_or_die(fen);
        std::vector<std::pair<Move, int>> scores = child_scores(backend, p, GameHistory{}, kDepth);
        if (scores.empty()) continue;
        ++fens;
        std::stable_sort(scores.begin(), scores.end(), [](const auto& x, const auto& y) { return x.second > y.second; });
        for (const int k : {1, 3}) {
            SearchOutcome outcome;
            const std::vector<RootLine> lines = last_lines(backend, p, GameHistory{}, kDepth, k, &outcome);
            if (k == 1) {  // the normal search, whose score is the best child's
                CHECK_MSG(lines.empty() && outcome.score == scores[0].second,
                          fen + ": " + std::to_string(outcome.score) + " vs " + std::to_string(scores[0].second));
                continue;
            }
            const size_t expected = std::min(size_t(k), scores.size());
            REQUIRE_MSG(lines.size() == expected, fen);
            for (size_t i = 0; i < lines.size(); ++i) {
                const auto own = std::find_if(scores.begin(), scores.end(),
                                              [&](const auto& s) { return s.first == lines[i].move; });
                REQUIRE(own != scores.end());
                CHECK_MSG(lines[i].score == scores[i].second && own->second == lines[i].score,
                          fen + " line " + std::to_string(i + 1) + ": " + move_to_uci(lines[i].move) + " " +
                              std::to_string(lines[i].score) + " vs " + std::to_string(scores[i].second));
                CHECK_MSG(!lines[i].pv.empty() && lines[i].pv[0] == lines[i].move && line_is_legal(p, lines[i].pv),
                          fen + " line " + std::to_string(i + 1));
                for (size_t j = 0; j < i; ++j) CHECK(lines[j].move != lines[i].move);
            }
            CHECK(outcome.best == lines[0].move && outcome.score == lines[0].score && outcome.pv == lines[0].pv);
        }
    }
    CHECK(fens > 15);
}

TEST_CASE(multipv_mates_draws_and_more_lines_than_moves) {
    CpuSearchOptions o{BoardKind::Quad, EvalKind::Terms, OrderingKind::Staged};
    o.features = FeatureKind::TTKillersSee;
    CpuBackend backend(o, 16);
    const Position mate = testutil::parse_or_die("6k1/5ppp/8/8/8/8/8/R5K1 w - - 0 1");
    std::vector<RootLine> lines = last_lines(backend, mate, GameHistory{}, kDepth, 2);
    REQUIRE(lines.size() == 2);
    CHECK(lines[0].move == parse(mate, "a1a8") && lines[0].score == kScoreMate - 1);
    CHECK(lines[1].move != lines[0].move && lines[1].score < lines[0].score);
    // The repeating move scores 0, its place among the others by that score.
    Position root;
    const GameHistory h = game("g1f3 g8f6 f3g1 f6g8 g1f3 g8f6 f3g1", root);
    lines = last_lines(backend, root, h, kDepth, kMaxMultiPv);
    const size_t legal = root_children(root, h).size();
    REQUIRE(lines.size() == legal);
    const auto back = std::find_if(lines.begin(), lines.end(), [&](const RootLine& l) { return l.move == parse(root, "f6g8"); });
    REQUIRE(back != lines.end());
    CHECK(back->score == kScoreDraw && back->pv.size() == 1);
    for (size_t i = 1; i < lines.size(); ++i) CHECK(lines[i - 1].score >= lines[i].score);
    // Stalemate: no lines, the normal search's answer.
    const Position stalemate = testutil::parse_or_die("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1");
    SearchOutcome outcome;
    lines = last_lines(backend, stalemate, GameHistory{}, kDepth, 3, &outcome);
    CHECK(lines.empty() && outcome.best.is_none() && outcome.score == kScoreDraw);
}

TEST_CASE(multipv_stop_right_after_go) {
    CpuBackend backend(CpuSearchOptions{BoardKind::Quad, EvalKind::Pesto, OrderingKind::Staged});
    const Position p = testutil::parse_or_die(testutil::test_fens()[1]);
    backend.request_stop();
    SearchOutcome outcome;
    const std::vector<RootLine> lines = last_lines(backend, p, GameHistory{}, 8, 3, &outcome);
    CHECK(outcome.depth == 1 && lines.size() == 3 && line_is_legal(p, {outcome.best}));
    backend.clear_stop();
}

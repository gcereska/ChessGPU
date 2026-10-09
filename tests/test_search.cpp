// Move ordering and the sequential search: pickers against the reference
// move generator, the two pickers against each other, the explicit-stack
// search against a recursive implementation, against minimax, and against
// an independent reference search; plus the draw and mate rules.

#include <algorithm>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "core/search.h"
#include "host/game_history.h"
#include "host/host_tables.h"
#include "host/search_cpu.h"
#include "host/uci_move.h"
#include "reference_search.h"
#include "test_harness.h"
#include "test_util.h"

using namespace cb;

namespace {

const HqAttacks& attacks() {
    static const HqAttacks a(host_attack_tables());
    return a;
}

const SearchTables& tables() { return host_search_tables(); }

std::string moves_text(const std::vector<Move>& moves) {
    std::string s;
    for (const Move m : moves) s += move_to_uci(m) + " ";
    return s;
}

// Moves the quiescence search tries when not in check, by the rules alone.
bool is_quiescence_move(const Position& p, Move m) {
    const bool capture = p.squares[m.to()] != 0 || m.flags() == EP_CAPTURE;
    const bool queen_promotion = m.is_promotion() && m.promotion_type() == QUEEN;
    return capture ? (!m.is_promotion() || queen_promotion) : queen_promotion;
}

template <class Board>
std::vector<Move> staged_sequence(const Board& b, PickMode mode, Move pv) {
    const CheckInfo ci = make_check_info(b, attacks());
    StagedPicker::State st;
    StagedPicker::init(st, b, attacks(), ci, mode);
    std::vector<Move> seq;
    for (Move m = StagedPicker::next(st, b, attacks(), ci, pv); !m.is_none();
         m = StagedPicker::next(st, b, attacks(), ci, pv)) {
        seq.push_back(m);
        if (seq.size() > 300) break;  // runaway guard
    }
    return seq;
}

template <class Board>
std::vector<Move> list_sequence(const Board& b, PickMode mode, Move pv) {
    static MoveArena<1024> arena;
    ListPicker::State st;
    ListPicker::init(st, arena, 100, b, attacks(), mode, pv);  // start mid-arena, like a child node
    std::vector<Move> seq;
    for (Move m = ListPicker::next(st, arena); !m.is_none(); m = ListPicker::next(st, arena)) seq.push_back(m);
    return seq;
}

// At every node of the test trees: both pickers produce exactly the legal
// moves (or the quiescence subset), each once, in strictly decreasing order
// key, with the PV move first when given, and identical sequences.
template <class Board>
void check_pickers() {
    long long nodes = 0;
    for (const std::string& fen : testutil::test_fens()) {
        long long budget = testutil::kTreeBudget / 2;
        auto visit = [&](const Position& p, const std::vector<Move>& ref_moves) {
            ++nodes;
            const Board b = to_board<Board>(p);
            const bool check = make_check_info(b, attacks()).in_check();
            std::vector<Move> pvs = {Move()};
            if (!ref_moves.empty()) {
                pvs.push_back(ref_moves.back());
                pvs.push_back(ref_moves[ref_moves.size() / 2]);
            }
            for (const Move pv : pvs) {
                for (const PickMode mode : {PickMode::All, PickMode::Quiescence}) {
                    if (mode == PickMode::Quiescence && (check || !pv.is_none())) continue;
                    const std::vector<Move> staged = staged_sequence(b, mode, pv);
                    const std::string where = std::string(Board::kName) + " " + format_fen(p) + " pv " +
                                              move_to_uci(pv) + (mode == PickMode::All ? " all" : " quiescence");
                    std::vector<Move> expected;
                    for (const Move m : ref_moves)
                        if (mode == PickMode::All || is_quiescence_move(p, m)) expected.push_back(m);
                    CHECK_MSG(testutil::sorted_raw(staged) == testutil::sorted_raw(expected),
                              where + "\n staged   " + moves_text(staged) + "\n expected " + moves_text(expected));
                    if (mode == PickMode::All && !pv.is_none() && !staged.empty())
                        CHECK_MSG(staged.front() == pv, where + ": PV move not first");
                    for (size_t i = 1; i < staged.size(); ++i) {
                        if (order_key(b, staged[i - 1], pv) <= order_key(b, staged[i], pv)) {
                            CHECK_MSG(false, where + ": out of order at " + move_to_uci(staged[i]) + "\n " +
                                                 moves_text(staged));
                            break;
                        }
                    }
                    const std::vector<Move> listed = list_sequence(b, mode, pv);
                    CHECK_MSG(listed == staged,
                              where + "\n list   " + moves_text(listed) + "\n staged " + moves_text(staged));
                }
            }
        };
        testutil::walk_tree(testutil::parse_or_die(fen), 3, budget, visit);
    }
    CHECK(nodes > 5000);
}

}  // namespace

TEST_CASE(picker_std_matches_reference_and_order) { check_pickers<StdBoard>(); }
TEST_CASE(picker_quad_matches_reference_and_order) { check_pickers<QuadBoard>(); }

TEST_CASE(picker_order_examples) {
    // Captures by victim value, then by attacker value: the pawn on e4 and
    // the knight on c3 can both take the queen on d5; the pawn goes first.
    const Position p = testutil::parse_or_die("r3k3/8/8/3q4/4P3/2N5/8/4K3 w - - 0 1");
    const QuadBoard b = to_board<QuadBoard>(p);
    const std::vector<Move> seq = staged_sequence(b, PickMode::All, Move());
    REQUIRE(seq.size() >= 3);
    CHECK_EQ(move_to_uci(seq[0]), std::string("e4d5"));
    CHECK_EQ(move_to_uci(seq[1]), std::string("c3d5"));
    // Quiet moves come later: the next move is a quiet knight move.
    CHECK(!seq[2].is_capture());
    CHECK_EQ(int(b.piece_type_on(seq[2].from())), int(KNIGHT));
    // Quiescence: only the two captures.
    CHECK_EQ(staged_sequence(b, PickMode::Quiescence, Move()).size(), size_t(2));
}

namespace {

// The engine's search, written recursively with plain move lists: same
// rules and same move order, so node counts must match search_task exactly.
template <class Board, class Eval>
struct RecursiveSearch {
    SearchBoard<Board, Eval> sb;
    std::vector<u64> keys;
    std::vector<Move> pv_line;
    u64 nodes = 0;

    int search(int depth, int alpha, int beta, int ply, bool on_pv) {
        ++nodes;
        keys.resize(size_t(ply + 1));
        keys[size_t(ply)] = sb.key;
        const int halfmove = state::halfmove(sb.board.state());
        if (ply > 0) {
            for (int d = 4; d <= halfmove && d <= ply; d += 2)
                if (keys[size_t(ply - d)] == sb.key && ply - d > 0) return kScoreDraw;
            if (insufficient_material(sb.board)) return kScoreDraw;
        }
        const CheckInfo ci = make_check_info(sb.board, attacks());
        if (ply > 0 && halfmove >= 100)
            return (ci.in_check() && count_legal(sb.board, attacks()) == 0) ? mated_in(ply) : kScoreDraw;
        if (ply >= kMaxPly) return evaluate_position<Eval>(sb.board, sb.acc, ci.us);

        const bool quiescence = depth <= 0;
        int best = -kScoreInfinite;
        if (quiescence && !ci.in_check()) {
            const int stand = evaluate_position<Eval>(sb.board, sb.acc, ci.us);
            if (stand >= beta) return stand;
            best = stand;
            alpha = std::max(alpha, stand);
        }
        const Move pv = (!quiescence && on_pv && ply < int(pv_line.size())) ? pv_line[size_t(ply)] : Move();
        MoveList list;
        generate_legal(sb.board, attacks(), list);
        std::vector<std::pair<u32, Move>> ordered;
        for (int i = 0; i < list.size; ++i) {
            const u32 key = order_key(sb.board, list.moves[i], pv);
            if (quiescence && !ci.in_check() && !is_quiescence_group(key)) continue;
            ordered.emplace_back(key, list.moves[i]);
        }
        std::sort(ordered.begin(), ordered.end(), [](const auto& x, const auto& y) { return x.first > y.first; });
        if (ordered.empty()) return ci.in_check() ? mated_in(ply) : (quiescence ? best : kScoreDraw);
        for (const auto& [key, m] : ordered) {
            const u64 key_before = sb.key;
            const Undo undo = make_search_move(sb, m, attacks(), tables());
            const int v = -search(depth - 1, -beta, -alpha, ply + 1, on_pv && m == pv);
            unmake_search_move(sb, m, undo, key_before, tables());
            if (v > best) {
                best = v;
                if (v > alpha) {
                    alpha = v;
                    if (v >= beta) return best;
                }
            }
        }
        return best;
    }
};

#ifdef NDEBUG
constexpr int kMaxTestDepth = 4;
#else
constexpr int kMaxTestDepth = 2;
#endif

template <class Board, class Eval>
void check_against_recursive() {
    for (const std::string& fen : testutil::test_fens()) {
        const Position p = testutil::parse_or_die(fen);
        std::vector<Move> pv;
        for (int depth = 1; depth <= kMaxTestDepth; ++depth) {
            const CpuSearchOptions opt{BoardKind::Quad, std::is_same_v<Eval, PestoEval> ? EvalKind::Pesto : EvalKind::Material,
                                       OrderingKind::Staged};
            CpuSearchOptions o = opt;
            o.board = std::is_same_v<Board, StdBoard> ? BoardKind::Std : BoardKind::Quad;
            const FixedDepthResult fast = cpu_search_fixed(p, GameHistory{}, depth, -kScoreInfinite, kScoreInfinite, o, pv);
            RecursiveSearch<Board, Eval> rec{make_search_board<Eval>(to_board<Board>(p), attacks(), tables())};
            rec.pv_line = pv;
            const int score = rec.search(depth, -kScoreInfinite, kScoreInfinite, 0, true);
            const std::string where = fen + " depth " + std::to_string(depth);
            CHECK_MSG(fast.search.score == score,
                      where + ": " + std::to_string(fast.search.score) + " vs " + std::to_string(score));
            CHECK_MSG(fast.search.nodes == rec.nodes,
                      where + ": nodes " + std::to_string(fast.search.nodes) + " vs " + std::to_string(rec.nodes));
            pv = fast.pv;
        }
    }
}

}  // namespace

TEST_CASE(search_matches_recursive_quad_pesto) { check_against_recursive<QuadBoard, PestoEval>(); }
TEST_CASE(search_matches_recursive_std_material) { check_against_recursive<StdBoard, MaterialEval>(); }

TEST_CASE(search_variants_visit_identical_trees) {
    // Staged and list ordering, std and quad boards: the same tree, so the
    // same score, best move, PV and node count.
    for (const std::string& fen : testutil::test_fens()) {
        const Position p = testutil::parse_or_die(fen);
        for (const EvalKind eval : {EvalKind::Material, EvalKind::Pesto}) {
            std::vector<Move> pv;
            for (int depth = 1; depth <= kMaxTestDepth + 1; ++depth) {
                const FixedDepthResult base = cpu_search_fixed(p, GameHistory{}, depth, -kScoreInfinite, kScoreInfinite,
                                                               {BoardKind::Quad, eval, OrderingKind::Staged}, pv);
                for (const BoardKind board : {BoardKind::Std, BoardKind::Quad}) {
                    for (const OrderingKind ordering : {OrderingKind::Staged, OrderingKind::List}) {
                        const FixedDepthResult r = cpu_search_fixed(p, GameHistory{}, depth, -kScoreInfinite,
                                                                    kScoreInfinite, {board, eval, ordering}, pv);
                        const std::string where = fen + " depth " + std::to_string(depth) + " " + name_of(board) +
                                                  "/" + name_of(ordering) + "/" + name_of(eval);
                        CHECK_MSG(r.search.score == base.search.score, where);
                        CHECK_MSG(r.search.best == base.search.best, where);
                        CHECK_MSG(r.search.nodes == base.search.nodes, where);
                        CHECK_MSG(r.pv == base.pv, where);
                    }
                }
                pv = base.pv;
            }
        }
    }
}

namespace {

// Minimax over the first plies with no pruning, where every leaf is searched
// as its own task (window: everything) that links to its ancestors through
// the history chain, the way a parallel coordinator hands out subtrees. The
// result must equal one alpha-beta search of the whole tree.
template <class Board, class Eval>
struct TaskMinimax {
    std::vector<HistoryEntry> chain;
    LocalStack<SearchLevel<StagedPicker::State>, kMaxPly + 1> stack;
    u64 tasks = 0;

    int search(const Board& b, int depth, int ply, int parent_entry) {
        if (depth == 0) {
            SearchTask<Board> task;
            task.board = b;
            task.alpha = short(-kScoreInfinite);
            task.beta = short(kScoreInfinite);
            task.depth = 0;
            task.ply = u8(ply);
            task.flags = 0;
            task.history = parent_entry;
            const SearchEnv<HqAttacks> env{attacks(), &tables(), chain.empty() ? nullptr : chain.data(), nullptr, 0};
            StagedOrdering::Workspace ws;
            NoStop control;
            ++tasks;
            return search_task<Board, Eval, StagedOrdering, decltype(stack), NoStop, false>(task, env, stack, ws,
                                                                                           control, nullptr)
                .score;
        }
        if (ply > 0) {
            if (insufficient_material(b)) return kScoreDraw;
            if (state::halfmove(b.state()) >= 100) {
                const bool check = make_check_info(b, attacks()).in_check();
                return (check && count_legal(b, attacks()) == 0) ? mated_in(ply) : kScoreDraw;
            }
        }
        MoveList list;
        generate_legal(b, attacks(), list);
        if (list.size == 0) return make_check_info(b, attacks()).in_check() ? mated_in(ply) : kScoreDraw;
        HistoryEntry e;
        e.key = compute_key(b, attacks(), tables().zobrist);
        e.prev = parent_entry;
        e.ply = short(ply);
        e.repeated = 0;
        chain.push_back(e);
        const int entry = int(chain.size()) - 1;
        int best = -kScoreInfinite;
        for (int i = 0; i < list.size; ++i) {
            Board child = b;
            make_move(child, list.moves[i]);
            best = std::max(best, -search(child, depth - 1, ply + 1, entry));
        }
        chain.pop_back();
        return best;
    }
};

// Depth 3 (leaves at ply 3, linking through 3 explicit ancestors) only for
// positions with few moves; minimax visits every leaf.
#ifdef NDEBUG
constexpr int kMinimaxDepth = 3;
constexpr int kMinimaxDeepMaxMoves = 20;
#else
constexpr int kMinimaxDepth = 2;
constexpr int kMinimaxDeepMaxMoves = 0;
#endif

template <class Board, class Eval>
void check_against_task_minimax() {
    const EvalKind eval = std::is_same_v<Eval, PestoEval> ? EvalKind::Pesto : EvalKind::Material;
    const BoardKind board = std::is_same_v<Board, StdBoard> ? BoardKind::Std : BoardKind::Quad;
    int deep = 0;
    for (const std::string& fen : testutil::test_fens()) {
        const Position p = testutil::parse_or_die(fen);
        const bool few_moves = int(count_legal(to_board<Board>(p), attacks())) <= kMinimaxDeepMaxMoves;
        deep += few_moves;
        for (int depth = 1; depth <= (few_moves ? kMinimaxDepth : 2); ++depth) {
            auto mm = std::make_unique<TaskMinimax<Board, Eval>>();
            const int expected = mm->search(to_board<Board>(p), depth, 0, -1);
            const FixedDepthResult r = cpu_search_fixed(p, GameHistory{}, depth, -kScoreInfinite, kScoreInfinite,
                                                        {board, eval, OrderingKind::Staged});
            CHECK_MSG(r.search.score == expected, fen + " depth " + std::to_string(depth) + ": " +
                                                      std::to_string(r.search.score) + " vs " +
                                                      std::to_string(expected));
        }
    }
#ifdef NDEBUG
    CHECK(deep >= 8);
#endif
}

}  // namespace

TEST_CASE(search_equals_minimax_with_task_leaves_pesto) { check_against_task_minimax<QuadBoard, PestoEval>(); }
TEST_CASE(search_equals_minimax_with_task_leaves_material) { check_against_task_minimax<StdBoard, MaterialEval>(); }

namespace {

// The test positions to depth 3, more positions from their trees to depth 2.
#ifdef NDEBUG
constexpr int kReferenceDepth = 3;
constexpr long long kReferenceTreeBudget = 8;
#else
constexpr int kReferenceDepth = 2;
constexpr long long kReferenceTreeBudget = 2;
#endif

}  // namespace

TEST_CASE(search_matches_independent_reference) {
    // Root positions: the test positions plus some positions from their trees.
    std::vector<Position> roots;
    std::vector<int> max_depth;
    for (const std::string& fen : testutil::test_fens()) {
        long long budget = kReferenceTreeBudget;
        auto visit = [&](const Position& p, const std::vector<Move>&) {
            max_depth.push_back(roots.empty() || budget == kReferenceTreeBudget - 1 ? kReferenceDepth : 2);
            roots.push_back(p);
        };
        testutil::walk_tree(testutil::parse_or_die(fen), 2, budget, visit);
    }
    for (size_t i = 0; i < roots.size(); ++i) {
        const Position& p = roots[i];
        const ref::Board r = ref::from_position(p);
        for (int depth = 1; depth <= max_depth[i]; ++depth) {
            for (const EvalKind eval : {EvalKind::Material, EvalKind::Pesto}) {
                const int expected = refsearch::search_root(
                    r, {}, depth, eval == EvalKind::Pesto ? refsearch::Eval::Pesto : refsearch::Eval::Material);
                const FixedDepthResult got = cpu_search_fixed(p, GameHistory{}, depth, -kScoreInfinite, kScoreInfinite,
                                                              {BoardKind::Quad, eval, OrderingKind::Staged});
                CHECK_MSG(got.search.score == expected, format_fen(p) + " depth " + std::to_string(depth) + " " +
                                                            name_of(eval) + ": " + std::to_string(got.search.score) +
                                                            " vs reference " + std::to_string(expected));
            }
        }
    }
    CHECK(roots.size() > testutil::test_fens().size());  // positions from the trees too, not just the roots
}

// ---------------------------------------------------------------------------
// Rules
// ---------------------------------------------------------------------------

namespace {

FixedDepthResult search_fen(const std::string& fen, int depth, EvalKind eval = EvalKind::Pesto) {
    return cpu_search_fixed(testutil::parse_or_die(fen), GameHistory{}, depth, -kScoreInfinite, kScoreInfinite,
                            {BoardKind::Quad, eval, OrderingKind::Staged});
}

}  // namespace

TEST_CASE(search_terminal_roots) {
    // Stalemate and checkmate at the root: no move, draw or mated-in-0.
    const FixedDepthResult stalemate = search_fen("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1", 3);
    CHECK_EQ(stalemate.search.score, kScoreDraw);
    CHECK(stalemate.search.best.is_none());
    const FixedDepthResult mated = search_fen("R5k1/5ppp/8/8/8/8/8/6K1 b - - 0 1", 3);
    CHECK_EQ(mated.search.score, mated_in(0));
    CHECK(mated.search.best.is_none());
}

TEST_CASE(search_finds_mate_in_one) {
    for (int depth = 1; depth <= 4; ++depth) {
        const FixedDepthResult r = search_fen("6k1/5ppp/8/8/8/8/8/R5K1 w - - 0 1", depth);
        CHECK_EQ(r.search.score, mate_in(1));
        CHECK_EQ(move_to_uci(r.search.best), std::string("a1a8"));
    }
    // Mated in one: black's only move Kb8 allows Rh8#.
    for (int depth = 2; depth <= 4; ++depth) {
        const FixedDepthResult r = search_fen("k7/8/1K6/8/8/8/8/7R b - - 0 1", depth);
        CHECK_EQ(r.search.score, mated_in(2));
        CHECK_EQ(move_to_uci(r.search.best), std::string("a8b8"));
    }
    // At depth 1 the mate is past the horizon (quiescence doesn't look at checks).
    CHECK(!is_mate_score(search_fen("k7/8/1K6/8/8/8/8/7R b - - 0 1", 1).search.score));
}

TEST_CASE(search_insufficient_material_is_draw) {
    CHECK_EQ(search_fen("8/8/8/4k3/8/8/8/4KN2 w - - 0 1", 3).search.score, kScoreDraw);
    CHECK_EQ(search_fen("8/8/8/4k3/8/8/8/4KB2 b - - 0 1", 3).search.score, kScoreDraw);
}

TEST_CASE(search_fifty_move_rule) {
    // With the clock at 99, every move but a capture or pawn move ends the
    // game in a draw, so the extra rook is worth nothing...
    CHECK_EQ(search_fen("4k3/8/8/8/8/8/8/R3K3 w Q - 99 80", 1).search.score, kScoreDraw);
    CHECK_EQ(search_fen("4k3/8/8/8/8/8/8/R3K3 w Q - 99 80", 3).search.score, kScoreDraw);
    CHECK(search_fen("4k3/8/8/8/8/8/8/R3K3 w Q - 0 80", 3).search.score > 300);
    // ...unless the move checkmates: mate takes precedence.
    const FixedDepthResult mate = search_fen("k7/8/1K6/8/8/8/8/7R w - - 99 80", 3);
    CHECK_EQ(mate.search.score, mate_in(1));
    CHECK_EQ(move_to_uci(mate.search.best), std::string("h1h8"));
}

TEST_CASE(search_repetition_rule_unit) {
    // A stack of keys and a history chain set up by hand.
    struct KeyStack {
        std::vector<u64> keys;
        u64 key(int i) const { return keys[size_t(i)]; }
    };
    SearchTask<QuadBoard> task;
    task.flags = 0;

    // Inside the task: a position equal to the one 4 plies up is a draw
    // when that earlier position is after the search root.
    task.ply = 3;
    task.history = -1;
    KeyStack s{{11, 12, 13, 14, 11}};
    CHECK(detail::is_repetition(s, 4, 11, 10, task, nullptr));
    CHECK(!detail::is_repetition(s, 4, 11, 3, task, nullptr));  // beyond the halfmove clock
    CHECK(!detail::is_repetition(s, 4, 99, 10, task, nullptr));
    // The match is the search root (task ply 0): only a draw if the root
    // itself repeated a game position.
    task.ply = 0;
    CHECK(!detail::is_repetition(s, 4, 11, 10, task, nullptr));
    task.flags = kTaskRootRepeated;
    CHECK(detail::is_repetition(s, 4, 11, 10, task, nullptr));
    task.flags = 0;

    // Through the history chain: game positions (ply < 0) and explicit
    // nodes (ply > 0) above a task at ply 2.
    std::vector<HistoryEntry> h(5);
    // entries: 0: ply -2 (key 50, repeated), 1: ply -1 (51), 2: ply 0 root (52), 3: ply 1 (53), then the
    // task root at ply 2 (stack level 0).
    h[0] = {50, -1, -2, 1};
    h[1] = {51, 0, -1, 0};
    h[2] = {52, 1, 0, 0};
    h[3] = {53, 2, 1, 0};
    h[4] = {60, 3, 2, 0};  // unused entry
    task.ply = 2;
    task.history = 3;
    KeyStack t{{54, 55, 56}};  // levels 0..2 = plies 2..4
    // Current position at level 2 (ply 4): 4 plies back is ply 0 (entry 2, key 52), 6 back is ply -2 (entry 0).
    CHECK(!detail::is_repetition(t, 2, 52, 10, task, h.data()));  // root not repeated: no draw
    h[2].repeated = 1;
    CHECK(detail::is_repetition(t, 2, 52, 10, task, h.data()));   // root repeated: third occurrence
    CHECK(detail::is_repetition(t, 2, 50, 10, task, h.data()));   // ply -2 position had repeated: draw
    CHECK(!detail::is_repetition(t, 2, 50, 5, task, h.data()));   // 6 plies back is beyond the clock
    CHECK(!detail::is_repetition(t, 2, 51, 10, task, h.data()));  // odd distance: never compared
    // A key 4 plies up inside the chain at ply > 0: level 3 (ply 5), 4 back = ply 1 (entry 3).
    KeyStack u{{54, 55, 56, 57}};
    CHECK(detail::is_repetition(u, 3, 53, 10, task, h.data()));
}

TEST_CASE(search_repetition_against_reference_with_game_history) {
    // Games with repeated positions before the root: the engine and the
    // reference must agree, and the history must matter in at least one case.
    struct Game {
        const char* fen;
        const char* moves;
    };
    const Game games[] = {
        {kStartFen, "g1f3 g8f6 f3g1 f6g8"},
        {kStartFen, "g1f3 g8f6 f3g1 f6g8 g1f3 g8f6 f3g1"},
        {"4k3/8/8/8/8/8/4q3/K7 w - - 0 1", "a1b1 e2e1 b1b2 e1e2 b2b1 e2e1 b1b2"},
        {"7k/8/8/8/8/1q6/8/K2Q4 w - - 0 1", "d1h5 h8g7 h5g5 g7h7 g5h5 h7g7 h5g5 g7h7"},
        // Black, far behind, can repeat a position a third time: a draw only with the history.
        {"6k1/5ppp/8/8/8/8/5PPP/3QR1K1 w - - 0 1", "g1h1 g8h8 h1g1 h8g8 g1h1 g8h8 h1g1"},
    };
    int differ = 0;
    for (const Game& g : games) {
        const Position start = testutil::parse_or_die(g.fen);
        Position root;
        GameHistory history;
        std::string error;
        if (!build_game(start, g.moves, root, history, error)) {
            CHECK_MSG(false, std::string(g.moves) + ": " + error);
            continue;
        }
        // The reference gets the game positions as boards.
        std::vector<ref::Board> game;
        {
            ref::Board b = ref::from_position(start);
            std::string text = g.moves;
            size_t i = 0;
            while (i < text.size()) {
                const size_t end = std::min(text.find(' ', i), text.size());
                const std::string mv = text.substr(i, end - i);
                i = end + 1;
                game.push_back(b);
                const QuadBoard qb = to_board<QuadBoard>(ref::to_position(b));
                b = ref::play(b, parse_uci_move(qb, attacks(), mv));
            }
        }
        const ref::Board r = ref::from_position(root);
        for (int depth = 1; depth <= kReferenceDepth + 1; ++depth) {
            const int expected = refsearch::search_root(r, game, depth, refsearch::Eval::Pesto);
            const FixedDepthResult got = cpu_search_fixed(root, history, depth, -kScoreInfinite, kScoreInfinite,
                                                          {BoardKind::Quad, EvalKind::Pesto, OrderingKind::Staged});
            CHECK_MSG(got.search.score == expected, std::string(g.moves) + " depth " + std::to_string(depth) + ": " +
                                                        std::to_string(got.search.score) + " vs reference " +
                                                        std::to_string(expected));
            const FixedDepthResult no_history = cpu_search_fixed(root, GameHistory{}, depth, -kScoreInfinite,
                                                                 kScoreInfinite, {BoardKind::Quad, EvalKind::Pesto,
                                                                                  OrderingKind::Staged});
            if (no_history.search.score != got.search.score) ++differ;
        }
    }
    CHECK(differ > 0);
}

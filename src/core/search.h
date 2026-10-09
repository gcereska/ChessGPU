#pragma once

// Sequential alpha-beta search of one task (a position, a depth and a window),
// as an explicit-stack loop: the same code runs on the CPU and on one GPU
// thread. Negamax form, fail-soft, with quiescence search at the leaves.
//
// Template parameters choose the variants:
//   Board     StdBoard or QuadBoard
//   Eval      MaterialEval, PestoEval or PestoTermsEval
//   Ordering  StagedOrdering or ListOrdering<N> (both give the same move order)
//   Storage   where the per-ply stack lives (local memory, shared memory, ...)
//   Control   when to stop early (time limit, stop command, sibling cutoff)
//   kTrackPv  keep the principal variation (CPU searches; GPU workers don't)
//   Features  pruning techniques (core/features.h)
//   Layout    transposition table entry layout (core/tt.h), if Features has one
//
// Rules: checkmate scores mated_in(ply); stalemate, the 50-move rule,
// repetitions and insufficient material score a draw. A repetition counts
// when the earlier occurrence is inside the search (after the root), or when
// it is the third occurrence counting the game before the root.
//
// Null-move pruning (feature sets with kNullMove): before its moves, a node
// lets the opponent move twice in a row (a "pass", see make_null_move) and
// searches that at reduced depth with the window (beta - 1, beta). A result
// of at least beta ends the node: if even a free move doesn't help the
// opponent, a real move won't either (zugzwang, where every move hurts,
// breaks this; see null_move_allowed for when it's tried). The node then
// scores the null result, as a lower bound, but never a mate: a mate found
// after passing proves nothing.
//
// Late move reductions (feature sets with kLmr): a quiet move late in a
// node's order (LmrTable decides how late, by depth and move number) that
// doesn't give check, at a node not in check, is first searched R plies
// shallower with the zero window (alpha, alpha + 1). If that beats alpha,
// the same move is searched again at full depth with the full window; the
// reduced result counts only when it doesn't.
//
// Principal variation search (feature sets with kPvs): at a main-search
// node with a real window (beta - alpha > 1), every move after the first is
// first searched with the zero window (alpha, alpha + 1); if that beats
// alpha, the move is searched again with the full window. (A zero-window
// result at or above beta is searched again too, like a reduced one: rare at
// such nodes, and it keeps one flag for both.) Sets without the flag (and the
// tests' hand-written sets, which don't spell it out) don't use it.
//
// Checks in quiescence (feature sets with kQChecks): at quiescence's first
// ply (depth 0) when not in check, quiet moves that give check (not
// castling) come after the captures and queen promotions, in their usual
// order, exempt from quiescence pruning. Their replies are evasions at depth
// -1, whose own captures get no more checks. A safety net for null moves and
// reductions: threats by check just past the horizon.
//
// Futility pruning (feature sets with kFutility): near the leaves (see
// futility_bound), a quiet move that doesn't give check is skipped when the
// static evaluation plus a margin can't reach alpha. The node scores at
// least that bound for it (fail-soft), and a node whose moves were all
// skipped is no stalemate: it scores the best bound.
//
// Check extensions (feature sets with kCheckExt): in the main search (1 or
// more plies left), a move that gives check is searched a ply deeper than
// other moves: its child keeps the node's depth.
//
// Killer moves (feature sets with kKillers): each ply remembers the last two
// quiet moves (not castling) that cut off there, and the move order tries
// them after the captures (see order_key). A task starts with none.
//
// Quiescence pruning (feature sets with kQPrune, core/see.h): at a
// quiescence node not in check, a capture or queen promotion is skipped when
// the stand pat plus its material gain and a margin can't reach alpha (delta
// pruning; the node then scores at least that bound), or when it loses
// material in the exchange on its square (SEE pruning).
//
// Evaluations reach the board through evaluate_position (core/eval.h), so an
// eval may count terms from the board (PestoTermsEval) as well as use its
// incremental accumulator.
//
// ABDADA (core/abdada.h): with a Control that has `kAbdada` (busy, claim and
// release over a shared table of positions being searched) and an Ordering
// that can defer moves (the list orderings), a node with at least
// kAbdadaMinDepth plies left defers every move but its first whose position
// another worker is searching, and comes back to it after the others.

#include "core/abdada.h"
#include "core/eval_terms.h"
#include "core/features.h"
#include "core/move_order.h"
#include "core/see.h"
#include "core/score.h"
#include "core/search_board.h"
#include "core/tt.h"

namespace cb {

// A position before the current task's root: the game's positions before
// the search root, and the coordinator's explicit nodes above the task. Each
// entry links to the position one ply earlier.
struct HistoryEntry {
    u64 key;
    int prev;       // entry of the position one ply earlier, or -1
    short ply;      // plies from the search root (0 = the root, negative = earlier game positions)
    u8 repeated;    // for ply <= 0: the position also occurred earlier in the game
    u8 unused = 0;
};

// Task flags.
constexpr u8 kTaskOnPv = 1;          // the task root lies on the previous iteration's PV
constexpr u8 kTaskRootRepeated = 2;  // the task root is the search root, and it repeats a game position
constexpr u8 kTaskAfterNull = 4;     // the move into the task root was a null move

template <class Board>
struct SearchTask {
    Board board;
    short alpha;
    short beta;
    signed char depth;  // remaining main-search depth (<= 0: quiescence only)
    u8 ply;             // plies from the search root
    u8 flags;
    u8 unused = 0;
    int history;           // entry of the task root's parent in the history, or -1
    u32 node_budget = 0;   // stop when the search reaches this many nodes and goes on; 0: none
};

struct SearchResult {
    int score = 0;
    Move best;      // empty when the root has no legal moves or the search stopped early
    u64 nodes = 0;  // every position entered, quiescence included
    bool aborted = false;      // stopped early (aborted alone: stop request or deadline)
    bool over_budget = false;  // stopped early because the node budget ran out (aborted is set too)
};

// Read-only inputs shared by every task of a search.
template <class Attacks>
struct SearchEnv {
    Attacks attacks;
    const SearchTables* tables;
    const HistoryEntry* history;  // may be null if no task links into it
    const Move* pv;               // previous iteration's PV indexed by ply from the search root, or null
    int pv_length;
    TTDesc tt{};                  // the transposition table (feature sets with one)
    PruneParams prune{};          // settings of the pruning techniques
    const LmrTable* lmr = nullptr;  // late move reductions (feature sets with them; built from prune)
};

// Per-ply search state. The current ply's entry lives in registers; the
// stack holds the entries of the plies above it.
template <class PickerState>
struct SearchLevel {
    PickerState picker;
    u64 key;        // Zobrist key of this ply's position
    u64 pinned;     // CheckInfo of this position, kept so it isn't recomputed after every child
    u64 checkmask;
    Undo undo;      // of `move`
    Move move;      // the move being searched from here
    Move best_move;
    Move pv_move;   // previous iteration's best move here, if this position is on its PV
    Move tt_move;   // the transposition table's best move here, if any
    short alpha;
    short beta;
    short best;
    short alpha_orig;  // alpha on entry, before a stand pat raised it (for the table's bound)
    signed char depth;
    u8 flags;
    u8 moves_searched;  // real moves searched so far (the null move and re-searches don't count)
};

namespace level_flags {
constexpr u8 kInCheck = 1;
constexpr u8 kQuiescence = 2;
constexpr u8 kSearchedAny = 4;  // at least one move was searched
constexpr u8 kOnPv = 8;
constexpr u8 kStore = 16;  // the node is searched (not cut off or scored at once): store its result
// The move into this node was a null move: no second one in a row. With
// today's evaluations (exactly antisymmetric) a null child's static
// evaluation is minus its parent's, so it's below the child's beta and the
// evaluation test already forbids passing again; the flag keeps the rule
// for evaluations with a side-to-move term.
constexpr u8 kAfterNull = 32;
constexpr u8 kNullPending = 64;  // the node tries a null move before its first move
constexpr u8 kReduced = 128;     // the move being searched got a reduced, zero-window search
}  // namespace level_flags

// The score of a node whose null move reached beta (null search result v,
// fail-soft): v, but never a mate, since a mate found after passing proves
// nothing. Shared with the GPU-resident YBWC's explicit nodes.
CB_HD int null_move_score(int v, int beta) { return is_mate_score(v) ? beta : v; }

// The futility bound of a quiet move at a node (static evaluation `eval`,
// `depth` plies left, flags, window alpha): the most such a move could
// score, or kScoreInfinite where futility pruning doesn't apply (no plies
// left or too many, in check, on the previous PV, alpha a mate score). The
// caller skips the move if the bound is at most alpha and the move doesn't
// give check. Shared with the GPU-resident YBWC's explicit nodes.
CB_HD int futility_bound(const PruneParams& prune, int eval, int depth, u8 flags, int alpha) {
    if (depth < 1 || depth > prune.fut_depth || (flags & (level_flags::kInCheck | level_flags::kOnPv)) != 0 ||
        is_mate_score(alpha))
        return kScoreInfinite;
    return eval + prune.fut_margin * depth;
}

// May a node try a null move? Not at the root (it must return a move), on
// the previous iteration's PV, in check (passing would leave the king in
// check), right after a null move, when beta is a mate score, near the
// leaves, with only pawns besides the king (zugzwang is common there), or
// when the static evaluation is below beta (the pass would fail anyway).
template <class Eval, class Board, class Attacks = TableFreeAttacks>
CB_HD bool null_move_allowed(const PruneParams& prune, int ply, int depth, u8 flags, bool in_check, int beta,
                             const Board& b, Color us, const typename Eval::Accumulator& acc,
                             const Attacks& a = Attacks{}) {
    return ply > 0 && depth >= prune.null_min_depth && !in_check &&
           (flags & (level_flags::kOnPv | level_flags::kAfterNull)) == 0 && !is_mate_score(beta) &&
           has_non_pawn_material(b, us) && evaluate_position<Eval>(b, acc, us, a) >= beta;
}

// The stack in local memory (or the host stack). kLevels must be at least
// kMaxPly + 1 - (task root ply).
template <class Level, int kLevels>
struct LocalStack {
    static constexpr int capacity = kLevels;
    Level levels[kLevels];

    CB_HD void store(int i, const Level& l) { levels[i] = l; }
    CB_HD Level load(int i) const { return levels[i]; }
    CB_HD u64 key(int i) const { return levels[i].key; }
};

// Ordering policies: pick moves with StagedPicker or ListPicker.
struct StagedOrdering {
    static constexpr const char* kName = "staged";
    using State = StagedPicker::State;
    struct Workspace {};

    template <class Board, class Attacks>
    CB_HD static void init(State& st, Workspace&, int, const Board& b, const Attacks& a, const CheckInfo& ci,
                           PickMode mode, Move, Killers killers) {
        StagedPicker::init(st, b, a, ci, mode, killers);
    }
    CB_HD static int top(const State&) { return 0; }
    CB_HD static bool overflowed(const State&) { return false; }
    template <class Board, class Attacks>
    CB_HD static Move next(State& st, Workspace&, const Board& b, const Attacks& a, const CheckInfo& ci, Move pv) {
        return StagedPicker::next(st, b, a, ci, pv);
    }
};

template <int kArenaMoves>
struct ListOrdering {
    static constexpr const char* kName = "list";
    using State = ListPicker::State;
    using Workspace = MoveArena<kArenaMoves>;

    template <class Board, class Attacks>
    CB_HD static void init(State& st, Workspace& ws, int top, const Board& b, const Attacks& a, const CheckInfo&,
                           PickMode mode, Move pv, Killers killers) {
        ListPicker::init(st, ws, top, b, a, mode, pv, killers);
    }
    CB_HD static int top(const State& st) { return st.end; }
    CB_HD static bool overflowed(const State& st) { return st.overflow != 0; }
    // ABDADA: put the move next() returned back for later; was it one put back?
    CB_HD static void defer(State& st, Workspace& ws) { ListPicker::defer(st, ws); }
    CB_HD static bool returned_deferred(const State& st, const Workspace& ws) {
        return ListPicker::returned_deferred(st, ws);
    }
    template <class Board, class Attacks>
    CB_HD static Move next(State& st, Workspace& ws, const Board&, const Attacks&, const CheckInfo&, Move) {
        return ListPicker::next(st, ws);
    }
};

// Whether Control coordinates workers ABDADA-style (core/abdada.h), and
// whether an Ordering can put moves back for later (ABDADA needs it).
template <class Control>
constexpr bool abdada_v = requires { requires Control::kAbdada; };

// Whether feature set Features uses principal variation search (sets that
// don't name the flag don't).
template <class Features>
constexpr bool pvs_v = requires { requires Features::kPvs; };

// Whether feature set Features tries quiet checks in quiescence (sets that
// don't name the flag don't).
template <class Features>
constexpr bool qchecks_v = requires { requires Features::kQChecks; };
template <class Ordering>
constexpr bool can_defer_v = requires { &Ordering::defer; };

// Never stops.
struct NoStop {
    CB_HD bool should_stop(u64) { return false; }
};

// Table access through the Control: a warp model control (one search run by
// all 32 lanes in step) has lane 0 probe and broadcast, so every lane takes
// the same path even while other workers write the entry, and only lane 0
// stores. Other controls use the table directly.
template <class Control, class View>
CB_HD TTData tt_probe(Control& control, const View& tt, u64 key) {
    if constexpr (requires { control.warp_tt_probe(tt, key); }) {
        return control.warp_tt_probe(tt, key);
    } else {
        return tt.probe(key);
    }
}

template <class Control, class View>
CB_HD void tt_store(Control& control, const View& tt, u64 key, TTData d) {
    if constexpr (requires { control.warp_tt_store(tt, key, d); }) {
        control.warp_tt_store(tt, key, d);
    } else {
        tt.store(key, d);
    }
}

// Principal variation of every ply on the current path (CPU only).
struct PvTable {
    Move moves[kMaxPly + 1][kMaxPly + 1];
    int length[kMaxPly + 1];
};

// Nodes between calls to Control::should_stop (a power of two). The first
// node of every task polls too, so a stop that came before a batch ends even
// its small tasks at once.
constexpr u64 kStopPollInterval = 1024;

namespace detail {

// Is the position at stack level `level` (key `key`, halfmove clock
// `halfmove`) a draw by repetition? Positions at levels below `level` come
// from the stack; earlier ones from the history chain.
template <class Storage, class Board>
CB_HD bool is_repetition(const Storage& stack, int level, u64 key, int halfmove, const SearchTask<Board>& task,
                         const HistoryEntry* history) {
    // A position can first repeat 4 plies later (each side moves out and back).
    int d = 4;
    for (; d <= halfmove && level - d >= 0; d += 2) {
        if (stack.key(level - d) == key) {
            const int ply = task.ply + level - d;
            return ply > 0 || (task.flags & kTaskRootRepeated) != 0;
        }
    }
    if (d > halfmove || history == nullptr) return false;
    // Continue into the history: entry `e` is the position `steps` plies
    // before the task root's parent... walk to distance d.
    int e = task.history;
    int distance = level + 1;  // distance of entry e from the current position
    while (e >= 0 && d <= halfmove) {
        while (e >= 0 && distance < d) {
            e = history[e].prev;
            ++distance;
        }
        if (e < 0) break;
        const HistoryEntry& h = history[e];
        if (h.key == key) return h.ply > 0 || h.repeated != 0;
        d += 2;
    }
    return false;
}

}  // namespace detail

template <class Board, class Eval, class Ordering, class Storage, class Control, bool kTrackPv,
          class Features = FeaturesNone, class Layout = TTNone, class Attacks>
CB_HD SearchResult search_task(const SearchTask<Board>& task, const SearchEnv<Attacks>& env, Storage& stack,
                               typename Ordering::Workspace& ws, Control& control, PvTable* pv_table) {
    using Level = SearchLevel<typename Ordering::State>;
    static_assert(Features::kTT || Layout::kEntries == 0, "a layout without the TT feature");
    const Attacks& a = env.attacks;
    const SearchTables& tables = *env.tables;
    const TTView<Layout> tt(env.tt);
    (void)tt;  // unused without the TT feature

    SearchResult result;
    SearchBoard<Board, Eval> sb = make_search_board<Eval>(task.board, a, tables);
    int level = 0;
    int ply = task.ply;
    int value = 0;  // score of the node just finished, from its own side's point of view

    Level cur;
    cur.key = sb.key;
    cur.alpha = task.alpha;
    cur.beta = task.beta;
    cur.depth = task.depth;
    cur.flags = u8(((task.flags & kTaskOnPv) ? level_flags::kOnPv : 0) |
                   ((task.flags & kTaskAfterNull) ? level_flags::kAfterNull : 0));
    int parent_top = 0;  // ListOrdering: end of the parent's move segment
    bool research = false;  // late move reductions: the next step searches cur.move again, at full depth
    // Killer moves by ply from the search root (feature sets with kKillers).
    Killers killers[Features::kKillers ? kMaxPly + 1 : 1];
    CheckInfo ci{};      // of the node at `level`; restored from cur when a child returns

    // A flat state machine: each pass of the loop takes one step, and every
    // path meets again at the top of the loop. Threads of a GPU warp running
    // different searches then fall back into step after each pass, instead
    // of drifting apart in nested loops (where they run one at a time).
    enum Step : int { kEnter, kPick, kReturn };
    int step = kEnter;
    for (;;) {
        if (step == kEnter) {
            // Enter the node at `level` (position sb, window and depth in cur).
            ++result.nodes;
            // Stop requests are polled every kStopPollInterval nodes; the
            // node budget is checked exactly (at budget + 1, or at a poll).
            if ((result.nodes & (kStopPollInterval - 1)) == 1 || result.nodes == u64(task.node_budget) + 1) {
                const bool over_budget = task.node_budget != 0 && result.nodes > task.node_budget;
                if (over_budget || control.should_stop(result.nodes)) {
                    result.aborted = true;
                    result.over_budget = over_budget;
                    result.best = Move();
                    return result;
                }
            }
            if constexpr (kTrackPv) pv_table->length[level] = 0;
            step = kReturn;  // unless the node has moves to search
            const u32 st = sb.board.state();
            if (ply > 0 && (detail::is_repetition(stack, level, sb.key, state::halfmove(st), task, env.history) ||
                            insufficient_material(sb.board))) {
                value = kScoreDraw;
            } else {
                ci = make_check_info(sb.board, a);
                if (ply > 0 && state::halfmove(st) >= 100) {
                    // The 50-move rule, unless this very position is checkmate.
                    value = (ci.in_check() && count_legal(sb.board, a) == 0) ? mated_in(ply) : kScoreDraw;
                } else if (ply >= kMaxPly) {
                    value = evaluate_position<Eval>(sb.board, sb.acc, ci.us, a);
                } else {
                    const bool quiescence = cur.depth <= 0;
                    cur.pinned = ci.pinned;
                    cur.checkmask = ci.checkmask;
                    cur.best = short(-kScoreInfinite);
                    cur.best_move = Move();
                    cur.tt_move = Move();
                    cur.alpha_orig = cur.alpha;
                    cur.moves_searched = 0;
                    cur.flags = u8((cur.flags & (level_flags::kOnPv | level_flags::kAfterNull)) |
                                   (ci.in_check() ? level_flags::kInCheck : 0) |
                                   (quiescence ? level_flags::kQuiescence : 0));
                    bool finished = false;  // scored without searching moves
                    bool use_tt = false;    // this node probes and stores the table
                    if constexpr (Features::kTT) {
                        use_tt = !quiescence || Features::kTTQuiescence;
                        if (use_tt) {
                            const TTUse u = tt_use<Features::kTTCutoffs>(
                                tt_probe(control, tt, cur.key), ply, cur.depth, quiescence,
                                (cur.flags & level_flags::kOnPv) != 0, cur.alpha, cur.beta);
                            cur.tt_move = u.move;  // checked for legality by the picker
                            if (u.cut) {
                                value = u.score;
                                finished = true;
                            }
                        }
                    }
                    if (!finished && quiescence && !ci.in_check()) {
                        // Stand pat: the side to move can usually do at least
                        // as well as the static evaluation by not capturing.
                        const int stand = evaluate_position<Eval>(sb.board, sb.acc, ci.us, a);
                        if (stand >= cur.beta) {
                            value = stand;
                            finished = true;
                            if (use_tt) cur.flags |= level_flags::kStore;
                        } else {
                            cur.best = short(stand);
                            if (stand > cur.alpha) cur.alpha = short(stand);
                        }
                    }
                    if (!finished) {
                        PickMode mode = (quiescence && !ci.in_check()) ? PickMode::Quiescence : PickMode::All;
                        if constexpr (qchecks_v<Features>) {
                            if (mode == PickMode::Quiescence && cur.depth == 0) mode = PickMode::QuiescenceChecks;
                        }
                        cur.pv_move = Move();
                        if (!quiescence && (cur.flags & level_flags::kOnPv) && env.pv != nullptr && ply < env.pv_length)
                            cur.pv_move = env.pv[ply];
                        const Move first = cur.pv_move.is_none() ? cur.tt_move : cur.pv_move;
                        Killers node_killers{};
                        if constexpr (Features::kKillers) node_killers = killers[ply];
                        Ordering::init(cur.picker, ws, parent_top, sb.board, a, ci, mode, first, node_killers);
                        if (Ordering::overflowed(cur.picker)) {
                            // The move arena is full: score the node
                            // statically. Only possible with a tiny arena.
                            value = evaluate_position<Eval>(sb.board, sb.acc, ci.us, a);
                        } else {
                            if (use_tt) cur.flags |= level_flags::kStore;
                            step = kPick;
                            if constexpr (Features::kNullMove) {
                                if (!quiescence && null_move_allowed<Eval>(env.prune, ply, cur.depth, cur.flags,
                                                                           ci.in_check(), cur.beta, sb.board, ci.us,
                                                                           sb.acc, a))
                                    cur.flags |= level_flags::kNullPending;
                            }
                        }
                    }
                }
            }
        } else if (step == kPick) {
            // Search the next move of the node at `level` (or first its null
            // move), or finish it.
            bool null_move = false;
            if constexpr (Features::kNullMove) null_move = (cur.flags & level_flags::kNullPending) != 0;
            Move m;
            if (research) {
                m = cur.move;  // a reduced move beat alpha: the same move again, at full depth
            } else if (!null_move) {
                m = Ordering::next(cur.picker, ws, sb.board, a, ci, cur.pv_move.is_none() ? cur.tt_move : cur.pv_move);
            }
            bool skip = false;  // futility or quiescence pruning skips m
            if constexpr (Features::kFutility) {
                if (!research && !null_move && !m.is_none() && !m.is_capture() && !m.is_promotion()) {
                    const int bound = futility_bound(env.prune, evaluate_position<Eval>(sb.board, sb.acc, ci.us, a),
                                                     cur.depth, cur.flags, cur.alpha);
                    if (bound <= cur.alpha && !gives_check(sb.board, m, a)) {
                        skip = true;
                        cur.flags |= level_flags::kSearchedAny;  // the node has moves: no stalemate
                        if (bound > cur.best) cur.best = short(bound);
                    }
                }
            }
            if constexpr (Features::kQPrune) {
                // At a quiescence node not in check (its moves are captures
                // and queen promotions): delta and SEE pruning.
                if (!null_move && !m.is_none() && (m.is_capture() || m.is_promotion()) &&
                    (cur.flags & (level_flags::kQuiescence | level_flags::kInCheck)) == level_flags::kQuiescence) {
                    const int stand = evaluate_position<Eval>(sb.board, sb.acc, ci.us, a);
                    const int bound = quiescence_prune_bound(sb.board, a, env.prune, m, stand, cur.alpha);
                    if (bound != kScoreInfinite) {
                        skip = true;
                        if (bound > cur.best) cur.best = short(bound);  // delta: at least the bound (fail-soft)
                    }
                }
            }
            if (skip) {
                // Skipped: the next step picks the next move.
            } else if (!null_move && m.is_none()) {
                if (cur.flags & level_flags::kSearchedAny) {
                    value = cur.best;
                } else if (cur.flags & level_flags::kInCheck) {
                    value = mated_in(ply);
                } else {
                    value = (cur.flags & level_flags::kQuiescence) ? cur.best : kScoreDraw;  // stand pat or stalemate
                }
                step = kReturn;
            } else {
                // Go down one ply.
                short child_alpha = short(-cur.beta);
                short child_beta = short(-cur.alpha);
                int child_depth = cur.depth - 1;
                u8 child_flags = 0;
                if (null_move) {
                    // The null move (an empty move on the stack), searched
                    // with the window (beta - 1, beta) at reduced depth.
                    cur.flags &= u8(~level_flags::kNullPending);
                    cur.move = Move();
                    cur.undo = make_null_search_move(sb, a, tables);
                    child_beta = short(1 - cur.beta);
                    child_depth -= null_move_reduction(env.prune, cur.depth);
                    child_flags = level_flags::kAfterNull;
                } else {
                    cur.undo = make_search_move(sb, m, a, tables);
                    if constexpr (abdada_v<Control>) {
                        static_assert(can_defer_v<Ordering>, "ABDADA needs an ordering that can defer moves");
                        // ABDADA: every move but the first (with defer_first,
                        // every move) goes back for later if another worker is
                        // searching its position (moves that come back are
                        // searched whatever the table says).
                        if (!research && cur.depth >= control.params.min_depth &&
                            (cur.moves_searched > 0 || control.params.defer_first) &&
                            !Ordering::returned_deferred(cur.picker, ws) && control.busy(sb.key)) {
                            unmake_search_move(sb, m, cur.undo, cur.key, tables);
                            Ordering::defer(cur.picker, ws);
                            control.count_deferred();
                            continue;  // the next step picks the next move
                        }
                    }
                    // A late move reduction (decided before the move gives
                    // check or not is known) is dropped if the move gives check.
                    int reduction = 0;
                    bool zero_window = false;  // PVS: a later move at a node with a real window
                    if (!research) {
                        ++cur.moves_searched;
                        if constexpr (pvs_v<Features>)
                            zero_window = cur.depth > 0 && cur.moves_searched > 1 && cur.beta - cur.alpha > 1;
                        if constexpr (Features::kLmr) {
                            if (!(cur.flags & level_flags::kInCheck) && !m.is_capture() && !m.is_promotion())
                                reduction = env.lmr->reduction(cur.depth, cur.moves_searched);
                        }
                    }
                    if constexpr (abdada_v<Control>) {
                        if (cur.depth >= control.params.min_depth) control.claim(sb.key);
                    }
                    research = false;
                    cur.flags |= level_flags::kSearchedAny;
                    cur.move = m;
                    if ((cur.flags & level_flags::kOnPv) && m == cur.pv_move) child_flags = level_flags::kOnPv;
                    if constexpr (Features::kCheckExt) {
                        if (cur.depth >= 1 && side_to_move_in_check(sb.board, a)) ++child_depth;
                    }
                    if constexpr (Features::kLmr) {
                        if (reduction > 0 && !side_to_move_in_check(sb.board, a)) {
                            // R plies shallower, with the zero window (alpha, alpha + 1).
                            cur.flags |= level_flags::kReduced;
                            child_alpha = short(-cur.alpha - 1);
                            child_depth -= reduction;
                        }
                    }
                    if constexpr (pvs_v<Features>) {
                        // PVS: the zero window first (searched again if it
                        // beats alpha; decided above, before `research` resets).
                        if (zero_window && !(cur.flags & level_flags::kReduced)) {
                            cur.flags |= level_flags::kReduced;
                            child_alpha = short(-cur.alpha - 1);
                        }
                    }
                }
                parent_top = Ordering::top(cur.picker);
                stack.store(level, cur);
                ++level;
                ++ply;
                cur.key = sb.key;
                cur.alpha = child_alpha;
                cur.beta = child_beta;
                cur.depth = static_cast<signed char>(child_depth);
                cur.flags = child_flags;
                step = kEnter;
            }
        } else {
            // The node at `level` is finished with `value`. Store it, then go back up.
            if constexpr (Features::kTT) {
                if (cur.flags & level_flags::kStore)
                    tt_store(control, tt, cur.key,
                             tt_entry(cur.best_move, value, ply, cur.depth, cur.alpha_orig, cur.beta, tt.generation));
            }
            if (level == 0) {
                result.score = value;
                result.best = cur.best_move;
                return result;
            }
            --level;
            --ply;
            const int child_value = value;
            cur = stack.load(level);
            bool from_null = false;  // the child was the null move
            if constexpr (Features::kNullMove) from_null = cur.move.is_none();
            if constexpr (abdada_v<Control>) {
                if (!from_null && cur.depth >= control.params.min_depth) control.release(sb.key);
            }
            if (from_null) {
                unmake_null_search_move(sb, cur.undo, cur.key);
            } else {
                unmake_search_move(sb, cur.move, cur.undo, cur.key, tables);
            }
            ci.pinned = cur.pinned;
            ci.checkmask = cur.checkmask;
            ci.us = state::side(sb.board.state());
            ci.king = king_square(sb.board, ci.us);
            parent_top = 0;  // unused until the next child is entered

            const int v = -child_value;
            step = kPick;
            bool reduced = false;  // the child got a reduced or a PVS zero-window search
            if constexpr (Features::kLmr || pvs_v<Features>) {
                reduced = (cur.flags & level_flags::kReduced) != 0;
                cur.flags &= u8(~level_flags::kReduced);
            }
            if (reduced && v > cur.alpha) {
                research = true;  // the next step searches the move again, at full depth
            } else if (from_null) {
                // Null-move cutoff: the node ends with the null result (a
                // lower bound; stored like any result). Otherwise its moves
                // come next.
                if (v >= cur.beta) {
                    value = null_move_score(v, cur.beta);
                    step = kReturn;
                }
            } else if (v > cur.best) {
                cur.best = short(v);
                cur.best_move = cur.move;
                if (v > cur.alpha) {
                    cur.alpha = short(v);
                    if constexpr (kTrackPv) {
                        pv_table->moves[level][0] = cur.move;
                        const int child_length = pv_table->length[level + 1];
                        for (int i = 0; i < child_length; ++i)
                            pv_table->moves[level][i + 1] = pv_table->moves[level + 1][i];
                        pv_table->length[level] = child_length + 1;
                    }
                    if (v >= cur.beta) {
                        value = v;  // cutoff (fail-soft: the best score found)
                        step = kReturn;
                        if constexpr (Features::kKillers) {
                            // A quiet move that cuts off becomes this ply's newest killer.
                            if (can_be_killer(cur.move)) killers[ply] = add_killer(killers[ply], cur.move);
                        }
                    }
                }
            }
        }
    }
}

}  // namespace cb

#pragma once

// The engine's search written recursively with plain move lists, with null
// moves, late move reductions, futility pruning, check extensions, killer
// moves, quiescence pruning and principal variation search as switches: the same rules, move order, pruning conditions and
// settings as core/search.h, spelled out again, so scores and node counts
// must match search_task exactly (tests/test_nmp.cpp, tests/test_lmr.cpp,
// tests/test_fut.cpp, tests/test_ext.cpp, tests/test_killers.cpp). No
// transposition table.

#include <algorithm>
#include <memory>
#include <utility>
#include <vector>

#include "core/search.h"
#include "host/host_tables.h"
#include "host/position.h"

namespace testref {

inline const cb::HqAttacks& attacks() {
    static const cb::HqAttacks a(cb::host_attack_tables());
    return a;
}

inline const cb::SearchTables& tables() { return cb::host_search_tables(); }

// `ply` counts from the search root, `level` from this search's first node
// (where its key list starts).
template <class Board, class Eval, bool kNullMove, bool kLmr, bool kFutility = false, bool kCheckExt = false,
          bool kKillers = false, bool kQPrune = false, bool kPvs = false, bool kQChecks = false>
struct RecursiveSearch {
    cb::SearchBoard<Board, Eval> sb;
    cb::PruneParams prune;
    cb::LmrTable lmr{};
    std::vector<cb::u64> keys;
    std::vector<cb::Move> pv_line;
    cb::u64 nodes = 0;
    cb::u64 null_tries = 0;
    cb::u64 null_cuts = 0;
    cb::u64 reductions = 0;
    cb::u64 researches = 0;
    cb::u64 futile = 0;
    cb::u64 extensions = 0;
    cb::Killers killers[cb::kMaxPly + 1] = {};  // by ply from the search root
    cb::u64 killer_updates = 0;
    cb::u64 delta_pruned = 0;
    cb::u64 see_pruned = 0;
    cb::u64 quiet_checks = 0;    // quiet checks searched in quiescence
    cb::u64 pvs_zero = 0;        // PVS zero-window searches
    cb::u64 pvs_researches = 0;  // ... searched again with the full window

    int search(int depth, int alpha, int beta, int ply, int level, bool on_pv, bool after_null) {
        using namespace cb;
        ++nodes;
        keys.resize(size_t(level + 1));
        keys[size_t(level)] = sb.key;
        const int halfmove = state::halfmove(sb.board.state());
        if (ply > 0) {
            for (int d = 4; d <= halfmove && d <= level; d += 2)
                if (keys[size_t(level - d)] == sb.key && ply - d > 0) return kScoreDraw;
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
        if constexpr (kNullMove) {
            // The null move, before any move (even before finding out whether
            // there is one).
            const Board& b = sb.board;
            const u64 pieces = (b.template pieces<KNIGHT>() | b.template pieces<BISHOP>() |
                                b.template pieces<ROOK>() | b.template pieces<QUEEN>()) &
                               b.color(ci.us);
            if (!quiescence && ply > 0 && depth >= prune.null_min_depth && !ci.in_check() && !on_pv && !after_null &&
                beta >= -kScoreMateBound && beta <= kScoreMateBound && pieces != 0 &&
                evaluate_position<Eval>(sb.board, sb.acc, ci.us) >= beta) {
                ++null_tries;
                const int r = prune.null_base + (prune.null_divisor ? depth / prune.null_divisor : 0);
                const u64 key_before = sb.key;
                const Undo undo = make_null_search_move(sb, attacks(), tables());
                const int v = -search(depth - 1 - r, -beta, -beta + 1, ply + 1, level + 1, false, true);
                unmake_null_search_move(sb, undo, key_before);
                if (v >= beta) {
                    ++null_cuts;
                    return v <= kScoreMateBound ? v : beta;
                }
            }
        }
        const Move pv = (!quiescence && on_pv && ply < int(pv_line.size())) ? pv_line[size_t(ply)] : Move();
        MoveList list;
        generate_legal(sb.board, attacks(), list);
        std::vector<std::pair<u32, Move>> ordered;
        for (int i = 0; i < list.size; ++i) {
            // Killers order every node searched with all its moves (not
            // quiescence's captures-and-checks nodes).
            const u32 key = order_key(sb.board, list.moves[i], pv,
                                      kKillers && !(quiescence && !ci.in_check()) ? killers[ply] : Killers{});
            if (quiescence && !ci.in_check() && !is_quiescence_group(key)) {
                // Quiet checks at quiescence's first ply (kQChecks).
                if (!(kQChecks && depth == 0 && is_quiet_check(sb.board, attacks(), list.moves[i]))) continue;
            }
            ordered.emplace_back(key, list.moves[i]);
        }
        std::sort(ordered.begin(), ordered.end(), [](const auto& x, const auto& y) { return x.first > y.first; });
        if (ordered.empty()) return ci.in_check() ? mated_in(ply) : (quiescence ? best : kScoreDraw);
        int number = 0;  // of the move, counting from 1 (futile moves don't count)
        for (const auto& [key, m] : ordered) {
            if constexpr (kQPrune) {
                // Quiescence, not in check, captures and queen promotions
                // (quiet checks are exempt): skip a move whose victim,
                // promotion gain and the margin can't lift the stand pat to
                // alpha (it scores at most that), or one that loses material
                // in the exchange.
                if (quiescence && !ci.in_check() && (m.is_capture() || m.is_promotion())) {
                    const int stand_pat = evaluate_position<Eval>(sb.board, sb.acc, ci.us);
                    const auto value = [](int pt) {
                        constexpr int v[7] = {0, 100, 320, 330, 500, 900, 20000};
                        return v[pt];
                    };
                    int gain = m.flags() == EP_CAPTURE ? 100 : value(sb.board.piece_type_on(m.to()));
                    if (m.is_promotion()) gain += value(m.promotion_type()) - 100;
                    if (prune.delta_margin > 0 && stand_pat + gain + prune.delta_margin <= alpha) {
                        ++delta_pruned;
                        best = std::max(best, stand_pat + gain + prune.delta_margin);
                        continue;
                    }
                    if (prune.see_prune && see(sb.board, attacks(), m) < 0) {
                        ++see_pruned;
                        continue;
                    }
                }
            }
            if constexpr (kFutility) {
                // Near the leaves, not in check, not on the PV, alpha not a
                // mate: a quiet move that doesn't give check is skipped if
                // the static evaluation plus the margin can't reach alpha.
                if (depth >= 1 && depth <= prune.fut_depth && !ci.in_check() && !on_pv && alpha >= -kScoreMateBound &&
                    alpha <= kScoreMateBound && !m.is_capture() && !m.is_promotion()) {
                    const int bound = evaluate_position<Eval>(sb.board, sb.acc, ci.us) + prune.fut_margin * depth;
                    Board after = sb.board;
                    make_move(after, m);
                    if (bound <= alpha && !make_check_info(after, attacks()).in_check()) {
                        ++futile;
                        best = std::max(best, bound);
                        continue;
                    }
                }
            }
            ++number;
            if (quiescence && !ci.in_check() && !m.is_capture() && !m.is_promotion()) ++quiet_checks;
            const bool child_pv = on_pv && m == pv;
            const u64 key_before = sb.key;
            const Undo undo = make_search_move(sb, m, attacks(), tables());
            const bool gives_check = make_check_info(sb.board, attacks()).in_check();
            int full = depth - 1;  // the child's depth without a reduction
            if constexpr (kCheckExt) {
                if (!quiescence && gives_check) {
                    ++extensions;
                    ++full;
                }
            }
            int v;
            int reduction = 0;
            if constexpr (kLmr) {
                // Late, quiet, not giving check, at a node not in check.
                if (!quiescence && !ci.in_check() && !m.is_capture() && !m.is_promotion() && !gives_check) {
                    const int d = std::min(depth, 63), n = std::min(number, 63);
                    reduction = lmr.r[d][n];
                }
            }
            if (reduction > 0) {
                ++reductions;
                v = -search(full - reduction, -alpha - 1, -alpha, ply + 1, level + 1, child_pv, false);
                if (v > alpha) {
                    ++researches;
                    v = -search(full, -beta, -alpha, ply + 1, level + 1, child_pv, false);
                }
            } else if (kPvs && !quiescence && number > 1 && beta - alpha > 1) {
                ++pvs_zero;
                v = -search(full, -alpha - 1, -alpha, ply + 1, level + 1, child_pv, false);
                if (v > alpha) {
                    ++pvs_researches;
                    v = -search(full, -beta, -alpha, ply + 1, level + 1, child_pv, false);
                }
            } else {
                v = -search(full, -beta, -alpha, ply + 1, level + 1, child_pv, false);
            }
            unmake_search_move(sb, m, undo, key_before, tables());
            if (v > best) {
                best = v;
                if (v > alpha) {
                    alpha = v;
                    if (v >= beta) {
                        // A quiet move (not castling) that cuts off becomes this ply's newest killer.
                        const u32 f = m.flags();
                        if (kKillers && !m.is_capture() && !m.is_promotion() && f != KING_CASTLE &&
                            f != QUEEN_CASTLE && killers[ply].first != m) {
                            killers[ply].second = killers[ply].first;
                            killers[ply].first = m;
                            ++killer_updates;
                        }
                        return best;
                    }
                }
            }
        }
        return best;
    }
};

template <class Board, class Eval, bool kNullMove, bool kLmr, bool kFutility = false, bool kCheckExt = false,
          bool kKillers = false, bool kQPrune = false, bool kPvs = false, bool kQChecks = false>
RecursiveSearch<Board, Eval, kNullMove, kLmr, kFutility, kCheckExt, kKillers, kQPrune, kPvs, kQChecks> reference(
    const cb::Position& p, const cb::PruneParams& prune, const std::vector<cb::Move>& pv) {
    RecursiveSearch<Board, Eval, kNullMove, kLmr, kFutility, kCheckExt, kKillers, kQPrune, kPvs, kQChecks> rec{
        cb::make_search_board<Eval>(cb::to_board<Board>(p), attacks(), tables())};
    rec.prune = prune;
    rec.lmr = cb::make_lmr_table(prune);
    rec.pv_line = pv;
    return rec;
}

struct Outcome {
    int score = 0;
    cb::u64 nodes = 0;
    cb::Move best;
    std::vector<cb::Move> pv;
};

// search_task with feature set Features and no table.
template <class Board, class Eval, class Features>
Outcome flat_search(const cb::Position& p, int depth, int alpha, int beta, int ply, cb::u8 flags,
                    const cb::PruneParams& prune, const std::vector<cb::Move>& pv_hint) {
    using namespace cb;
    SearchTask<Board> task;
    task.board = to_board<Board>(p);
    task.alpha = short(alpha);
    task.beta = short(beta);
    task.depth = static_cast<signed char>(depth);
    task.ply = u8(ply);
    task.flags = flags;
    task.history = -1;
    const LmrTable lmr = make_lmr_table(prune);
    const SearchEnv<HqAttacks> env{attacks(), &tables(), nullptr, pv_hint.empty() ? nullptr : pv_hint.data(),
                                   int(pv_hint.size()), TTDesc{}, prune, &lmr};
    using Stack = LocalStack<SearchLevel<StagedOrdering::State>, kMaxPly + 1>;
    const auto stack = std::make_unique<Stack>();
    const auto pv_table = std::make_unique<PvTable>();
    StagedOrdering::Workspace ws;
    NoStop control;
    const SearchResult r = search_task<Board, Eval, StagedOrdering, Stack, NoStop, true, Features, TTNone>(
        task, env, *stack, ws, control, pv_table.get());
    Outcome o;
    o.score = r.score;
    o.nodes = r.nodes;
    o.best = r.best;
    o.pv.assign(pv_table->moves[0], pv_table->moves[0] + pv_table->length[0]);
    return o;
}

}  // namespace testref

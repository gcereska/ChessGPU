#include "host/multipv.h"

#include <algorithm>

#include "core/movegen.h"
#include "host/host_tables.h"

namespace cb {

namespace {

// Is a position one ply after the root (key `key`, halfmove clock
// `halfmove`) a repetition the search scores as a draw? As
// detail::is_repetition (core/search.h) at ply 1: only game positions can
// repeat there, from 4 plies back, and only one that occurred before counts.
bool repeats_game_position(u64 key, int halfmove, const GameHistory& h) {
    int e = h.root_parent();
    int distance = 2;  // of entry e from the position
    for (int d = 4; e >= 0 && d <= halfmove; d += 2) {
        while (e >= 0 && distance < d) {
            e = h.entries[size_t(e)].prev;
            ++distance;
        }
        if (e < 0) break;
        if (h.entries[size_t(e)].key == key) return h.entries[size_t(e)].repeated != 0;
    }
    return false;
}

// One iteration: the `wanted` best root moves at `depth`, best first, into
// `lines`. The previous iteration's lines go first. False if a search was
// stopped (lines are then incomplete).
bool multipv_iteration(SearchBackend& backend, const std::vector<RootChild>& children,
                       const std::vector<RootLine>& previous, int depth, size_t wanted, std::vector<RootLine>& lines,
                       u64& nodes) {
    const auto previous_line = [&](Move m) -> const RootLine* {
        for (const RootLine& l : previous)
            if (l.move == m) return &l;
        return nullptr;
    };
    std::vector<const RootChild*> order;
    for (const RootLine& l : previous)
        for (const RootChild& c : children)
            if (c.move == l.move) order.push_back(&c);
    for (const RootChild& c : children)
        if (!previous_line(c.move)) order.push_back(&c);

    for (const RootChild* c : order) {
        RootLine line;
        line.move = c->move;
        line.pv = {c->move};
        if (c->draw) {
            line.score = kScoreDraw;
        } else {
            std::vector<Move> hint;
            if (const RootLine* p = previous_line(c->move); p && p->pv.size() > 1)
                hint.assign(p->pv.begin() + 1, p->pv.end());
            FixedDepthResult r;
            if (lines.size() < wanted) {
                r = backend.search_depth(c->position, c->history, depth - 1, hint);
                nodes += r.search.nodes;
                if (r.search.aborted) return false;
            } else {
                // Does it beat the worst line kept? From the child's side: does
                // it score below that line's score seen from there?
                const int bound = score_to_child(lines.back().score);
                const int alpha = std::max(-kScoreInfinite, bound - 1);
                const int beta = alpha + 1;
                r = backend.search_window(c->position, c->history, depth - 1, hint, alpha, beta);
                nodes += r.search.nodes;
                if (r.search.aborted) return false;
                if (r.search.score >= beta) continue;  // no better than the worst line
                if (backend.honors_window()) {         // (otherwise the score is exact already)
                    r = backend.search_window(c->position, c->history, depth - 1, hint, -kScoreInfinite, beta);
                    nodes += r.search.nodes;
                    if (r.search.aborted) return false;
                }
            }
            line.score = score_from_child(r.search.score);
            line.pv.insert(line.pv.end(), r.pv.begin(), r.pv.end());
        }
        // Best first; of equal scores, the one searched first.
        const auto at = std::find_if(lines.begin(), lines.end(), [&](const RootLine& l) { return line.score > l.score; });
        lines.insert(at, std::move(line));
        if (lines.size() > wanted) lines.pop_back();
    }
    return true;
}

}  // namespace

std::vector<RootChild> root_children(const Position& root, const GameHistory& history) {
    const HqAttacks a(host_attack_tables());
    const ZobristKeys& z = host_search_tables().zobrist;
    const QuadBoard b = to_board<QuadBoard>(root);
    MoveList moves;
    generate_legal(b, a, moves);
    HistoryEntry parent;  // the root, as a game position before each child
    parent.key = compute_key(b, a, z);
    parent.prev = history.root_parent();
    parent.ply = -1;
    parent.repeated = history.root_repeated ? 1 : 0;
    const int fullmove = root.fullmove + (state::side(b.state()) == BLACK ? 1 : 0);
    std::vector<RootChild> out;
    out.reserve(size_t(moves.size));
    for (int i = 0; i < moves.size; ++i) {
        RootChild c;
        c.move = moves.moves[i];
        QuadBoard child = b;
        make_move(child, c.move);
        c.position = to_position(child, fullmove);
        c.history.entries = history.entries;
        for (HistoryEntry& e : c.history.entries) e.ply = short(e.ply - 1);
        c.history.entries.push_back(parent);
        c.history.root_repeated = true;  // coming back to it is a draw, as to any position after the root
        const int halfmove = state::halfmove(child.state());
        // The 50-move rule doesn't apply to a checkmate (nor matter for a
        // stalemate): positions without moves are left to the search.
        c.draw = repeats_game_position(compute_key(child, a, z), halfmove, history) || insufficient_material(child) ||
                 (halfmove >= 100 && count_legal(child, a) > 0);
        out.push_back(std::move(c));
    }
    return out;
}

int score_from_child(int s) {
    if (s > kScoreMateBound) return -s + 1;
    if (s < -kScoreMateBound) return -s - 1;
    return -s;
}

int score_to_child(int s) {
    if (s > kScoreMateBound) return -s - 1;
    if (s < -kScoreMateBound) return -s + 1;
    return -s;
}

SearchOutcome multipv_deepening(SearchBackend& backend, const SearchRequest& request,
                                const std::function<void(const IterationReport&)>& report) {
    const std::vector<RootChild> children = root_children(request.root, request.history);
    if (children.empty()) {  // checkmate or stalemate: nothing to rank
        SearchRequest single = request;
        single.multipv = 1;
        return iterative_deepening(backend, single, report);
    }
    const Clock::time_point start = Clock::now();
    const auto elapsed_ms = [&] { return std::chrono::duration<double, std::milli>(Clock::now() - start).count(); };
    const size_t wanted = std::min(size_t(std::max(request.multipv, 1)), children.size());
    const int max_depth = std::clamp(request.max_depth, 1, kMaxPly - 1);
    SearchOutcome out;
    std::vector<RootLine> previous;
    backend.new_search();
    backend.set_deadline(Clock::time_point::max());
    for (int depth = 1; depth <= max_depth; ++depth) {
        if (depth > 1) {
            if (Clock::now() >= request.soft) break;
            if (request.max_nodes && out.nodes >= request.max_nodes) break;
            backend.set_deadline(request.hard);
        }
        std::vector<RootLine> lines;
        u64 nodes = 0;
        const bool finished = multipv_iteration(backend, children, previous, depth, wanted, lines, nodes);
        out.nodes += nodes;
        if (!finished) {
            if (depth > 1) break;
            // Stopped during depth 1 (a stop right after "go"): finish depth
            // 1 anyway so there is a move, then stop.
            backend.clear_stop();
            backend.set_deadline(Clock::time_point::max());
            lines.clear();
            nodes = 0;
            multipv_iteration(backend, children, previous, 1, wanted, lines, nodes);
            out.nodes += nodes;
            backend.request_stop();
            if (lines.empty()) break;  // stopped again: no move to give
        }
        previous = lines;
        out.best = lines.front().move;
        out.score = lines.front().score;
        out.depth = depth;
        out.pv = lines.front().pv;
        if (report) {
            IterationReport it;
            it.depth = depth;
            it.score = out.score;
            it.nodes = nodes;
            it.total_nodes = out.nodes;
            it.ms = elapsed_ms();
            it.pv = out.pv;
            it.lines = lines;
            report(it);
        }
    }
    backend.set_deadline(Clock::time_point::max());
    out.ms = elapsed_ms();
    return out;
}

}  // namespace cb

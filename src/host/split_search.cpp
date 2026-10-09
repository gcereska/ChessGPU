#include "host/split_search.h"

#include <algorithm>
#include <stdexcept>

#include "host/host_tables.h"
#include "host/profiling.h"

namespace cb {

namespace {

// Never consulted: explicit nodes have no stack below them.
struct NoStack {
    CB_HD u64 key(int) const { return 0; }
};

// The CPU part of search_independent: the first plies as explicit nodes.
class Splitter {
  public:
    Splitter(const GameHistory& history, int depth, int split, const std::vector<Move>& pv)
        : a_(host_attack_tables()), t_(host_search_tables()), history_(history), depth_(depth), split_(split), pv_(pv) {
        chain_ = history.entries;
    }

    struct Node {
        Move move;  // from the parent
        std::vector<int> children;
        int task = -1;          // leaf: its task
        bool terminal = false;  // decided by the rules alone
        int value = 0;
    };

    int build(const QuadBoard& b, int ply, int parent_entry, bool on_pv, Move move) {
        const int index = int(nodes_.size());
        nodes_.emplace_back();
        nodes_[size_t(index)].move = move;
        if (ply == split_) {
            TaskSpec task;
            task.board = b;
            task.depth = depth_ - split_;
            task.ply = ply;
            task.flags = u8((on_pv ? kTaskOnPv : 0) | (ply == 0 && history_.root_repeated ? kTaskRootRepeated : 0));
            task.history = parent_entry;
            nodes_[size_t(index)].task = int(tasks_.size());
            tasks_.push_back(task);
            return index;
        }
        ++explicit_nodes_;
        const u64 key = compute_key(b, a_, t_.zobrist);
        const int halfmove = state::halfmove(b.state());
        if (ply > 0) {
            SearchTask<QuadBoard> probe;
            probe.ply = u8(ply);
            probe.flags = 0;
            probe.history = parent_entry;
            if (detail::is_repetition(NoStack{}, 0, key, halfmove, probe, chain_.empty() ? nullptr : chain_.data()) ||
                insufficient_material(b))
                return terminal(index, kScoreDraw);
        }
        const CheckInfo ci = make_check_info(b, a_);
        if (ply > 0 && halfmove >= 100)
            return terminal(index, (ci.in_check() && count_legal(b, a_) == 0) ? mated_in(ply) : kScoreDraw);

        MoveList list;
        generate_legal(b, a_, list);
        if (list.size == 0) return terminal(index, ci.in_check() ? mated_in(ply) : kScoreDraw);
        const Move pv_move = (on_pv && ply < int(pv_.size())) ? pv_[size_t(ply)] : Move();
        std::vector<std::pair<u32, Move>> ordered;
        for (int i = 0; i < list.size; ++i) ordered.emplace_back(order_key(b, list.moves[i], pv_move), list.moves[i]);
        std::sort(ordered.begin(), ordered.end(), [](const auto& x, const auto& y) { return x.first > y.first; });

        HistoryEntry e;
        e.key = key;
        e.prev = parent_entry;
        e.ply = short(ply);
        e.repeated = u8(ply == 0 && history_.root_repeated ? 1 : 0);
        chain_.push_back(e);
        const int entry = int(chain_.size()) - 1;
        for (const auto& [k, m] : ordered) {
            QuadBoard child = b;
            make_move(child, m);
            const int c = build(child, ply + 1, entry, on_pv && m == pv_move, m);
            nodes_[size_t(index)].children.push_back(c);
        }
        return index;
    }

    // Minimax over the explicit nodes; records each node's best child.
    int value(int index, const std::vector<SearchResult>& results, bool& aborted) {
        const Node& n = nodes_[size_t(index)];
        if (n.terminal) return n.value;
        if (n.task >= 0) {
            aborted = aborted || results[size_t(n.task)].aborted;
            return results[size_t(n.task)].score;
        }
        int best = -kScoreInfinite;
        int best_child = -1;
        for (const int c : n.children) {
            const int v = -value(c, results, aborted);
            if (v > best) {
                best = v;
                best_child = c;
            }
        }
        best_child_.resize(nodes_.size(), -1);
        best_child_[size_t(index)] = best_child;
        return best;
    }

    // The principal variation: best children down to a task, then its best move.
    std::vector<Move> pv(const std::vector<SearchResult>& results) const {
        std::vector<Move> line;
        int index = 0;
        for (;;) {
            const Node& n = nodes_[size_t(index)];
            if (n.terminal) break;
            if (n.task >= 0) {
                if (!results[size_t(n.task)].best.is_none()) line.push_back(results[size_t(n.task)].best);
                break;
            }
            const int c = best_child_[size_t(index)];
            if (c < 0) break;
            line.push_back(nodes_[size_t(c)].move);
            index = c;
        }
        return line;
    }

    const std::vector<TaskSpec>& tasks() const { return tasks_; }
    const std::vector<HistoryEntry>& chain() const { return chain_; }
    size_t explicit_nodes() const { return explicit_nodes_; }

  private:
    int terminal(int index, int v) {
        nodes_[size_t(index)].terminal = true;
        nodes_[size_t(index)].value = v;
        return index;
    }

    HqAttacks a_;
    const SearchTables& t_;
    const GameHistory& history_;
    int depth_;
    int split_;
    const std::vector<Move>& pv_;
    std::vector<Node> nodes_;
    std::vector<TaskSpec> tasks_;
    std::vector<HistoryEntry> chain_;
    std::vector<int> best_child_;
    size_t explicit_nodes_ = 0;
};

}  // namespace

FixedDepthResult search_independent(TaskRunner& runner, const Position& root, const GameHistory& history, int depth,
                                    int split_depth, const std::vector<Move>& pv_hint, SplitSearchStats* stats) {
    if (depth < 0 || depth >= kMaxPly) throw std::invalid_argument("search depth out of range");
    const int split = std::clamp(split_depth, 0, depth);
    Splitter s(history, depth, split, pv_hint);
    {
        const NvtxRange range("split");
        s.build(to_board<QuadBoard>(root), 0, history.root_parent(), true, Move());
    }
    const std::vector<SearchResult> results = runner.run(s.tasks(), s.chain(), 0, pv_hint);

    FixedDepthResult r;
    bool aborted = false;
    r.search.score = s.value(0, results, aborted);
    r.search.aborted = aborted;
    r.search.nodes = s.explicit_nodes();
    for (const SearchResult& t : results) r.search.nodes += t.nodes;
    if (!aborted) {
        r.pv = s.pv(results);
        r.search.best = r.pv.empty() ? Move() : r.pv.front();
    }
    if (stats) {
        stats->split_depth = split;
        stats->explicit_nodes = s.explicit_nodes();
        stats->tasks = s.tasks().size();
        stats->batches = 1;
    }
    return r;
}

}  // namespace cb

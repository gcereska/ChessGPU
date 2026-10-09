#include "host/ybwc.h"

#include <algorithm>
#include <cstdio>
#include <stdexcept>

#include "core/ybwc_rules.h"
#include "host/host_tables.h"
#include "host/profiling.h"

namespace cb {

namespace {

// Never consulted: explicit nodes have no stack below them.
struct NoStack {
    CB_HD u64 key(int) const { return 0; }
};

enum class NodeState : u8 {
    Idle,      // created, not started
    Eldest,    // explicit: waiting for the first move's result
    Brothers,  // explicit: waiting for the other moves' results
    Queued,    // task: waiting for its batch
    Done,
};

struct Node {
    QuadBoard board;
    int parent = -1;
    int slot = 0;  // index of `move` among the parent's moves (lower = searched first)
    Move move;     // from the parent
    int entry = -1;  // history entry (explicit nodes); tasks link to their parent's
    int ply = 0;
    int depth = 0;
    bool on_pv = false;
    bool task = false;
    YbwcNodeType type = YbwcNodeType::Pv;
    NodeState state = NodeState::Idle;
    int alpha = 0;         // window, own point of view
    int beta = 0;
    int parent_alpha = 0;  // the parent's alpha this node was searched with: its result is exact above it
    // Explicit nodes.
    std::vector<Move> moves;    // in search order
    std::vector<int> children;  // node of each move once created, else -1
    int best = -kScoreInfinite;
    int best_child = -1;
    bool best_exact = false;
    int brothers_alpha = 0;  // alpha the young brothers start from
    int remaining = 0;       // young brothers not finished
    int next_slot = 0;       // next young brother to start
    int in_flight = 0;       // young brothers started and not finished
    int wave = 0;            // size of the last wave
    // Finished nodes.
    int value = 0;  // own point of view
    Move task_best;  // tasks: the best move their search found
};

class Coordinator {
  public:
    Coordinator(TaskRunner& runner, const GameHistory& history, int depth, const YbwcOptions& options,
                const std::vector<Move>& pv)
        : runner_(runner), a_(host_attack_tables()), t_(host_search_tables()), history_(history), depth_(depth),
          options_(options), pv_(pv), chain_(history.entries), eval_(runner.eval_kind()) {}

    FixedDepthResult run(const Position& root, YbwcStats* stats) {
        const Stopwatch total;
        Node r;
        r.board = to_board<QuadBoard>(root);
        r.depth = depth_;
        r.on_pv = true;
        r.task = !is_explicit(depth_, 0);
        nodes_.push_back(r);
        activate(0, -kScoreInfinite, kScoreInfinite);

        size_t chain_sent = 0;  // the runner already has chain_[0, chain_sent)
        std::vector<TaskSpec> tasks;
        std::vector<int> batch;
        while (nodes_[0].state != NodeState::Done) {
            build_batch(tasks, batch);
            if (tasks.empty()) throw std::logic_error("YBWC: no task is ready but the root is unfinished");
            ++stats_.rounds;
            stats_.tasks += tasks.size();
            stats_.max_batch = std::max(stats_.max_batch, tasks.size());
            const Stopwatch timer;
            const std::vector<SearchResult> results = runner_.run(tasks, chain_, chain_sent, pv_);
            stats_.runner_ms += timer.elapsed_ms();
            stats_.device_ms += runner_.last_device_ms();
            chain_sent = chain_.size();
            for (const SearchResult& res : results) nodes_searched_ += res.nodes;
            for (const SearchResult& res : results) {
                if (res.aborted && !res.over_budget) {
                    aborted_ = true;
                    break;
                }
            }
            if (aborted_) break;
            for (size_t i = 0; i < batch.size(); ++i) {
                const int n = batch[i];
                if (ancestor_done(n)) continue;  // a sibling's result already cut its parent off
                if (results[i].over_budget) {
                    ++stats_.over_budget;
                    split_task(n);
                    continue;
                }
                nodes_[size_t(n)].task_best = results[i].best;
                finish(n, results[i].score);
            }
        }

        FixedDepthResult out;
        out.search.nodes = nodes_searched_ + stats_.explicit_nodes;
        out.search.aborted = aborted_;
        if (!aborted_) {
            out.search.score = nodes_[0].value;
            out.pv = principal_variation();
            out.search.best = out.pv.empty() ? Move() : out.pv.front();
        }
        stats_.total_ms = total.elapsed_ms();
        if (stats) *stats = stats_;
        return out;
    }

  private:
    bool is_explicit(int depth, int ply) const { return depth > options_.leaf_depth && ply < options_.max_explicit_ply; }

    // Entry of the position before node n (where its history chain starts).
    int parent_entry(int n) const {
        const int p = nodes_[size_t(n)].parent;
        return p < 0 ? history_.root_parent() : nodes_[size_t(p)].entry;
    }

    bool ancestor_done(int n) const {
        for (int p = nodes_[size_t(n)].parent; p >= 0; p = nodes_[size_t(p)].parent)
            if (nodes_[size_t(p)].state == NodeState::Done) return true;
        return false;
    }

    int create_child(int p, int slot) {
        Node c;
        const Node& parent = nodes_[size_t(p)];
        c.board = parent.board;
        make_move(c.board, parent.moves[size_t(slot)]);
        c.parent = p;
        c.slot = slot;
        c.move = parent.moves[size_t(slot)];
        c.ply = parent.ply + 1;
        c.depth = parent.depth - 1;
        const Move pv_move = parent.ply < int(pv_.size()) ? pv_[size_t(parent.ply)] : Move();
        c.on_pv = parent.on_pv && c.move == pv_move;
        c.task = !is_explicit(c.depth, c.ply);
        c.type = child_type(parent.type, slot);
        const int index = int(nodes_.size());
        nodes_.push_back(c);
        nodes_[size_t(p)].children[size_t(slot)] = index;
        return index;
    }

    // A task ran out of node budget: search its root as an explicit node, so
    // its moves become tasks (with the budget again).
    void split_task(int n) {
        Node& node = nodes_[size_t(n)];
        node.task = false;
        node.state = NodeState::Idle;
        activate(n, node.alpha, node.beta);
    }

    int static_eval(const QuadBoard& b) const {
        const Color us = state::side(b.state());
        if (eval_ == EvalKind::Material) return MaterialEval::evaluate(compute_accumulator<MaterialEval>(b, t_.eval), us);
        if (eval_ == EvalKind::Terms)
            return evaluate_position<PestoTermsEval>(b, compute_accumulator<PestoTermsEval>(b, t_.eval), us);
        if (eval_ == EvalKind::Nnue) return NnueEval::evaluate(compute_accumulator<NnueEval>(b, t_.eval), us);
        return PestoEval::evaluate(compute_accumulator<PestoEval>(b, t_.eval), us);
    }

    // Starts node n with window (alpha, beta), its own point of view.
    void activate(int n, int alpha, int beta) {
        {
            Node& node = nodes_[size_t(n)];
            node.alpha = alpha;
            node.beta = beta;
            if (node.task) {
                node.state = NodeState::Queued;
                pending_.push_back(n);
                return;
            }
        }
        ++stats_.explicit_nodes;
        const QuadBoard b = nodes_[size_t(n)].board;
        const int ply = nodes_[size_t(n)].ply;
        const u64 key = compute_key(b, a_, t_.zobrist);
        const int halfmove = state::halfmove(b.state());
        if (ply > 0) {
            SearchTask<QuadBoard> probe;
            probe.ply = u8(ply);
            probe.flags = 0;
            probe.history = parent_entry(n);
            if (detail::is_repetition(NoStack{}, 0, key, halfmove, probe, chain_.empty() ? nullptr : chain_.data()) ||
                insufficient_material(b)) {
                finish(n, kScoreDraw);
                return;
            }
        }
        const CheckInfo ci = make_check_info(b, a_);
        if (ply > 0 && halfmove >= 100) {
            finish(n, (ci.in_check() && count_legal(b, a_) == 0) ? mated_in(ply) : kScoreDraw);
            return;
        }
        if (ply >= kMaxPly) {
            finish(n, static_eval(b));
            return;
        }
        // The same node logic as search_task: quiescence nodes stand pat and
        // try only the quiescence moves, unless in check.
        const bool quiescence = nodes_[size_t(n)].depth <= 0;
        if (quiescence) ++stats_.explicit_quiescence;
        const bool stand_pat = quiescence && !ci.in_check();
        int best = -kScoreInfinite;
        if (stand_pat) {
            best = static_eval(b);
            if (best >= beta) {
                finish(n, best);
                return;
            }
            alpha = std::max(alpha, best);
        }
        MoveList list;
        generate_legal(b, a_, list);
        const Move pv_move =
            (!quiescence && nodes_[size_t(n)].on_pv && ply < int(pv_.size())) ? pv_[size_t(ply)] : Move();
        std::vector<std::pair<u32, Move>> ordered;
        for (int i = 0; i < list.size; ++i) {
            const u32 k = order_key(b, list.moves[i], pv_move);
            if (stand_pat && !is_quiescence_group(k)) continue;
            ordered.emplace_back(k, list.moves[i]);
        }
        if (ordered.empty()) {
            finish(n, ci.in_check() ? mated_in(ply) : (quiescence ? best : kScoreDraw));
            return;
        }
        std::sort(ordered.begin(), ordered.end(), [](const auto& x, const auto& y) { return x.first > y.first; });

        HistoryEntry e;
        e.key = key;
        e.prev = parent_entry(n);
        e.ply = short(ply);
        e.repeated = u8(ply == 0 && history_.root_repeated ? 1 : 0);
        chain_.push_back(e);
        {
            Node& node = nodes_[size_t(n)];
            node.entry = int(chain_.size()) - 1;
            for (const auto& [k, m] : ordered) node.moves.push_back(m);
            node.children.assign(node.moves.size(), -1);
            node.state = NodeState::Eldest;
            node.alpha = alpha;  // raised by the stand pat
            node.best = best;    // the stand pat, or -infinity
            node.best_child = -1;
            node.best_exact = false;
        }
        const int eldest = create_child(n, 0);
        nodes_[size_t(eldest)].parent_alpha = alpha;
        activate(eldest, -beta, -alpha);
    }

    void finish(int n, int value) {
        Node& node = nodes_[size_t(n)];
        node.state = NodeState::Done;
        node.value = value;
        if (node.parent >= 0) report(node.parent, n);
    }

    // Child c of explicit node p has finished.
    void report(int p, int c) {
        if (nodes_[size_t(p)].state == NodeState::Done) return;
        const int v = -nodes_[size_t(c)].value;
        const bool exact = v > nodes_[size_t(c)].parent_alpha;
        if (nodes_[size_t(p)].state == NodeState::Eldest) {
            Node& node = nodes_[size_t(p)];
            if (v > node.best) {  // always, except below a stand pat
                node.best = v;
                node.best_child = c;
                node.best_exact = exact;
            }
            if (node.best >= node.beta) {
                finish(p, node.best);
                return;
            }
            node.brothers_alpha = std::max(node.alpha, node.best);
            node.remaining = int(node.moves.size()) - 1;
            node.next_slot = 1;
            node.in_flight = 0;
            node.wave = 0;
            node.state = NodeState::Brothers;
            if (node.remaining == 0) {
                finish(p, node.best);
                return;
            }
            start_wave(p);
            return;
        }
        Node& node = nodes_[size_t(p)];
        --node.remaining;
        --node.in_flight;
        const int best_slot = node.best_child >= 0 ? nodes_[size_t(node.best_child)].slot : 1 << 30;
        if (better_child(v, exact, nodes_[size_t(c)].slot, node.best, node.best_exact, best_slot)) {
            node.best = v;
            node.best_child = c;
            node.best_exact = exact;
        }
        if (v >= node.beta) {
            finish(p, node.best);
            return;
        }
        if (node.remaining == 0) {
            finish(p, node.best);
            return;
        }
        if (node.in_flight == 0) start_wave(p);  // the last wave ended without a cutoff
    }

    // Starts the next young brothers of p: all that are left, or the next
    // wave at a CUT node with cut_waves. Starting one can finish it at once
    // (a draw by rule) and even cut p off.
    void start_wave(int p) {
        Node& node = nodes_[size_t(p)];
        const int count = int(node.moves.size());
        int size = count - node.next_slot;
        if (options_.cut_waves && node.type == YbwcNodeType::Cut) {
            node.wave = node.wave == 0 ? 1 : node.wave * 2;
            size = std::min(size, node.wave);
        }
        const int first = node.next_slot;
        node.next_slot += size;
        node.in_flight += size;
        // Later waves start from the best score found so far.
        const int a = std::max(node.brothers_alpha, node.best);
        for (int slot = first; slot < first + size && nodes_[size_t(p)].state != NodeState::Done; ++slot) {
            const int child = create_child(p, slot);
            nodes_[size_t(child)].parent_alpha = a;
            activate(child, -nodes_[size_t(p)].beta, -a);
        }
    }

    // Collects the ready tasks. Young brothers that are tasks start from
    // their parent's best score so far, which can be better than the bound
    // they were made ready with.
    void build_batch(std::vector<TaskSpec>& tasks, std::vector<int>& batch) {
        tasks.clear();
        batch.clear();
        for (const int n : pending_) {
            if (ancestor_done(n)) {
                ++stats_.skipped;
                continue;
            }
            Node& node = nodes_[size_t(n)];
            if (node.parent >= 0 && node.slot > 0) {
                const Node& p = nodes_[size_t(node.parent)];
                const int a = std::max(p.brothers_alpha, p.best);
                node.parent_alpha = a;
                node.alpha = -p.beta;
                node.beta = -a;
            }
            TaskSpec t;
            t.board = node.board;
            t.alpha = node.alpha;
            t.beta = node.beta;
            t.depth = node.depth;
            t.ply = node.ply;
            t.flags = u8((node.on_pv ? kTaskOnPv : 0) | (node.ply == 0 && history_.root_repeated ? kTaskRootRepeated : 0));
            t.history = parent_entry(n);
            t.node_budget = options_.node_budget;
            tasks.push_back(t);
            batch.push_back(n);
        }
        pending_.clear();
    }

    std::vector<Move> principal_variation() const {
        std::vector<Move> line;
        int n = 0;
        for (;;) {
            const Node& node = nodes_[size_t(n)];
            if (node.task) {
                if (!node.task_best.is_none()) line.push_back(node.task_best);
                break;
            }
            if (node.best_child < 0) break;  // decided by the rules
            n = node.best_child;
            line.push_back(nodes_[size_t(n)].move);
        }
        return line;
    }

    TaskRunner& runner_;
    HqAttacks a_;
    const SearchTables& t_;
    const GameHistory& history_;
    int depth_;
    YbwcOptions options_;
    const std::vector<Move>& pv_;
    std::vector<Node> nodes_;
    std::vector<HistoryEntry> chain_;
    std::vector<int> pending_;
    YbwcStats stats_;
    u64 nodes_searched_ = 0;
    bool aborted_ = false;
    EvalKind eval_;
};

}  // namespace

std::string describe(const YbwcStats& s) {
    char text[256];
    std::snprintf(text, sizeof text,
                  "rounds %d | tasks %zu (max batch %zu, skipped %zu, over budget %zu) | explicit %zu (quiescence "
                  "%zu) | runner %.1f ms (device %.1f) of %.1f ms",
                  s.rounds, s.tasks, s.max_batch, s.skipped, s.over_budget, s.explicit_nodes, s.explicit_quiescence,
                  s.runner_ms, s.device_ms, s.total_ms);
    return text;
}

FixedDepthResult search_ybwc(TaskRunner& runner, const Position& root, const GameHistory& history, int depth,
                             const YbwcOptions& options, const std::vector<Move>& pv_hint, YbwcStats* stats) {
    if (depth < 0 || depth >= kMaxPly) throw std::invalid_argument("search depth out of range");
    if (options.leaf_depth < 0 || options.max_explicit_ply < 0)
        throw std::invalid_argument("YBWC leaf depth and explicit plies must be >= 0");
    const NvtxRange range("ybwc");
    Coordinator c(runner, history, depth, options, pv_hint);
    return c.run(root, stats);
}

}  // namespace cb

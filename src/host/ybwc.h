#pragma once

// YBWC (Young Brothers Wait Concept) coordinated by the CPU in rounds.
//
// The top of the tree is a set of explicit nodes on the CPU; below them,
// subtrees are tasks for a TaskRunner. At every explicit node the first
// (eldest) move is searched alone; only when it doesn't cut off are the
// other moves (young brothers) searched, all in parallel, with its score as
// the bound. Each round hands every task that is ready to the runner as one
// batch, then feeds the results back up the tree, which makes more tasks
// ready for the next round. A node is explicit while its remaining depth is
// above `leaf_depth` and its ply below `max_explicit_ply`.
//
// Classic YBWC starts all young brothers at once. At a node expected to cut
// off (a CUT node) whose first move didn't, the sequential search stops at
// the next move that does, often the second or third, while classic YBWC
// searches all of them. With `cut_waves`, young brothers of expected CUT
// nodes start in waves of 1, 2, 4, ... moves, each wave after the previous
// one finished without a cutoff; PV and ALL nodes still start all at once.
// Node types are predicted as usual: the root and first children of PV nodes
// are PV, other children of PV nodes and all children of ALL nodes are CUT,
// children of CUT nodes are ALL.
//
// With a `node_budget`, a task stops once it has searched that many nodes;
// the coordinator then turns its root into an explicit node, splitting the
// big subtree one more ply (its children become tasks, with the budget
// again). The explicit tree grows where subtrees are big, and no single task
// holds up a round for long. Quiescence tasks split the same way: explicit
// quiescence nodes stand pat and try the same moves as the sequential search.
//
// Deterministic: the same runner results give the same tree, so CPU and GPU
// runners give identical searches. The score equals the sequential search's.

#include <string>
#include <vector>

#include "host/game_history.h"
#include "host/search_cpu.h"
#include "host/task_runner.h"

namespace cb {

// Defaults: the fastest settings on the time-to-depth suite (docs/ARCHITECTURE.md, Results).
struct YbwcOptions {
    int leaf_depth = 2;         // tasks search at most this deep (plus quiescence)
    int max_explicit_ply = 8;   // nodes at this ply are always tasks
    bool cut_waves = true;      // young brothers of CUT nodes in growing waves (see above)
    u32 node_budget = 2048;     // split tasks that search more nodes than this (0: never; see above)
};

struct YbwcStats {
    size_t explicit_nodes = 0;
    size_t explicit_quiescence = 0;  // of which quiescence nodes (split quiescence tasks)
    size_t tasks = 0;      // tasks run
    size_t skipped = 0;    // ready tasks dropped because an ancestor had already cut off
    size_t over_budget = 0;  // tasks that ran out of node budget (and were split or rerun)
    int rounds = 0;
    size_t max_batch = 0;
    double runner_ms = 0;  // time inside TaskRunner::run
    double device_ms = 0;  // of which the device spent searching (GPU kernels)
    double total_ms = 0;
};

std::string describe(const YbwcStats& s);

FixedDepthResult search_ybwc(TaskRunner& runner, const Position& root, const GameHistory& history, int depth,
                             const YbwcOptions& options, const std::vector<Move>& pv_hint = {},
                             YbwcStats* stats = nullptr);

}  // namespace cb

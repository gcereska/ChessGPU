#pragma once

// Parallel searches that split the tree on the CPU and hand the subtrees to
// a TaskRunner. Common parts (explicit nodes above the tasks) live here.

#include <vector>

#include "host/game_history.h"
#include "host/search_cpu.h"
#include "host/task_runner.h"

namespace cb {

struct SplitSearchStats {
    int split_depth = 0;
    size_t explicit_nodes = 0;  // positions above the tasks, searched on the CPU
    size_t tasks = 0;
    int batches = 0;
};

// Fixed-depth search of the root with independent subtrees: the CPU expands
// the first `split_depth` plies, the runner searches every position at that
// ply with a full window (no bounds shared between subtrees), and the CPU
// combines the scores by minimax. The score and best move are exactly those
// of the sequential search. The "independent subtrees" baseline
// (docs/ARCHITECTURE.md, Parallel search on the GPU).
FixedDepthResult search_independent(TaskRunner& runner, const Position& root, const GameHistory& history, int depth,
                                    int split_depth, const std::vector<Move>& pv_hint = {},
                                    SplitSearchStats* stats = nullptr);

}  // namespace cb

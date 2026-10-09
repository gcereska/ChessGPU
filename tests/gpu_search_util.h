#pragma once

// Helpers shared by the GPU search tests.

#include <vector>

#include "core/search.h"
#include "gpu/search_gpu.h"
#include "host/host_tables.h"
#include "host/search_cpu.h"

namespace gpu_search_util {

inline const cb::HqAttacks& host_attacks() {
    static const cb::HqAttacks a(cb::host_attack_tables());
    return a;
}

// Every compiled search variant (model, ordering, stack), for one board and eval.
inline std::vector<cb::GpuSearchOptions> compiled_variants(cb::BoardKind board, cb::EvalKind eval) {
    using namespace cb;
    std::vector<GpuSearchOptions> v;
    for (const ExecModel model : compiled_models()) {
        GpuSearchOptions o;
        o.board = board;
        o.eval = eval;
        o.model = model;
        if (model == ExecModel::Warp) {
            o.block_size = 64;
            v.push_back(o);
            continue;
        }
        for (const OrderingKind ordering : {OrderingKind::Staged, OrderingKind::List}) {
            if (!is_compiled(ordering)) continue;
            o.ordering = ordering;
            for (const StackKind stack : {StackKind::Local, StackKind::Shared}) {
                if (ordering == OrderingKind::List && stack == StackKind::Shared) continue;
                if (ordering == OrderingKind::Staged && !is_compiled(stack)) continue;
                o.stack = stack;
                v.push_back(o);
            }
        }
    }
    return v;
}

// Under compute-sanitizer (--quick), a variant x feature-set matrix is
// covered pairwise: every variant with the first feature set, and the first
// variant with every feature set. That runs each kernel's code at a
// fraction of the full product's cost.
inline bool quick_skip(bool quick, size_t variant, size_t features) { return quick && variant != 0 && features != 0; }

// The score `move` gets at `depth` from p (the child searched as a root of
// its own at depth - 1, quiescence for depth 1), without a table: tells
// whether a best move that differs from the sequential search's ties with it.
inline int score_of_move(const cb::Position& p, cb::Move m, int depth, cb::EvalKind eval) {
    using namespace cb;
    QuadBoard b = to_board<QuadBoard>(p);
    make_move(b, m);
    const int child = cpu_search_fixed(to_position(b), GameHistory{}, depth - 1, -kScoreInfinite, kScoreInfinite,
                                       {BoardKind::Quad, eval, OrderingKind::Staged})
                          .search.score;
    int s = -child;
    if (s > kScoreMateBound) s -= 1;  // a mate seen from the child is one ply further from p
    if (s < -kScoreMateBound) s += 1;
    return s;
}

}  // namespace gpu_search_util

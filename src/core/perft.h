#pragma once

// Perft (count leaf positions of the legal move tree) as an explicit-stack
// depth-first loop, so the same code runs on the CPU and on a GPU thread.

#include <cassert>

#include "core/move32.h"

namespace cb {

// Per-level DFS state. The caller owns it, so where it lives (registers,
// local memory, shared memory, host stack) can vary independently of the
// algorithm. List: MoveList, or MoveBuffer32 for 32-bit moves (Moves16 and
// Moves32 in core/move32.h).
template <int kMaxLevels, class List = MoveList>
struct PerftStack {
    List lists[kMaxLevels];      // legal moves at each level
    Undo undo[kMaxLevels];       // undo info of the move currently played at each level
    int next[kMaxLevels];        // index of the next move to try at each level
};

// Number of leaf positions `depth` plies below b. The last ply is counted
// without playing its moves ("bulk counting"). Needs depth - 1 <= kMaxLevels.
template <class Board, class Attacks, int kMaxLevels, class List>
CB_HD u64 perft_iterative(Board b, const Attacks& a, int depth, PerftStack<kMaxLevels, List>& st) {
    if (depth <= 0) return 1;
    if (depth == 1) return count_legal(b, a);
    assert(depth - 1 <= kMaxLevels);

    st.lists[0].bind(b);
    generate_legal(b, a, st.lists[0]);
    st.next[0] = 0;
    int ply = 0;
    u64 nodes = 0;
    for (;;) {
        if (st.next[ply] == st.lists[ply].size) {
            // Every move at this level is done: go back up and take back the
            // parent's move.
            if (ply == 0) break;
            --ply;
            unmake_move(b, st.lists[ply].moves[st.next[ply] - 1], st.undo[ply]);
            continue;
        }
        const auto m = st.lists[ply].moves[st.next[ply]++];
        st.undo[ply] = make_move(b, m);
        if (depth - ply - 1 == 1) {
            nodes += count_legal(b, a);
            unmake_move(b, m, st.undo[ply]);
        } else {
            ++ply;
            st.lists[ply].bind(b);
            generate_legal(b, a, st.lists[ply]);
            st.next[ply] = 0;
        }
    }
    return nodes;
}

}  // namespace cb

#pragma once

// ABDADA (docs/ARCHITECTURE.md, Parallel search on the GPU): workers
// search the same tree from the root, sharing the transposition table, and
// coordinate through a table of positions being searched. At a node, every
// move but the first is checked once made: if another worker is searching the
// position it leads to, the move is deferred (the list picker hands it back
// after all the others) and searched then, whatever the table says. Workers
// claim each position they enter and release it when they leave.
//
// This is the "simplified ABDADA" form: a separate table of claims instead of
// a counter in each transposition table entry, so the table layouts stay as
// they are. A slot holds a position's key (its top 40 bits) and the claiming
// worker (its low 24 bits, worker + 1); a worker never defers to its own
// claim. Slots are written with relaxed atomics: a lost or stale claim only
// costs some duplicated work, never a wrong result.

#include "core/types.h"

namespace cb {

// Searches use ABDADA at main-search nodes with at least this many plies left
// (claims below would cost more traffic than the work they save): the default
// of AbdadaParams::min_depth.
constexpr int kAbdadaMinDepth = 2;

// Run-time settings of the coordination (UCI AbdadaMinDepth, AbdadaDeferFirst),
// carried by the workers' Control: one compiled search serves every setting.
struct AbdadaParams {
    u8 min_depth = kAbdadaMinDepth;  // plies left from which moves are claimed and deferred (1 to 20)
    u8 defer_first = 0;              // 1: a node's first move is deferred too while another worker searches it
};

namespace abdada {

constexpr u64 kWorkerMask = 0xFFFFFFull;  // low 24 bits: worker + 1
constexpr u64 kKeyMask = ~kWorkerMask;

CB_HD constexpr u64 slot_value(u64 key, u32 worker) { return (key & kKeyMask) | (u64(worker) + 1); }

// Whether `slot` claims position `key` for a worker other than `worker`.
CB_HD constexpr bool busy_for(u64 slot, u64 key, u32 worker) {
    const u64 owner = slot & kWorkerMask;
    return owner != 0 && owner != u64(worker) + 1 && (slot & kKeyMask) == (key & kKeyMask);
}

}  // namespace abdada

}  // namespace cb

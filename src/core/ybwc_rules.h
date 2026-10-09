#pragma once

// Rules shared by both YBWC coordinators (host/ybwc.cpp, gpu/ybwc_device.cu).

#include "core/types.h"

namespace cb {

// Should a finished child replace the best child of an explicit node? `v` is
// the child's score from the node's point of view; it is exact when it beat
// the alpha the child was searched with, else only an upper bound. The
// highest score wins; on equal scores an exact one beats a bound (a bound
// can equal the best without being as good), and between exact scores the
// earlier move in search order wins. That is the move the sequential search
// picks: the first one reaching the best score.
CB_HD constexpr bool better_child(int v, bool exact, int slot, int best, bool best_exact, int best_slot) {
    return v > best || (v == best && exact && (!best_exact || slot < best_slot));
}

// Predicted node types (Knuth and Moore): the root and the first child of a
// PV node are PV nodes; other children of PV nodes and all children of ALL
// nodes are CUT nodes; children of CUT nodes are ALL nodes.
enum class YbwcNodeType : u8 { Pv, Cut, All };

CB_HD constexpr YbwcNodeType child_type(YbwcNodeType parent, int slot) {
    return parent == YbwcNodeType::Pv  ? (slot == 0 ? YbwcNodeType::Pv : YbwcNodeType::Cut)
           : parent == YbwcNodeType::Cut ? YbwcNodeType::All
                                         : YbwcNodeType::Cut;
}

}  // namespace cb

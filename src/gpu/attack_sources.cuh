#pragma once

// Where kernels find the tables of their sliding-attack variant
// (docs/ARCHITECTURE.md, Move generation): the attack tables every variant uses, plus the
// magic bitboard tables for MagicAttacks (device pointers). Kernels take the
// variant as a template parameter and build it with make_attacks.

#include "core/attacks.h"
#include "core/attacks_kogge.h"
#include "core/attacks_magic.h"

namespace cb::gpu {

struct AttackSources {
    const AttackTables* tables;
    const MagicTables* magic;  // MagicAttacks only (may be null otherwise)
};

template <class Attacks>
__device__ Attacks make_attacks(const AttackSources& s);
template <>
__device__ inline HqAttacks make_attacks<HqAttacks>(const AttackSources& s) {
    return HqAttacks(*s.tables);
}
template <>
__device__ inline MagicAttacks make_attacks<MagicAttacks>(const AttackSources& s) {
    return MagicAttacks(*s.tables, *s.magic);
}
template <>
__device__ inline KoggeStoneAttacks make_attacks<KoggeStoneAttacks>(const AttackSources& s) {
    return KoggeStoneAttacks(*s.tables);
}

}  // namespace cb::gpu

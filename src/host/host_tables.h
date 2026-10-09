#pragma once

#include "core/attacks.h"
#include "core/attacks_magic.h"
#include "core/search_board.h"

namespace cb {

// Tables for host code, built on first use. Thread-safe.
const AttackTables& host_attack_tables();
const SearchTables& host_search_tables();
const MagicTables& host_magic_tables();  // magic bitboards: searched for on first use

// Replaces the NNUE weights of host_search_tables() (chessgpu --nnue-file).
// Only before anything searches or creates a device context, which copies
// the tables to the GPU: nothing else may use the tables meanwhile.
void set_host_nnue_weights(const NnueWeights& w);

}  // namespace cb

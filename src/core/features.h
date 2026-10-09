#pragma once

// Search feature sets: which pruning techniques a search uses
// (docs/ARCHITECTURE.md, Search). Each set is a template parameter of the search, selected at
// launch like Board and Eval, so techniques that are switched off cost
// nothing. The sets are listed in core/feature_list.h; every set stays
// available (the CMake option CHESSGPU_FEATURES chooses which get compiled
// for the GPU).
//
// Techniques with numbers to tune take them at run time (PruneParams), so
// one compiled search serves every setting.

#include <cmath>

#include "core/feature_list.h"
#include "core/types.h"

namespace cb {

// The techniques a feature set uses, as flags (core/feature_list.h lists the
// sets).
namespace ft {
constexpr u32 kTT = 1u << 0;            // probe and store the transposition table
constexpr u32 kTTCutoffs = 1u << 1;     // stored results cut off (else the table only orders moves)
constexpr u32 kTTQuiescence = 1u << 2;  // the table in quiescence search too
constexpr u32 kNullMove = 1u << 3;      // null-move pruning
constexpr u32 kLmr = 1u << 4;           // late move reductions
constexpr u32 kFutility = 1u << 5;      // futility pruning
constexpr u32 kCheckExt = 1u << 6;      // check extensions
constexpr u32 kKillers = 1u << 7;       // killer moves
constexpr u32 kQPrune = 1u << 8;        // SEE and delta pruning in quiescence
constexpr u32 kPvs = 1u << 9;           // principal variation search: zero windows after a node's first move
constexpr u32 kQChecks = 1u << 10;      // quiet checks at quiescence's first ply
constexpr u32 kTTAll = kTT | kTTCutoffs | kTTQuiescence;
}  // namespace ft

// A feature set's techniques as the compile-time constants the search tests
// (Features::kTT, ...).
template <u32 kFlags>
struct FeatureFlags {
    static constexpr u32 kAll = kFlags;
    static constexpr bool kTT = (kFlags & ft::kTT) != 0;
    static constexpr bool kTTCutoffs = (kFlags & ft::kTTCutoffs) != 0;
    static constexpr bool kTTQuiescence = (kFlags & ft::kTTQuiescence) != 0;
    static constexpr bool kNullMove = (kFlags & ft::kNullMove) != 0;
    static constexpr bool kLmr = (kFlags & ft::kLmr) != 0;
    static constexpr bool kFutility = (kFlags & ft::kFutility) != 0;
    static constexpr bool kCheckExt = (kFlags & ft::kCheckExt) != 0;
    static constexpr bool kKillers = (kFlags & ft::kKillers) != 0;
    static constexpr bool kQPrune = (kFlags & ft::kQPrune) != 0;
    static constexpr bool kPvs = (kFlags & ft::kPvs) != 0;
    static constexpr bool kQChecks = (kFlags & ft::kQChecks) != 0;
    static_assert(kTT || (!kTTCutoffs && !kTTQuiescence), "table options without a table");
};

// One struct per listed set (FeaturesNone, FeaturesTT, ...), with its name.
#define CB_DEFINE_FEATURE_SET(Kind, Type, name, flags)    \
    struct Type : FeatureFlags<(flags)> {                 \
        static constexpr const char* kName = name;        \
    };
CB_FEATURE_SETS(CB_DEFINE_FEATURE_SET)
#undef CB_DEFINE_FEATURE_SET

// Run-time settings of the pruning techniques. Feature sets without a
// technique ignore its settings.
struct PruneParams {
    // Null move: tried at nodes with at least null_min_depth plies left, and
    // searched with depth - 1 - R, R = null_base + depth / null_divisor.
    u8 null_min_depth = 2;
    u8 null_base = 3;
    u8 null_divisor = 6;  // 0: R = null_base at every depth
    // Late move reductions (see LmrTable): at nodes with at least
    // lmr_min_depth plies left, quiet moves after the first lmr_moves are
    // searched R plies shallower, R = lmr_base / 100 + ln(depth) ln(move
    // number) / (lmr_divisor / 100), rounded down.
    u8 lmr_min_depth = 3;
    u8 lmr_moves = 3;
    u8 lmr_base = 75;
    u8 lmr_divisor = 225;
    // Futility pruning: at nodes with 1 to fut_depth plies left, quiet moves
    // that don't give check are skipped when the static evaluation plus
    // fut_margin per ply left can't reach alpha.
    u8 fut_depth = 3;
    u8 fut_margin = 150;
    // Quiescence pruning (core/see.h), at quiescence nodes not in check:
    // delta pruning skips a capture or promotion when the stand pat plus its
    // material gain plus delta_margin can't reach alpha (0: off); SEE
    // pruning (see_prune 1) skips one that loses material in the exchange on
    // its square.
    u16 delta_margin = 200;
    u8 see_prune = 1;
};

// Ranges the host accepts (validate_prune, UCI options).
constexpr int kMaxNullMinDepth = 20;
constexpr int kMaxNullBase = 10;
constexpr int kMaxNullDivisor = 20;
constexpr int kMaxLmrMinDepth = 20;
constexpr int kMaxLmrMoves = 63;
constexpr int kMaxFutDepth = 10;
constexpr int kMaxDeltaMargin = 2000;

// The null move's reduction at `depth`.
CB_HD int null_move_reduction(const PruneParams& p, int depth) {
    return p.null_base + (p.null_divisor ? depth / p.null_divisor : 0);
}

// Late move reductions by depth left and move number (1 for the first move
// searched at the node), both capped at 63. Built on the host from
// PruneParams (make_lmr_table), so searches only look reductions up: no
// floating point on the GPU, where contracted multiply-adds could round
// differently from the CPU. A reduction never takes the move below depth 1.
struct LmrTable {
    u8 r[64][64];

    CB_HD int reduction(int depth, int move_number) const {
        if (depth <= 0) return 0;  // quiescence
        return r[depth < 63 ? depth : 63][move_number < 63 ? move_number : 63];
    }
};

// Host only.
inline LmrTable make_lmr_table(const PruneParams& p) {
    LmrTable t{};
    if (p.lmr_divisor == 0) return t;  // invalid (validate_prune rejects it): no reductions
    for (int d = 0; d < 64; ++d) {
        for (int m = 0; m < 64; ++m) {
            if (d < p.lmr_min_depth || m <= p.lmr_moves || d < 2) continue;
            const double r = p.lmr_base / 100.0 + std::log(double(d)) * std::log(double(m)) / (p.lmr_divisor / 100.0);
            const int ri = int(r);  // rounded down (r >= 0)
            t.r[d][m] = u8(ri < d - 2 ? ri : d - 2);
        }
    }
    return t;
}

}  // namespace cb

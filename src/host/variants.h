#pragma once

// Names of the variant choices the host picks at runtime (see
// docs/ARCHITECTURE.md, Variants and build). Which GPU variants are compiled into a build is
// in gpu/perft_gpu.h (is_compiled).

#include <string>
#include <string_view>

#include "core/features.h"

namespace cb {

enum class BoardKind { Std, Quad };
enum class ExecModel { Thread, Warp };
enum class EvalKind { Material, Pesto, Terms, Nnue };
enum class OrderingKind { Staged, List };
enum class StackKind { Local, Shared };
// Search feature sets (core/feature_list.h) and transposition table layouts (core/tt.h).
#define CB_FEATURE_KIND(Kind, Type, name, flags) Kind,
enum class FeatureKind { CB_FEATURE_SETS(CB_FEATURE_KIND) };
#undef CB_FEATURE_KIND
enum class TTLayoutKind { Packed64, Wide128 };
// Sliding-piece attacks (core/attacks.h, attacks_magic.h, attacks_kogge.h):
// a perft variant; the searches use hyperbola quintessence.
enum class AttacksKind { Hq, Magic, Kogge };

// Move encoding of the perft kernels and the CPU perft (core/move32.h;
// docs/ARCHITECTURE.md, Move generation): 16 bits, or 32 with the moving and captured
// pieces. The search keeps 16-bit moves.
enum class MovesKind { Bits16, Bits32 };

template <class F>
struct FeaturesTag {
    using type = F;
};

// Every feature set, in the list's order.
#define CB_FEATURE_KIND_VALUE(Kind, Type, name, flags) FeatureKind::Kind,
inline constexpr FeatureKind kAllFeatureKinds[] = {CB_FEATURE_SETS(CB_FEATURE_KIND_VALUE)};
#undef CB_FEATURE_KIND_VALUE

// Calls fn(FeaturesTag<F>{}), F being feature set f's struct (core/features.h).
template <class Fn>
constexpr decltype(auto) visit_features(FeatureKind f, Fn&& fn) {
    switch (f) {
#define CB_VISIT_FEATURES_CASE(Kind, Type, name, flags) \
    case FeatureKind::Kind:                             \
        return fn(FeaturesTag<Type>{});
        CB_FEATURE_SETS(CB_VISIT_FEATURES_CASE)
#undef CB_VISIT_FEATURES_CASE
    }
    return fn(FeaturesTag<FeaturesNone>{});  // not reached
}

// The techniques feature set f uses.
constexpr bool uses_tt(FeatureKind f) {
    return visit_features(f, [](auto t) { return decltype(t)::type::kTT; });
}
constexpr bool uses_null_move(FeatureKind f) {
    return visit_features(f, [](auto t) { return decltype(t)::type::kNullMove; });
}
constexpr bool uses_lmr(FeatureKind f) {
    return visit_features(f, [](auto t) { return decltype(t)::type::kLmr; });
}
constexpr bool uses_futility(FeatureKind f) {
    return visit_features(f, [](auto t) { return decltype(t)::type::kFutility; });
}
constexpr bool uses_check_extensions(FeatureKind f) {
    return visit_features(f, [](auto t) { return decltype(t)::type::kCheckExt; });
}
constexpr bool uses_killers(FeatureKind f) {
    return visit_features(f, [](auto t) { return decltype(t)::type::kKillers; });
}
constexpr bool uses_qprune(FeatureKind f) {
    return visit_features(f, [](auto t) { return decltype(t)::type::kQPrune; });
}

// The settings of the techniques feature set f uses, for descriptions: ""
// or e.g. " null 3+d/6 from 2" (R = 3 + depth / 6, at depth 2 and up) and
// " lmr 75/225 from 3 after 3" (R = 0.75 + ln(depth) ln(move) / 2.25, at
// depth 3 and up, after the first 3 moves) and " fut 150 to 3" (a margin of
// 150 per ply left, up to 3 plies).
std::string describe_prune(FeatureKind f, const PruneParams& p);

// Throws std::invalid_argument if a setting is out of range.
void validate_prune(const PruneParams& p);

const char* name_of(BoardKind k);
const char* name_of(ExecModel m);
const char* name_of(EvalKind e);
const char* name_of(OrderingKind o);
const char* name_of(StackKind s);
const char* name_of(FeatureKind f);
const char* name_of(TTLayoutKind l);
const char* name_of(AttacksKind a);
const char* name_of(MovesKind m);  // "16", "32"

// Each returns false (leaving `out` unchanged) if `text` names no choice.
bool parse_board_kind(std::string_view text, BoardKind& out);
bool parse_exec_model(std::string_view text, ExecModel& out);
bool parse_eval_kind(std::string_view text, EvalKind& out);
bool parse_ordering_kind(std::string_view text, OrderingKind& out);
bool parse_stack_kind(std::string_view text, StackKind& out);
bool parse_feature_kind(std::string_view text, FeatureKind& out);
bool parse_tt_layout_kind(std::string_view text, TTLayoutKind& out);
bool parse_attacks_kind(std::string_view text, AttacksKind& out);
bool parse_moves_kind(std::string_view text, MovesKind& out);

}  // namespace cb

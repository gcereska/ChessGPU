#include "host/variants.h"

#include <stdexcept>

namespace cb {

const char* name_of(BoardKind k) { return k == BoardKind::Std ? "std" : "quad"; }
const char* name_of(ExecModel m) { return m == ExecModel::Thread ? "thread" : "warp"; }
const char* name_of(EvalKind e) {
    switch (e) {
        case EvalKind::Material: return "material";
        case EvalKind::Pesto: return "pesto";
        case EvalKind::Terms: return "terms";
        case EvalKind::Nnue: return "nnue";
    }
    return "?";
}
const char* name_of(OrderingKind o) { return o == OrderingKind::Staged ? "staged" : "list"; }
const char* name_of(StackKind s) { return s == StackKind::Local ? "local" : "shared"; }
const char* name_of(FeatureKind f) {
    return visit_features(f, [](auto t) { return decltype(t)::type::kName; });
}
const char* name_of(TTLayoutKind l) { return l == TTLayoutKind::Packed64 ? "packed64" : "wide128"; }

bool parse_board_kind(std::string_view text, BoardKind& out) {
    if (text == "std") out = BoardKind::Std;
    else if (text == "quad") out = BoardKind::Quad;
    else return false;
    return true;
}

bool parse_exec_model(std::string_view text, ExecModel& out) {
    if (text == "thread") out = ExecModel::Thread;
    else if (text == "warp") out = ExecModel::Warp;
    else return false;
    return true;
}

bool parse_eval_kind(std::string_view text, EvalKind& out) {
    if (text == "material") out = EvalKind::Material;
    else if (text == "pesto") out = EvalKind::Pesto;
    else if (text == "terms") out = EvalKind::Terms;
    else if (text == "nnue") out = EvalKind::Nnue;
    else return false;
    return true;
}

bool parse_ordering_kind(std::string_view text, OrderingKind& out) {
    if (text == "staged") out = OrderingKind::Staged;
    else if (text == "list") out = OrderingKind::List;
    else return false;
    return true;
}

bool parse_stack_kind(std::string_view text, StackKind& out) {
    if (text == "local") out = StackKind::Local;
    else if (text == "shared") out = StackKind::Shared;
    else return false;
    return true;
}

bool parse_feature_kind(std::string_view text, FeatureKind& out) {
    for (const FeatureKind f : kAllFeatureKinds) {
        if (text == name_of(f)) {
            out = f;
            return true;
        }
    }
    return false;
}

std::string describe_prune(FeatureKind f, const PruneParams& p) {
    std::string s;
    if (uses_null_move(f)) {
        s += " null " + std::to_string(p.null_base);
        if (p.null_divisor) s += "+d/" + std::to_string(p.null_divisor);
        s += " from " + std::to_string(p.null_min_depth);
    }
    if (uses_lmr(f))
        s += " lmr " + std::to_string(p.lmr_base) + "/" + std::to_string(p.lmr_divisor) + " from " +
             std::to_string(p.lmr_min_depth) + " after " + std::to_string(p.lmr_moves);
    if (uses_futility(f)) s += " fut " + std::to_string(p.fut_margin) + " to " + std::to_string(p.fut_depth);
    if (uses_qprune(f)) {
        s += p.delta_margin ? " delta " + std::to_string(p.delta_margin) : std::string(" no delta");
        s += p.see_prune ? " see" : " no see";
    }
    return s;
}

void validate_prune(const PruneParams& p) {
    if (p.null_min_depth < 1 || p.null_min_depth > kMaxNullMinDepth)
        throw std::invalid_argument("null move minimum depth must be 1 to " + std::to_string(kMaxNullMinDepth));
    if (p.null_base < 1 || p.null_base > kMaxNullBase)
        throw std::invalid_argument("null move reduction must be 1 to " + std::to_string(kMaxNullBase));
    if (p.null_divisor > kMaxNullDivisor)
        throw std::invalid_argument("null move depth divisor must be 0 to " + std::to_string(kMaxNullDivisor));
    if (p.lmr_min_depth < 1 || p.lmr_min_depth > kMaxLmrMinDepth)
        throw std::invalid_argument("late move reduction minimum depth must be 1 to " +
                                    std::to_string(kMaxLmrMinDepth));
    if (p.lmr_moves > kMaxLmrMoves)
        throw std::invalid_argument("late move reductions: moves before them must be 0 to " +
                                    std::to_string(kMaxLmrMoves));
    if (p.lmr_divisor < 1) throw std::invalid_argument("late move reduction divisor must be 1 to 255");
    if (p.fut_depth < 1 || p.fut_depth > kMaxFutDepth)
        throw std::invalid_argument("futility pruning depth must be 1 to " + std::to_string(kMaxFutDepth));
    if (p.delta_margin > kMaxDeltaMargin)
        throw std::invalid_argument("delta pruning margin must be 0 to " + std::to_string(kMaxDeltaMargin));
    if (p.see_prune > 1) throw std::invalid_argument("SEE pruning must be on (1) or off (0)");
}

const char* name_of(AttacksKind a) {
    switch (a) {
        case AttacksKind::Hq: return "hq";
        case AttacksKind::Magic: return "magic";
        case AttacksKind::Kogge: return "kogge";
    }
    return "?";
}

const char* name_of(MovesKind m) { return m == MovesKind::Bits32 ? "32" : "16"; }

bool parse_moves_kind(std::string_view text, MovesKind& out) {
    if (text == "16") out = MovesKind::Bits16;
    else if (text == "32") out = MovesKind::Bits32;
    else return false;
    return true;
}

bool parse_attacks_kind(std::string_view text, AttacksKind& out) {
    for (const AttacksKind a : {AttacksKind::Hq, AttacksKind::Magic, AttacksKind::Kogge}) {
        if (text == name_of(a)) {
            out = a;
            return true;
        }
    }
    return false;
}

bool parse_tt_layout_kind(std::string_view text, TTLayoutKind& out) {
    if (text == "packed64") out = TTLayoutKind::Packed64;
    else if (text == "wide128") out = TTLayoutKind::Wide128;
    else return false;
    return true;
}

}  // namespace cb

#pragma once

// NNUE evaluation (docs/ARCHITECTURE.md, Evaluation): a small
// network whose first layer is updated incrementally in make/unmake, like
// PeSTO's tables, so the search's eval interface doesn't change (core/eval.h).
//
// Network: 768 inputs (piece type x color x square, seen from each side:
// "ours"/"theirs", squares mirrored vertically for black) -> kNnueHidden
// neurons per side, sharing weights -> clipped ReLU -> one output from both
// halves, the side to move's first. Integer arithmetic only, so the CPU and
// every GPU variant compute exactly the same score, and a position and its
// color-flipped twin score exactly alike.
//
// Quantization: first-layer weights and biases are int16 at scale kNnueQA
// (1.0 = 255), activations clip to [0, kNnueQA]; output weights int16 at
// scale kNnueQB (1.0 = 64), output bias int32 at kNnueQA * kNnueQB. The float
// network (host/nnue_train.h) outputs centipawns / kNnueOutputScale.

#include "core/types.h"

namespace cb {

constexpr int kNnueInputs = 768;
constexpr int kNnueHidden = 32;
constexpr int kNnueQA = 255;
constexpr int kNnueQB = 64;
constexpr int kNnueOutputScale = 400;  // centipawns per unit of the float network's output

struct NnueWeights {
    i16 w1[kNnueInputs][kNnueHidden];  // first layer: one row per input
    i16 b1[kNnueHidden];
    i16 w2[2][kNnueHidden];  // output layer: [0] the side to move's half, [1] the other side's
    i32 b2;
};

// The input index of piece (c, pt) on s seen from `perspective`.
CB_HD constexpr int nnue_feature(Color perspective, Color c, PieceType pt, Square s) {
    const int side = c == perspective ? 0 : 1;
    const int sq = perspective == WHITE ? int(s) : (int(s) ^ 56);
    return (side * 6 + (int(pt) - 1)) * 64 + sq;
}

CB_HD constexpr int nnue_clip(int x) { return x < 0 ? 0 : (x > kNnueQA ? kNnueQA : x); }

// The engine's default weights (host/nnue_default.cpp, written by chessgpu
// train-nnue), copied into EvalTables::nnue by init_eval_tables.
extern const NnueWeights kDefaultNnue;

// The evaluation policy (core/eval.h); the weights live in EvalTables::nnue.
struct NnueEval {
    static constexpr const char* kName = "nnue";

    // First-layer sums without the biases (added at evaluation), from each
    // side's point of view: v[WHITE] sees white's pieces as "ours". It also
    // remembers the weights it was built from (every add and remove sets
    // them), so evaluate needs no tables, as with the other policies.
    struct Accumulator {
        i16 v[2][kNnueHidden] = {};
        const NnueWeights* weights = nullptr;

        CB_HD bool operator==(const Accumulator& o) const {
            for (int p = 0; p < 2; ++p)
                for (int i = 0; i < kNnueHidden; ++i)
                    if (v[p][i] != o.v[p][i]) return false;
            return true;
        }
    };

    template <class Tables>
    CB_HD static void add(Accumulator& acc, const Tables& t, Color c, PieceType pt, Square s) {
        acc.weights = &t.nnue;
        const i16* w = t.nnue.w1[nnue_feature(WHITE, c, pt, s)];
        const i16* b = t.nnue.w1[nnue_feature(BLACK, c, pt, s)];
        CB_UNROLL
        for (int i = 0; i < kNnueHidden; ++i) {
            acc.v[WHITE][i] = i16(acc.v[WHITE][i] + load_readonly(&w[i]));
            acc.v[BLACK][i] = i16(acc.v[BLACK][i] + load_readonly(&b[i]));
        }
    }
    template <class Tables>
    CB_HD static void remove(Accumulator& acc, const Tables& t, Color c, PieceType pt, Square s) {
        acc.weights = &t.nnue;
        const i16* w = t.nnue.w1[nnue_feature(WHITE, c, pt, s)];
        const i16* b = t.nnue.w1[nnue_feature(BLACK, c, pt, s)];
        CB_UNROLL
        for (int i = 0; i < kNnueHidden; ++i) {
            acc.v[WHITE][i] = i16(acc.v[WHITE][i] - load_readonly(&w[i]));
            acc.v[BLACK][i] = i16(acc.v[BLACK][i] - load_readonly(&b[i]));
        }
    }

    // The network's output for the side to move, in centipawns.
    CB_HD static int evaluate(const Accumulator& acc, Color side_to_move) {
        const NnueWeights& n = *acc.weights;
        i32 out = load_readonly(&n.b2);
        CB_UNROLL
        for (int i = 0; i < kNnueHidden; ++i) {
            const int bias = load_readonly(&n.b1[i]);
            out += nnue_clip(acc.v[side_to_move][i] + bias) * int(load_readonly(&n.w2[0][i]));
            out += nnue_clip(acc.v[~side_to_move][i] + bias) * int(load_readonly(&n.w2[1][i]));
        }
        // Rounds toward zero; the color-flipped position computes the same sum.
        return int(i64(out) * kNnueOutputScale / (kNnueQA * kNnueQB));
    }
};

}  // namespace cb

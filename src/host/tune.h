#pragma once

// Texel tuning of the evaluation terms (core/eval_terms.h): fit the terms'
// middlegame and endgame weights so that a logistic curve of the evaluation
// predicts game results (docs/ARCHITECTURE.md, Evaluation). The terms are linear in
// their weights, so each position becomes a fixed base (PeSTO's tapered
// score) plus a feature vector, and the fit is plain gradient descent, with
// an L2 pull toward the starting weights against overfitting a small set.

#include <string>
#include <vector>

#include "core/eval_terms.h"
#include "host/position.h"

namespace cb {

constexpr int kTuneWeights = 2 * kTermCount;  // mg weights, then eg weights

struct TuningPosition {
    float base;                    // PeSTO's tapered score, white's point of view
    float features[kTuneWeights];  // count * mg phase / 24, then count * eg phase / 24 (white minus black)
    float result;                  // 1, 0.5 or 0 for white
};

using TuningPositions = std::vector<TuningPosition>;

struct TuningSet {
    TuningPositions training;
    TuningPositions validation;  // every fifth game (when the file numbers its games)
    size_t lines = 0;            // positions read
    size_t rejected = 0;         // not quiet: in check, or a capture that wins material
};

// The quiet positions of a file of "<FEN> | <result> [| <game>]" lines
// (scripts/make_tuning_set.py). Throws std::runtime_error on a bad line.
TuningSet load_tuning_set(const std::string& path);

// One position's features (exposed for tests).
TuningPosition tuning_position(const Position& p, float result);

struct TuneWeights {
    double w[kTuneWeights];
};

// The weights term_weight() holds now.
TuneWeights current_term_weights();

// Mean squared error between the results and 1 / (1 + 10^(-k * eval / 400)).
double tuning_loss(const TuningPositions& set, const TuneWeights& weights, double k);
// The gradient of tuning_loss with respect to the weights.
void tuning_gradient(const TuningPositions& set, const TuneWeights& weights, double k, TuneWeights& out);
// The k that minimizes the loss for these weights (golden-section search).
double fit_scale(const TuningPositions& set, const TuneWeights& weights);

struct TuneOptions {
    int iterations = 2000;
    double learning_rate = 0.5;  // centipawns per step (Adam)
    double l2 = 0;               // penalty l2 * sum (w - anchor)^2 added to the loss
    TuneWeights anchor{};
};

// Adam from `start`; returns the fitted weights (not rounded).
TuneWeights tune_weights(const TuningPositions& set, double k, const TuneWeights& start, const TuneOptions& options);

// The weights as the body of term_weight()'s switch, rounded to centipawns.
std::string format_term_weights(const TuneWeights& weights);

}  // namespace cb
